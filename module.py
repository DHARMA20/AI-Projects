
from itertools import chain
from typing import Any, List

from haystack.components.converters import PyPDFToDocument, MarkdownToDocument, TextFileToDocument, OutputAdapter
from haystack.components.routers import FileTypeRouter
from haystack.components.joiners import DocumentJoiner
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.components.builders import ChatPromptBuilder, PromptBuilder
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.core.component.types import Variadic

from haystack_experimental.chat_message_stores.in_memory import InMemoryChatMessageStore
from haystack_experimental.components.retrievers import ChatMessageRetriever
from haystack_experimental.components.writers import ChatMessageWriter
from haystack_integrations.components.generators.cohere import CohereChatGenerator, CohereGenerator


from haystack.dataclasses import ChatMessage
from haystack import Pipeline
from haystack import component

import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Access the API key
os.environ["COHERE_API_KEY"] = os.getenv('COHERE_API_KEY')


document_store = InMemoryDocumentStore()
file_type_router = FileTypeRouter(mime_types=['text/plain','application/pdf','text/markdown'])
pdf_converter = PyPDFToDocument()
text_file_converter = TextFileToDocument()
markdown_converter = MarkdownToDocument()
document_joiner = DocumentJoiner()
document_cleaner = DocumentCleaner()
document_splitter = DocumentSplitter(split_by='word', split_overlap=50)
document_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L12-v2")
document_writer = DocumentWriter(document_store)


preprocessing_pipeline = Pipeline()


# Adding Componenets
preprocessing_pipeline.add_component('file_type_router', file_type_router)
preprocessing_pipeline.add_component('text_file_converter', text_file_converter)
preprocessing_pipeline.add_component('markdown_converter', markdown_converter)
preprocessing_pipeline.add_component('pdf_converter', pdf_converter)
preprocessing_pipeline.add_component('document_joiner', document_joiner)
preprocessing_pipeline.add_component('document_cleaner', document_cleaner)
preprocessing_pipeline.add_component('document_splitter', document_splitter)
preprocessing_pipeline.add_component('document_embedder', document_embedder)
preprocessing_pipeline.add_component('document_writer', document_writer)


# Connections

preprocessing_pipeline.connect('file_type_router.text/plain', 'text_file_converter.sources')
preprocessing_pipeline.connect('file_type_router.application/pdf', 'pdf_converter.sources')
preprocessing_pipeline.connect('file_type_router.text/markdown', 'markdown_converter.sources')
preprocessing_pipeline.connect('text_file_converter', 'document_joiner')
preprocessing_pipeline.connect('markdown_converter', 'document_joiner')
preprocessing_pipeline.connect('pdf_converter', 'document_joiner')
preprocessing_pipeline.connect('document_joiner', 'document_cleaner')
preprocessing_pipeline.connect('document_cleaner', 'document_splitter')
preprocessing_pipeline.connect('document_splitter', 'document_embedder')
preprocessing_pipeline.connect('document_embedder', 'document_writer')


@component
class ListJoiner:
  def __init__(self, _type: Any):
    component.set_output_types(self, values=_type)

  def run(self, values:Variadic[Any]):
    result = list(chain(*values))
    return {'values':result}
  

memory_store = InMemoryChatMessageStore()

query_rephrase_template="""
        Rewrite the question for search while keeping its meaning and key terms intact.
        If the conversation history is empty, DO NOT change the query.
        Use conversation history only if necessary, and avoid extending the query with your own knowledge.
        If no changes are needed, output the current question as is.

        Conversation history:
        {% for memory in memories %}
            {{ memory.content }}
        {% endfor %}

        User Query: {{query}}
        Rewritten Query:
"""


conversational_rag = Pipeline()

#Query rephrasing components
conversational_rag.add_component("query_rephrase_prompt_builder",PromptBuilder(query_rephrase_template))
conversational_rag.add_component('query_rephrase_llm',CohereGenerator())
conversational_rag.add_component('list_to_str_adapter', OutputAdapter(template="{{ replies[0] }}", output_type=str))

#RAG components
conversational_rag.add_component('retriever', InMemoryBM25Retriever(document_store=document_store, top_k=3))
conversational_rag.add_component('prompt_builder', ChatPromptBuilder(variables=["query", "documents", "memories"],required_variables=['query', 'documents', 'memories']))
conversational_rag.add_component('llm', CohereChatGenerator())

#Memory components
conversational_rag.add_component('memory_retriever',ChatMessageRetriever(memory_store))
conversational_rag.add_component('memory_writer', ChatMessageWriter(memory_store))
conversational_rag.add_component('memory_joiner', ListJoiner(List[ChatMessage]))


#Query Rephrasing Connections
conversational_rag.connect('memory_retriever', 'query_rephrase_prompt_builder.memories')
conversational_rag.connect('query_rephrase_prompt_builder.prompt', 'query_rephrase_llm' )
conversational_rag.connect('query_rephrase_llm.replies', 'list_to_str_adapter')
conversational_rag.connect('list_to_str_adapter', 'retriever.query')

#RAG connections
conversational_rag.connect('retriever.documents', 'prompt_builder.documents')
conversational_rag.connect('prompt_builder.prompt', 'llm.messages')
conversational_rag.connect('llm.replies', 'memory_joiner')

#Memory Connections
conversational_rag.connect('memory_joiner','memory_writer')
conversational_rag.connect('memory_retriever','prompt_builder.memories')


system_message = ChatMessage.from_system("""You are an intelligent and cheerful AI assistant specialized in assisting humans with queries based on provided supporting documents and conversation history. 
                                         Always prioritize accurate and concise answers derived from the documents, and offer contextually relevant follow-up questions to maintain an engaging and helpful conversation. 
                                         If the answer is not present in the documents, politely inform the user while suggesting alternative ways to help""")

user_message_template ="""Based on the conversation history and the provided supporting documents, provide a brief and accurate answer to the question.
                          Make the conversation feel more natural and engaging

- Format your response for clarity and readability, using bullet points, paragraphs, or lists where necessary.
- Note: Supporting documents are not part of the conversation history.
- If the question cannot be answered using the supporting documents, respond with: "The answer is not available in the provided documents."

Conversation History:
{% for memory in memories %}
{{ memory.content }}
{% endfor %}

Supporting Documents:
{% for doc in documents %}
{{ doc.content }}
{% endfor %}

Question: {{ query }}
Answer:

"""
user_message = ChatMessage.from_user(user_message_template)

