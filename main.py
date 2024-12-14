import gradio as gr
from module import preprocessing_pipeline, conversational_rag
from module import system_message, user_message
from haystack.dataclasses import ChatMessage
import time


def process_files_into_docs(pdf_files,progress=gr.Progress()):
  preprocessing_pipeline.run({'file_type_router': {'sources': pdf_files}})
  return "Database created🤗🤗"


def rag(history,question):

  if history is None:
    history=[]
  messages = [system_message, user_message]
  res = conversational_rag.run(
      data = {'query_rephrase_prompt_builder' : {'query': question},
              'prompt_builder': {'template': messages, 'query': question},
              'memory_joiner': {'values': [ChatMessage.from_user(question)]}},
      include_outputs_from=['llm','query_rephrase_llm'])

  bot_message = res['llm']['replies'][0].content

  streamed_message = ""
  for token in bot_message.split():
    streamed_message += f"{token} "
    yield history + [(question, streamed_message.strip())], " " 
    time.sleep(0.05)

  history.append((question,bot_message))

  yield history, " "


with gr.Blocks(theme=gr.themes.Soft(font=gr.themes.GoogleFont('Open Sans')))as demo:
  
  gr.HTML("<center><h1>TalkToFiles - Query your documents! 📂📄</h1><center>") 
  gr.Markdown("""##### This AI chatbot🤖 can help you chat with your documents. Can upload <b>Text(.txt), PDF(.pdf) and Markdown(.md)</b> files.\
              <b>Please do not upload confidential documents.</b>""")
  with gr.Row():
    with gr.Column(scale=86):
      gr.Markdown("""#### ***Step 1 - Upload Documents and Initialize RAG pipeline***</br>
                   Can upload Multiple documents""")
      with gr.Row():
        file_input = gr.File(label='Upload Files', file_count='multiple',file_types=['.pdf', '.txt', '.md'],interactive=True)
      with gr.Row():
        process_files = gr.Button('Create Document store')
      with gr.Row():
        result = gr.Textbox(label="Document store", value='Document store not initialized')
        #Pre-processing Events    
        process_files.click(fn=process_files_into_docs, inputs=file_input, outputs=result ,show_progress=True)


    with gr.Column(scale=200):
      gr.Markdown("""#### ***Step 2 - Chat with your docs*** """)
      chatbot = gr.Chatbot(label='ChatBot')
      user_input = gr.Textbox(label='Enter your query', placeholder='Type here...')
      
      with gr.Row():
        submit_button = gr.Button("Submit")
        clear_btn = gr.ClearButton([user_input, chatbot], value='Clear')
        submit_button.click(rag, inputs=[chatbot, user_input], outputs=[chatbot, user_input])


demo.launch()




