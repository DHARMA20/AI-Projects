# TalkToFiles: Conversational RAG Chatbot

TalkToFiles is a web application built using **Gradio** and **Haystack**, designed to enable users to upload documents and interact with their content through a conversational AI interface. This chatbot leverages **Retrieval-Augmented Generation (RAG)** to provide accurate and context-aware answers by retrieving information from document content.

---
[App Demo Link](https://huggingface.co/spaces/Dharma20/Talk-to-Files-RAG-ChatBot)
---
## Features

- **Document Upload**: Upload multiple files in PDF, Markdown, or Text format.
- **Document Store Creation**: Automatically preprocess and store document content for querying.
- **Conversational Interface**: Ask questions and get responses based on uploaded document content.
- **Memory Integration**: Maintains context throughout the conversation for more natural interactions.

---

## Tech Stack

- **Gradio**: Provides the web interface for document uploads and chat functionality.
- **Haystack**: Powers the RAG pipeline for document preprocessing and retrieval.
- **Cohere**: Used for query rephrasing and response generation.
- **SentenceTransformers**: For creating document embeddings.

---

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```

2. Navigate to the project directory:
   ```bash
   cd TalkToFiles
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Create a `.env` file in the root directory.
   - Add your API key for Cohere:
     ```
     COHERE_API_KEY=your_api_key_here
     ```

---

## Usage

1. **Start the application**:
   ```bash
   python main.py
   ```

2. **Upload Documents**:
   - Use the interface to upload PDF, Markdown, or Text files.
   - Click the "Create Document Store" button to preprocess and store the documents.

3. **Chat with Documents**:
   - Type your query in the chat interface.
   - Receive contextually accurate responses based on the document content.

---

## File Structure

- `main.py`: Contains the Gradio interface and application logic.
- `module.py`: Defines the pipelines and components for preprocessing, retrieval, and query handling.

---

## Example Workflow

1. Upload a PDF document containing information about Artificial Intelligence.
2. Initialize the document store.
3. Ask the chatbot: "What is Artificial Intelligence?"
4. Receive an accurate, document-based response.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

