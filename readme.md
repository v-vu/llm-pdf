# MultiPDF Chat App

## 📖 Introduction

The **MultiPDF Chat App** is a Python application that allows you to **chat with multiple PDF documents**.  
You can ask questions about the PDFs using natural language, and the app will provide relevant responses based on the content of the documents.  

It leverages a language model to generate accurate answers to your queries.  
⚠️ *Note: The app will only respond to questions related to the loaded PDFs.*

---

## 🛠 How It Works

![MultiPDF Chat App Diagram](./PDF-LangChain.jpg)

The application follows these steps to provide responses to your questions:

1. **PDF Loading** – The app reads multiple PDF documents and extracts their text content.  
2. **Text Chunking** – The extracted text is divided into smaller chunks for efficient processing.  
3. **Language Model** – The app generates vector representations (embeddings) of the text chunks.  
4. **Similarity Matching** – When you ask a question, the app finds the most semantically similar chunks.  
5. **Response Generation** – The relevant chunks are passed to the language model, which generates a context-aware response.

---

## 📦 Dependencies and Installation

To install the MultiPDF Chat App, follow these steps:

1. Clone the repository to your local machine.  
   ```bash
   git clone https://github.com/your-username/multipdf-chat-app.git
   cd multipdf-chat-app
   ```

2. Install the required dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

3. Obtain an API key from [OpenAI](https://platform.openai.com/) and add it to the `.env` file in the project directory:  
   ```bash
   OPENAI_API_KEY=your_secret_api_key
   ```

---

## 🚀 Usage

1. Ensure dependencies are installed and the API key is set in `.env`.  
2. Run the app with the Streamlit CLI:  
   ```bash
   streamlit run app.py
   ```
3. The app will launch in your default web browser.  
4. Upload multiple PDF documents via the interface.  
5. Ask questions in natural language about the loaded PDFs using the chat interface.

---

