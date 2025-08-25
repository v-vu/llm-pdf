import streamlit as st
st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")

import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from htmlTemplates import css, bot_template, user_template

# --- load env BEFORE reading tokens ---
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# ----------------- helpers -----------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs or []:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += (page.extract_text() or "")
    return text

def get_text_chunks(text):
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    return splitter.split_text(text)

def build_embeddings(provider_label: str):
    if provider_label.startswith("OpenAI"):
        if not OPENAI_KEY:
            raise ValueError("Missing OPENAI_API_KEY in .env for OpenAI embeddings.")
        return OpenAIEmbeddings(model="text-embedding-3-small")
    # Default HF MiniLM
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_llm(provider_label: str):
    """
    Choose an LLM that won’t 404 or hit provider/task mismatches.
    - OpenAI gpt-4o-mini (chat)
    - HF FLAN‑T5 (seq2seq, text2text-generation)  <-- very stable via HF Inference
    - HF Qwen2.5-7B-Instruct (causal, text-generation)  <-- good causal fallback
    """
    if provider_label == "OpenAI: gpt-4o-mini":
        if not OPENAI_KEY:
            raise ValueError("Missing OPENAI_API_KEY in .env")
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    if provider_label == "HF: FLAN-T5 (text2text)":
        if not HF_TOKEN:
            raise ValueError("Missing HUGGINGFACEHUB_API_TOKEN in .env")
        # seq2seq → text2text-generation (don’t force provider)
        return HuggingFaceEndpoint(
            repo_id="google/flan-t5-large",
            task="text2text-generation",
            huggingfacehub_api_token=HF_TOKEN,
            temperature=0.5,
            max_new_tokens=512,
            do_sample=False,
        )

    if provider_label == "HF: Qwen2.5-7B-Instruct (text-generation)":
        if not HF_TOKEN:
            raise ValueError("Missing HUGGINGFACEHUB_API_TOKEN in .env")
        # causal → text-generation (don’t force provider to avoid router 404s)
        return HuggingFaceEndpoint(
            repo_id="Qwen/Qwen2.5-7B-Instruct",
            task="text-generation",
            huggingfacehub_api_token=HF_TOKEN,
            temperature=0.3,
            max_new_tokens=512,
            do_sample=False,
        )

    raise ValueError(f"Unknown LLM option: {provider_label}")

def build_conversation_chain(vectorstore, llm):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory, verbose=True
    )

def handle_userinput(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]
    for i, message in enumerate(st.session_state.chat_history):
        tpl = user_template if i % 2 == 0 else bot_template
        st.write(tpl.replace("{{MSG}}", message.content), unsafe_allow_html=True)

# ----------------- UI -----------------
def main():
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")

    with st.sidebar:
        st.subheader("Options")
        emb_choice = st.selectbox(
            "Embeddings",
            ["HF MiniLM (default)", "OpenAI (text-embedding-3-small)"],
            index=0
        )
        # Default to OpenAI if available; otherwise to FLAN‑T5 (stable HF path)
        default_llm_idx = 0 if OPENAI_KEY else 1
        llm_choice = st.selectbox(
            "LLM Provider",
            ["OpenAI: gpt-4o-mini", "HF: FLAN-T5 (text2text)", "HF: Qwen2.5-7B-Instruct (text-generation)"],
            index=default_llm_idx
        )

        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs then click 'Process'", accept_multiple_files=True)
        if st.button("Process") and pdf_docs:
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                if not raw_text.strip():
                    st.warning("No extractable text found in the PDFs.")
                    st.session_state.conversation = None
                else:
                    chunks = get_text_chunks(raw_text)
                    embeddings = build_embeddings("OpenAI" if emb_choice.startswith("OpenAI") else "HF")
                    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)

                    try:
                        llm = build_llm(llm_choice)
                    except Exception as e:
                        # Fallback: if HF model fails, try OpenAI automatically if key exists
                        if OPENAI_KEY:
                            st.warning(f"LLM build failed ({e}). Falling back to OpenAI gpt-4o-mini.")
                            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
                        else:
                            raise

                    st.session_state.conversation = build_conversation_chain(vectorstore, llm)
                    st.success("Ready! Ask a question in the box above.")

    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        if st.session_state.conversation is None:
            st.info("Upload PDFs and click Process first.")
        else:
            handle_userinput(user_question)

if __name__ == "__main__":
    main()
