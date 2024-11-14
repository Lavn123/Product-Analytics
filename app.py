import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import base64
import os
import tempfile

# Set page configuration
st.set_page_config(
    page_title="LEXI GENIUS - An LLM Brilliance",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add Google Analytics
st.markdown(
    """
    <!-- Google tag (gtag.js) -->
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-BP1FV8KVQD"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){dataLayer.push(arguments);}
      gtag('js', new Date());
      gtag('config', 'G-BP1FV8KVQD');
    </script>
    """,
    unsafe_allow_html=True,
)

# Load model and tokenizer
checkpoint = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
model = T5ForConditionalGeneration.from_pretrained(checkpoint)

# Function to process the PDF file
def file_preprocessing(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    final_texts = [text.page_content for text in texts]
    return final_texts

# Summarization function
def text_summarization(text, max_length=150, min_length=50):
    input_ids = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=max_length, min_length=min_length, truncation=True)
    summary_ids = model.generate(input_ids)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Function to display PDF
def displayPDF(file_path):
    try:
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
        return pdf_display
    except Exception as e:
        st.error(f"An error occurred while displaying the PDF: {str(e)}")
        return None

# Main application function
def main():
    # Sidebar and main content styling
    st.markdown(
        """
        <style>
            .stApp { background-color: black; }
            .stMarkdown { color: white; }
            .stProgress > div > div { background-color: #4CAF50; }
            .stButton { background-color: #008CBA; color: white; }
            .stButton:hover { background-color: #005682; }
            .stSuccess { color: #4CAF50; }
            .stError { color: #FF0000; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Layout setup
    col1, col2 = st.columns([1, 3])
    col1.image("https://www.shutterstock.com/image-vector/chatbot-icon-concept-chat-bot-600nw-2132342911.jpg", width=150)
    col2.title("Lexi Genius || An Summarization Brilliance")

    # Instructions and file upload
    col2.markdown(
        """
        Welcome to Lexi Genius! This tool allows you to upload a PDF file and generates a summary.
        
        Follow these simple steps:
        
        1. Upload your PDF file using the 'Upload your PDF file' button below.
        2. Adjust summarization parameters (if needed).
        3. Click the 'Generate Summary' button to generate a summary.
        """
    )
    uploaded_file = col2.file_uploader("Choose a PDF file to summarize", type=['pdf'])

    if uploaded_file is not None:
        col2.write(f"Selected File: {uploaded_file.name}")
        max_length = col2.slider("Maximum Summary Length", min_value=50, max_value=300, value=150)

        if col2.button("Generate Summary"):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name

            # Display PDF if possible
            pdf_display = displayPDF(temp_file_path)
            if pdf_display:
                col2.markdown(pdf_display, unsafe_allow_html=True)

            # Summarization progress and display
            with col2:
                progress_bar = st.progress(0)
                with st.spinner("Summarizing..."):
                    try:
                        input_texts = file_preprocessing(temp_file_path)
                        summaries = []
                        for i, text_chunk in enumerate(input_texts):
                            chunk_summary = text_summarization(text_chunk, max_length=max_length)
                            summaries.append(chunk_summary)
                            progress_bar.progress((i + 1) / len(input_texts))

                        summary_text = "\n\n".join(summaries)
                        st.success("Summarization Complete")

                        # Display the summary as Markdown
                        st.markdown("### Generated Summary:")
                        st.markdown(summary_text, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error during summarization: {e}")
                    finally:
                        os.remove(temp_file_path)

if __name__ == "__main__":
    main()
