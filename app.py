import streamlit as st
import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize
from summarizer import Summarizer

# Download the 'punkt' tokenizer models
nltk.download('punkt')

def extract_and_summarize(pdf_path):
    pdf = PyPDF2.PdfReader(pdf_path)
    full_text = ""

    for page_num in range(len(pdf.pages)):
        page = pdf.pages[page_num]
        text = page.extract_text()
        full_text += text

    return full_text

def main():
    st.title("PDF Summarizer")

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        st.sidebar.header("Summarization Settings")

        # Explanation of Summary Ratio
        st.sidebar.write("Summary Ratio controls the length of the summary relative to the original text.")
        st.sidebar.write("A lower ratio results in a more concise summary, while a higher ratio includes more content.")

        ratio = st.sidebar.slider("Summary Ratio", 0.1, 1.0, 0.2)

        full_text = extract_and_summarize(uploaded_file)
        sentences = sent_tokenize(full_text)
        input_text = " ".join(sentences)

        try:
            # Use the bert-extractive-summarizer as an alternative
            from summarizer import Summarizer
            model = Summarizer()
            summarized_text = model(input_text, ratio=ratio)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return

        # Display the summarized text with improved readability
        st.subheader("Summarized Text:")
        st.write(summarized_text)

if __name__ == "__main__":
    main()
