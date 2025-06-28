import streamlit as st
import math
import networkx as nx
import numpy as np
from nltk.tokenize.punkt import PunktSentenceTokenizer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from googletrans import Translator
from langdetect import detect
import zipfile
import io

# Set page config
st.set_page_config(page_title="Multilingual Summarizer", layout="wide")

# Language options (matching your notebook)
LANGUAGES = {
    'English': 'en',
    'Hindi': 'hi',
    'Marathi': 'mr',
    'Gujarati': 'gu',
    'Italian': 'it',
    'Dutch': 'nl',
    'Chinese (Simplified)': 'zh-cn',
    'French': 'fr',
    'Japanese': 'ja',
    'Kannada': 'kn'
}

# Check for docx module
try:
    from docx import Document
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False
    st.warning("Note: Full DOCX support requires 'python-docx' package. Install with: pip install python-docx")

# Title and description
st.title("🌍 Multilingual Text Summarizer")
st.markdown("""
This app summarizes text and provides translations between multiple languages.
Supports plain text (.txt) and Word documents (.docx).
""")

# Fallback DOCX reader
def read_docx_fallback(file):
    try:
        with zipfile.ZipFile(file) as z:
            with z.open('word/document.xml') as f:
                content = f.read().decode('utf-8')
                import re
                return ' '.join(re.findall(r'<w:t>([^<]+)</w:t>', content))
    except Exception as e:
        st.error(f"Error reading DOCX file: {str(e)}")
        return ""

# Function to extract text from DOCX
def read_docx(file):
    if DOCX_SUPPORT:
        try:
            doc = Document(file)
            return "\n".join([para.text for para in doc.paragraphs if para.text])
        except Exception as e:
            st.error(f"Error reading DOCX with python-docx: {str(e)}")
            return read_docx_fallback(file)
    else:
        return read_docx_fallback(file)

# TextRank summarization function (from your notebook)
def textrank(document, num_sentences=5):
    sentence_tokenizer = PunktSentenceTokenizer()
    sentences = sentence_tokenizer.tokenize(document)

    bow_matrix = CountVectorizer().fit_transform(sentences)
    normalized = TfidfTransformer().fit_transform(bow_matrix)

    similarity_graph = normalized * normalized.T
    nx_graph = nx.from_scipy_sparse_array(similarity_graph)
    scores = nx.pagerank(nx_graph)
    sentence_array = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    
    sentence_array = np.asarray(sentence_array)
    fmax = float(sentence_array[0][0])
    fmin = float(sentence_array[len(sentence_array) - 1][0])
    
    temp_array = []
    for i in range(0, len(sentence_array)):
        if fmax - fmin == 0:
            temp_array.append(0)
        else:
            temp_array.append((float(sentence_array[i][0]) - fmin) / (fmax - fmin))

    threshold = (sum(temp_array) / len(temp_array)) + 0.2
    
    sentence_list = []
    for i in range(0, len(temp_array)):
        if temp_array[i] > threshold:
            sentence_list.append(sentence_array[i][1])

    seq_list = []
    for sentence in sentences:
        if sentence in sentence_list:
            seq_list.append(sentence)
    
    return seq_list[:num_sentences]

# Sidebar for options
with st.sidebar:
    st.header("⚙️ Options")
    summary_length = st.slider("Summary Length (sentences)", 3, 10, 5)
    auto_detect = st.checkbox("Auto-detect language", value=True)
    
    st.header("🌐 Language Selection")
    target_language = st.selectbox(
        "Select target language for translation:",
        options=list(LANGUAGES.keys()),
        index=1  # Default to Hindi
    )

# Text input area
input_text = st.text_area("✍️ Enter your text here:", height=200)

# File upload option
st.subheader("📤 Or upload a file")
uploaded_file = st.file_uploader("Choose a file", type=["txt", "docx"], label_visibility="collapsed")

if uploaded_file is not None:
    if uploaded_file.name.endswith('.docx'):
        input_text = read_docx(uploaded_file)
    else:
        input_text = str(uploaded_file.read(), "utf-8")
    st.text_area("📄 File content:", input_text, height=200)

# Process button
if st.button("✨ Summarize and Translate"):
    if not input_text or input_text.strip() == "":
        st.warning("Please enter some text to summarize or upload a file.")
    else:
        with st.spinner("Processing..."):
            try:
                translator = Translator()
                
                # Detect or set input language
                if auto_detect:
                    try:
                        src_lang = detect(input_text)
                    except:
                        st.error("Could not detect language. Please try again or specify language manually.")
                        st.stop()
                else:
                    # If not auto-detecting, assume English as source
                    src_lang = 'en'
                
                # Summarize the text
                if src_lang == 'en':
                    # If input is English, summarize directly
                    summary = textrank(input_text, summary_length)
                    output_summary = ' '.join(summary)
                else:
                    # For other languages, first translate to English for summarization
                    eng_text = translator.translate(input_text, src=src_lang, dest='en').text
                    eng_summary = textrank(eng_text, summary_length)
                    output_summary = ' '.join(eng_summary)
                
                # Translate to target language
                target_lang_code = LANGUAGES[target_language]
                translated_summary = translator.translate(output_summary, src='en', dest=target_lang_code).text
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("English Summary")
                    st.write(output_summary)
                    st.download_button(
                        label="Download English Summary",
                        data=output_summary,
                        file_name="english_summary.txt",
                        mime="text/plain"
                    )
                
                with col2:
                    st.subheader(f"{target_language} Summary")
                    st.write(translated_summary)
                    st.download_button(
                        label=f"Download {target_language} Summary",
                        data=translated_summary,
                        file_name=f"{target_language.lower()}_summary.txt",
                        mime="text/plain"
                    )
                
                st.success("Processing complete!")
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

# Clear button
if st.button("🧹 Clear"):
    st.session_state.clear()
    st.rerun()
