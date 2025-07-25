
# 📄 PDF Whisperer

PDF Whisperer is an AI-powered chatbot designed to answer customer queries by retrieving relevant information from the
provided pdf. It processes PDF documents, extracts and chunks their content, embeds the data, and utilizes a Large
Language Model (LLM) to provide accurate responses. For this project, a PDF about Occupational Safety on Agriculture and
Safety is used. The document is in German.

---

## 🚀 Features

- **Text Extraction** PyMuPDF is used to extract text from PDF document.
- **Content Chunking** Documents are divided into sections based on the table of contents; large sections are further
  split using semantic chunking.
- **Embedding**: Data is embedded using Hugging Face(Sentence Transformer) multi-lingual model.
- **Search/Retrieval**: LanceDB is used for Search/Retrieval part. Hybrid search combining text and vector search
  is used.
- **LLM Integration**: Deepseek is used to generate responses based on the retrieved data.
- **Chatbot Interface**: Streamlit is used for chatbot interface for users to ask questions and receive answer.

---

## 📁 Project Structure

```plaintext
pdf-whisperer/
├── data/               # Directory for storing PDF files and processed data
├── src/                # Source code for the application
├── tests/              # Test cases and evaluation(In progress)
├── .gitignore          # Specifies files to ignore in version control
├── LICENSE             # MIT License file
└── README.md           # Project documentation
```

---

## 📅 Roadmap for First Cut - Completed

- [x] **Text Extraction**: Implement PDF text extraction using PyMPDF.
- [x] **Content Chunkin**: Develop logic to split documents based on the table of contents and semantic boundaries.
- [x] **Embedding and Search/Retrieval**: Integrate embedding models and set up a search/retrieval mechanism.
- [x] **LLM Integration**: Connect to a Large Language Model for generating responses.
- [x] **Chatbot Interface**: Build a user-friendly chatbot interface for interaction.
