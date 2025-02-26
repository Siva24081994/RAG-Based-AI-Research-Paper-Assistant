# DeepDive AI: Research Paper Insights

## Project Description
This project is a **Retrieval-Augmented Generation (RAG) application** that allows users to upload AI research papers in PDF format and ask questions about them. It retrieves relevant sections using **semantic search** and generates context-aware answers using **Hugging Face's Flan-T5 model**.

## Features
- **Paper Upload**: Users can upload AI research papers in PDF format.
- **Query-Based Retrieval**: Implements **FAISS-based semantic search** to fetch relevant sections of the paper.
- **Interactive Q&A**: Provides AI-generated answers to user queries based on paper content.
- **Lightweight & Fast**: Uses **FAISS and Sentence Transformers** for efficient retrieval.

## Tech Stack
### **Backend**
- **Text Extraction**: PyMuPDF (fitz) for extracting structured text from PDFs.
- **Vectorization**: Sentence Transformers (`all-MiniLM-L6-v2`).
- **Vector Database**: FAISS for fast retrieval.
- **LLM Integration**: Hugging Face model (`flan-t5-large`).

### **Frontend**
- **Framework**: Gradio for an interactive web interface.
- **Deployment**: Hugging Face Spaces for easy cloud hosting.

## Installation & Setup
### **1. Clone the Repository**
```sh
 git clone https://github.com/your-username/pdf-rag-bot.git
 cd pdf-rag-bot
```

### **2. Install Dependencies**
```sh
 pip install -r requirements.txt
```

### **3. Run the Application**
```sh
 python app.py
```

## Project Workflow

![image](https://github.com/user-attachments/assets/ca41a60b-8b87-4a5a-aa78-592b0fb16f98)

The biggest advantage of RAG (Retrieval-Augmented Generation) is that it reduces hallucination by retrieving real-world, factual data from external sources before generating an answer. This makes it more reliable compared to purely generative models like GPT or FLAN-T5, which can sometimes produce hypothetical or incorrect responses.



1. **Extract Text**: Extracts text from uploaded PDFs.
2. **Preprocess Data**: Cleans and chunks text for better retrieval.
3. **Embedding Creation**: Converts text into vector embeddings and stores them in FAISS.
4. **Query Processing**: Accepts user queries, retrieves relevant chunks, and generates AI-based answers.
5. **Frontend Interaction**: Users upload PDFs, input queries, and get AI-generated responses in real-time.

## Deployment Guide (Hugging Face Spaces)
### **1. Upload to Hugging Face**
- Create a new **Hugging Face Space**.
- Select **Gradio** as the framework.
- Upload your project files (`app.py`, `requirements.txt`).

### **2. Define Dependencies**
Ensure `requirements.txt` includes:
```sh
faiss-cpu
sentence-transformers
transformers
gradio
p
PyMuPDF
```

### **3. Launch the App**
Once uploaded, the app will automatically deploy in Hugging Face Spaces.

link  ---  https://huggingface.co/spaces/sivapurush/pdf-rag-bot

<img width="1434" alt="Screenshot 2025-01-30 at 11 02 09 PM" src="https://github.com/user-attachments/assets/600c2aa1-76f3-4a23-9e8f-3f173678a5b3" />


## Example Use Case
### **Input:**
"What is the main contribution of the paper?"

### **Process:**
- Retrieves relevant sections from the abstract and conclusion.
- Uses **FLAN-T5** to generate a response based on retrieved context.

### **Output:**
"The main contribution of the paper is the introduction of a novel transformer-based architecture that improves model efficiency by 25% while maintaining state-of-the-art performance on benchmark datasets."

## Contributing
Contributions are welcome! Feel free to submit **pull requests** or open **issues** for improvements.

## License
MIT License - Free to use and modify.

---
🚀 **Developed to empower AI researchers and students with intelligent document understanding.**

Link URL :
https://huggingface.co/spaces/sivapurush/pdf-rag-bot

