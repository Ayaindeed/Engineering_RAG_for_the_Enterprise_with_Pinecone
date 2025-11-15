# Enterprise RAG Q&A Assistant

## About This Project

This project demonstrates an enterprise-quality Retrieval Augmented Generation (RAG) application built with Pinecone and OpenAI. RAG is a critical component of search and chatbots that uses vector databases to retrieve facts and improve response quality. 

The system showcases enterprise RAG architecture including data management, access control, and performance monitoring considerations that are essential in production environments.

**Inspired by:** This implementation was inspired by the code-along video on DataCamp by Roie Schwaber-Cohen, Staff Developer Advocate at Pinecone, demonstrating how to build enterprise-ready RAG applications with proper architecture, namespaces, data privacy, and access control.

## What This System Does

This RAG application processes documentation (Pinecone docs), chunks the text intelligently, generates embeddings using OpenAI, stores them in Pinecone vector database, and provides a clean web interface for asking questions. The system retrieves relevant context and generates accurate answers with source references.

## Key Features

## Configuration

### Required API Keys:
- **OpenAI API Key:** Get from [OpenAI Platform](https://platform.openai.com/api-keys)
- **Pinecone API Key:** Get from [Pinecone Console](https://app.pinecone.io/)


## Features

- ✅ Enterprise-grade RAG architecture
- ✅ Intelligent document chunking and embedding
- ✅ Vector similarity search with reranking
- ✅ Clean, professional web interface
- ✅ Real-time question answering with context
- ✅ Source document references and citations
- ✅ Adjustable retrieval parameters
- ✅ Example questions for guidance
- ✅ Error handling and user feedback

## Technology Stack

- **Frontend:** Streamlit web interface
- **Vector Database:** Pinecone for similarity search
- **LLM:** OpenAI GPT-4 for answer generation
- **Embeddings:** OpenAI text-embedding-ada-002
- **Reranking:** FlashRank for improved relevance
