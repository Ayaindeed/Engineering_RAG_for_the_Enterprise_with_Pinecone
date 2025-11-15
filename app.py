import streamlit as st
import os
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from rerankers import Reranker, Document

# Configure Streamlit page
st.set_page_config(
    page_title="RAG Q&A Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'openai_client' not in st.session_state:
    st.session_state.openai_client = None
if 'pinecone_index' not in st.session_state:
    st.session_state.pinecone_index = None
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

def initialize_clients(openai_key, pinecone_key):
    """Initialize OpenAI and Pinecone clients"""
    try:
        # Initialize OpenAI client
        openai_client = OpenAI(api_key=openai_key)
        
        # Initialize Pinecone client
        pinecone_client = Pinecone(api_key=pinecone_key)
        
        # Connect to existing index
        index_name = "markdown-chunks"
        index = pinecone_client.Index(index_name)
        
        return openai_client, index, True
    except Exception as e:
        st.error(f"Error initializing clients: {str(e)}")
        return None, None, False

def retrieve_chunks(query_text, openai_client, index, num_chunks=3):
    """Retrieve relevant chunks using vector search and reranking"""
    try:
        # Generate query embedding
        query_embedding = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=query_text
        ).data[0].embedding
        
        # Search in Pinecone (get more results for reranking)
        results = index.query(
            vector=query_embedding, 
            top_k=num_chunks * 2, 
            include_metadata=True
        )
        
        # Rerank results for better relevance
        try:
            ranker = Reranker('flashrank')
            docs = [Document(
                text=chunk['metadata']['chunk_text'], 
                doc_id=i
            ) for i, chunk in enumerate(results.matches)]
            
            reranked_results = ranker.rank(query=query_text, docs=docs)
            chunks = [result.document.text for result in reranked_results.top_k(num_chunks)]
            
        except Exception as e:
            # Fallback to original results if reranking fails
            chunks = [chunk['metadata']['chunk_text'] for chunk in results.matches[:num_chunks]]
        
        return chunks, results.matches[:num_chunks]
        
    except Exception as e:
        st.error(f"Error retrieving chunks: {str(e)}")
        return [], []

def generate_answer(query_text, context, openai_client):
    """Generate answer using OpenAI GPT"""
    try:
        prompt = f"""
        Given the following context, answer the following question comprehensively.
        If the answer is not in the context, say "I don't have enough information to answer this question based on the provided context."
        
        Context: {context}
        
        Question: {query_text}
        
        Answer:"""
        
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context. Be accurate and cite the context when relevant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.1
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
        return "Sorry, I encountered an error while generating the answer."

def main():
    st.title("🤖 RAG Q&A Assistant")
    st.markdown("Ask questions and get AI-powered answers based on your document knowledge base!")
    
    # Sidebar for API keys
    with st.sidebar:
        st.header("🔑 Configuration")
        
        # API Key inputs
        openai_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key")
        pinecone_key = st.text_input("Pinecone API Key", type="password", help="Enter your Pinecone API key")
        
        if st.button("Initialize System"):
            if openai_key and pinecone_key:
                with st.spinner("Initializing..."):
                    openai_client, pinecone_index, success = initialize_clients(openai_key, pinecone_key)
                    if success:
                        st.session_state.openai_client = openai_client
                        st.session_state.pinecone_index = pinecone_index
                        st.session_state.initialized = True
                        st.success("✅ System initialized successfully!")
                    else:
                        st.session_state.initialized = False
            else:
                st.warning("Please enter both API keys")
        
        if st.session_state.initialized:
            st.success("🟢 System Ready")
        else:
            st.info("🔴 Enter API keys and click Initialize")
    
    # Main interface
    if st.session_state.initialized:
        st.markdown("---")
        
        # Question input
        question = st.text_input(
            "💭 Ask your question:",
            placeholder="e.g., How do I create an index in Pinecone?",
            help="Type your question about the documents in the knowledge base"
        )
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            ask_button = st.button("🔍 Get Answer", type="primary")
        
        with col2:
            num_chunks = st.slider("Number of sources", 1, 5, 3, help="How many source chunks to retrieve")
        
        if ask_button and question:
            with st.spinner("🔎 Searching knowledge base..."):
                # Retrieve relevant chunks
                chunks, raw_results = retrieve_chunks(
                    question, 
                    st.session_state.openai_client, 
                    st.session_state.pinecone_index, 
                    num_chunks
                )
                
                if chunks:
                    # Generate answer
                    with st.spinner("🧠 Generating answer..."):
                        context = "\n\n".join(chunks)
                        answer = generate_answer(question, context, st.session_state.openai_client)
                    
                    # Display results
                    st.markdown("## 💡 Answer")
                    st.markdown(answer)
                    
                    # Show sources
                    st.markdown("## 📚 Sources")
                    for i, (chunk, result) in enumerate(zip(chunks, raw_results), 1):
                        with st.expander(f"Source {i} (Score: {result['score']:.3f})"):
                            st.markdown(chunk)
                            if 'metadata' in result and result['metadata']:
                                st.caption(f"Metadata: {result['metadata']}")
                else:
                    st.error("No relevant information found. Please try a different question.")
        
        # Example questions
        st.markdown("---")
        st.markdown("### 💡 Example Questions")
        examples = [
            "How do I create an index in Pinecone?",
            "What is vertical scaling?",
            "How do I manage pod sizes?",
            "What are collections in Pinecone?",
            "How does horizontal scaling work?"
        ]
        
        cols = st.columns(len(examples))
        for i, example in enumerate(examples):
            with cols[i]:
                if st.button(f"📝 {example[:20]}...", key=f"ex_{i}"):
                    st.session_state.example_question = example
    
    else:
        st.info("👆 Please configure your API keys in the sidebar to get started!")
        
        # Instructions
        st.markdown("## 🚀 How to Use")
        st.markdown("""
        1. **Get API Keys:**
           - OpenAI API key from [OpenAI Platform](https://platform.openai.com/api-keys)
           - Pinecone API key from [Pinecone Console](https://app.pinecone.io/)
        
        2. **Enter Keys:** Add them in the sidebar
        
        3. **Initialize:** Click "Initialize System"
        
        4. **Ask Questions:** Type your question and get AI-powered answers!
        """)

if __name__ == "__main__":
    main()