import streamlit as st
import pandas as pd
import numpy as np
import json
import io
import base64
import subprocess
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go

# Attempt imports with graceful fallbacks
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    import PyPDF2
    import pytesseract
    from pdf2image import convert_from_bytes
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Configuration
st.set_page_config(
    page_title="BUDDHI-Enterprise_Reasoning_Engine",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .module-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        font-weight: 600;
    }
    .insight-box {
        background: #f0f8ff;
        padding: 1.5rem;
        border-left: 5px solid #1f77b4;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .status-success {
        color: #10b981;
        font-weight: 600;
    }
    .status-warning {
        color: #f59e0b;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'knowledge_base' not in st.session_state:
    st.session_state.knowledge_base = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'embeddings_model' not in st.session_state:
    st.session_state.embeddings_model = None
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# ============================================================================
# EMBEDDING MODULE
# ============================================================================

class EmbeddingEngine:
    """Handles embeddings with fallback strategy"""
    
    def __init__(self):
        self.model = None
        self.embedding_dim = 384  # Default for all-MiniLM-L6-v2
        self.method = self._initialize_model()
    
    def _initialize_model(self) -> str:
        """Initialize embedding model with fallback"""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.embedding_dim = 384
                return "sentence-transformers"
            except Exception as e:
                st.warning(f"Failed to load sentence-transformers: {e}")
        
        # Fallback to hash-based
        self.embedding_dim = 256
        return "hash-based"
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings"""
        if self.method == "sentence-transformers":
            return self.model.encode(texts, convert_to_numpy=True)
        else:
            return np.array([self._hash_embedding(text) for text in texts])
    
    def _hash_embedding(self, text: str) -> np.ndarray:
        """Create hash-based embedding"""
        vector = np.zeros(self.embedding_dim)
        
        # Character frequency features
        for char in text.lower():
            idx = ord(char) % self.embedding_dim
            vector[idx] += 1
        
        # Word-level features
        words = text.lower().split()
        for word in words[:50]:  # Limit to 50 words
            hash_val = int(hashlib.md5(word.encode()).hexdigest(), 16)
            idx = hash_val % self.embedding_dim
            vector[idx] += 2
        
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector

# ============================================================================
# VECTOR STORE MODULE
# ============================================================================

class VectorStore:
    """In-memory vector store with FAISS or NumPy fallback"""
    
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.documents = []
        self.embeddings = None
        self.method = "faiss" if FAISS_AVAILABLE else "numpy"
        
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatL2(embedding_dim)
        else:
            self.index = None
    
    def add_documents(self, docs: List[Dict], embeddings: np.ndarray):
        """Add documents and their embeddings to the store"""
        self.documents.extend(docs)
        
        if self.method == "faiss":
            self.index.add(embeddings.astype('float32'))
        else:
            if self.embeddings is None:
                self.embeddings = embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, embeddings])
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Search for similar documents"""
        if len(self.documents) == 0:
            return []
        
        query_embedding = query_embedding.reshape(1, -1)
        
        if self.method == "faiss":
            distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
            results = []
            for idx, dist in zip(indices[0], distances[0]):
                if idx < len(self.documents):
                    # Convert L2 distance to similarity score
                    similarity = 1 / (1 + dist)
                    results.append((self.documents[idx], similarity))
            return results
        else:
            # NumPy cosine similarity
            similarities = np.dot(self.embeddings, query_embedding.T).flatten()
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            return [(self.documents[idx], similarities[idx]) for idx in top_indices]
    
    def get_stats(self) -> Dict:
        """Get vector store statistics"""
        return {
            'method': self.method,
            'num_documents': len(self.documents),
            'embedding_dim': self.embedding_dim
        }

# ============================================================================
# PDF OCR MODULE
# ============================================================================

class PDFProcessor:
    """Process PDFs with OCR capability"""
    
    @staticmethod
    def extract_text_from_pdf(file) -> str:
        """Extract text from PDF with OCR fallback"""
        text = ""
        
        try:
            # Try standard PDF text extraction first
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            # If no text found, try OCR
            if not text.strip() and OCR_AVAILABLE:
                file.seek(0)
                text = PDFProcessor._ocr_pdf(file)
        
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            
            # Try OCR as fallback
            if OCR_AVAILABLE:
                try:
                    file.seek(0)
                    text = PDFProcessor._ocr_pdf(file)
                except Exception as ocr_error:
                    st.error(f"OCR failed: {ocr_error}")
        
        return text
    
    @staticmethod
    def _ocr_pdf(file) -> str:
        """Perform OCR on PDF"""
        text = ""
        
        try:
            # Convert PDF to images
            file.seek(0)
            images = convert_from_bytes(file.read())
            
            # OCR each page
            for i, image in enumerate(images):
                page_text = pytesseract.image_to_string(image)
                text += f"\n--- Page {i+1} ---\n{page_text}\n"
        
        except Exception as e:
            raise Exception(f"OCR processing failed: {e}")
        
        return text

# ============================================================================
# DOCUMENT CHUNKING MODULE
# ============================================================================

class DocumentChunker:
    """Chunk documents into smaller pieces"""
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
        """Chunk text with overlap"""
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            if chunk_text.strip():
                chunks.append({
                    'text': chunk_text,
                    'chunk_id': len(chunks),
                    'start_idx': i,
                    'word_count': len(chunk_words)
                })
        
        return chunks
    
    @staticmethod
    def chunk_dataframe(df: pd.DataFrame, rows_per_chunk: int = 50) -> List[Dict]:
        """Chunk dataframe into smaller pieces"""
        chunks = []
        
        for i in range(0, len(df), rows_per_chunk):
            chunk_df = df.iloc[i:i + rows_per_chunk]
            
            # Create text representation
            text_repr = chunk_df.to_string(index=False)
            
            chunks.append({
                'text': text_repr,
                'chunk_id': len(chunks),
                'data': chunk_df.to_dict('records'),
                'shape': chunk_df.shape,
                'columns': chunk_df.columns.tolist()
            })
        
        return chunks

# ============================================================================
# OLLAMA LLM MODULE
# ============================================================================

class OllamaLLM:
    """Interface to Ollama local LLM"""
    
    def __init__(self, model_id: str = "llama3.2:latest"):
        self.model_id = model_id
        self.available = self._check_ollama()
    
    def _check_ollama(self) -> bool:
        """Check if Ollama is available"""
        try:
            result = subprocess.run(
                ['ollama', 'list'],
                capture_output=True,
                text=True,
                timeout=5,
                encoding='utf-8',
                errors='ignore'
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def generate(self, prompt: str, context: str = "") -> str:
        """Generate response using Ollama"""
        if not self.available:
            return self._fallback_response(prompt, context)
        
        try:
            # Detect if user wants brief answer
            brief_keywords = ['one line', 'brief', 'short', 'quick', 'simply', 'just tell', 'in short', 'summarize','summary']
            is_brief = any(keyword in prompt.lower() for keyword in brief_keywords)
            
            # Construct full prompt based on request type
            if is_brief:
                full_prompt = f"""You are a financial assistant. Answer in ONE LINE ONLY. Be direct and concise.

Context: {context[:500]}

Question: {prompt}

Answer (one line only):"""
            else:
                full_prompt = f"""You are a financial analysis assistant. Use the following context to answer the question.

Context:
{context}

Question: {prompt}

Provide a clear, concise, and actionable financial analysis. Include specific recommendations.

Answer:"""
            
            # Call Ollama
            result = subprocess.run(
                ['ollama', 'run', self.model_id, full_prompt],
                capture_output=True,
                text=True,
                timeout=60,
                encoding='utf-8',
                errors='ignore'
            )
            
            if result.returncode == 0:
                response = result.stdout.strip()
                # If brief requested, take only first line
                if is_brief and '\n' in response:
                    response = response.split('\n')[0]
                return response
            else:
                return self._fallback_response(prompt, context, is_brief)
        
        except Exception as e:
            st.warning(f"Ollama error: {e}. Using fallback.")
            return self._fallback_response(prompt, context)
    
    def _fallback_response(self, prompt: str, context: str, is_brief: bool = False) -> str:
        """Fallback response when Ollama is unavailable"""
        
        # Detect brief request
        brief_keywords = ['one line', 'brief', 'short', 'quick', 'simply', 'just tell', 'in short', 'summarize']
        if not is_brief:
            is_brief = any(keyword in prompt.lower() for keyword in brief_keywords)
        
        if is_brief:
            # Return one-line answer
            if 'expense' in prompt.lower() or 'cost' in prompt.lower():
                return "Review expenses, identify top spending categories, and cut 10-15% from discretionary items."
            elif 'revenue' in prompt.lower() or 'income' in prompt.lower():
                return "Focus on high-margin products, diversify revenue streams, and optimize pricing strategy."
            elif 'profit' in prompt.lower():
                return "Increase revenue by 15%, reduce costs by 10%, and improve operational efficiency."
            elif 'budget' in prompt.lower():
                return "Use 50/30/20 rule: 50% needs, 30% wants, 20% savings/debt repayment."
            elif 'invest' in prompt.lower():
                return "Diversify with index funds, maintain emergency fund, and invest consistently for long-term growth."
            else:
                return "Review your financial data, set clear goals, and create actionable plans with regular monitoring."
        
        # Full response for detailed queries
        response = f"""**Financial Analysis** (Rule-based response - Ollama unavailable)

Based on the query: "{prompt}"

"""
        
        if context:
            response += "**Context Summary:**\n"
            response += context[:500] + "...\n\n"
        
        response += """**General Recommendations:**
1. Review your current financial position thoroughly
2. Identify key areas for optimization
3. Set specific, measurable financial goals
4. Create an action plan with timelines
5. Monitor progress regularly

**Next Steps:**
- Ensure Ollama is installed: `curl -fsSL https://ollama.com/install.sh | sh`
- Pull the model: `ollama pull llama3.2:latest`
- Restart the application

For more detailed analysis, please ensure Ollama is running."""
        
        return response

# ============================================================================
# MAIN RAG PIPELINE
# ============================================================================

class RAGPipeline:
    """Complete RAG pipeline"""
    
    def __init__(self):
        self.embedding_engine = EmbeddingEngine()
        self.vector_store = VectorStore(self.embedding_engine.embedding_dim)
        self.llm = OllamaLLM()
        self.chunker = DocumentChunker()
    
    def process_document(self, file, file_type: str) -> Dict:
        """Process uploaded document"""
        try:
            # Extract text
            if file_type == 'pdf':
                text = PDFProcessor.extract_text_from_pdf(file)
                chunks = self.chunker.chunk_text(text)
            elif file_type in ['csv', 'xlsx']:
                df = pd.read_csv(file) if file_type == 'csv' else pd.read_excel(file)
                chunks = self.chunker.chunk_dataframe(df)
            else:  # txt
                text = file.read().decode('utf-8')
                chunks = self.chunker.chunk_text(text)
            
            # Create embeddings
            chunk_texts = [chunk['text'] for chunk in chunks]
            embeddings = self.embedding_engine.encode(chunk_texts)
            
            # Add metadata
            for chunk in chunks:
                chunk['source'] = file.name
                chunk['timestamp'] = datetime.now().isoformat()
            
            # Add to vector store
            self.vector_store.add_documents(chunks, embeddings)
            
            return {
                'success': True,
                'num_chunks': len(chunks),
                'source': file.name
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def query(self, question: str, top_k: int = 5) -> Dict:
        """Query the RAG system"""
        # Embed query
        query_embedding = self.embedding_engine.encode([question])[0]
        
        # Retrieve relevant documents
        results = self.vector_store.search(query_embedding, top_k=top_k)
        
        # Prepare context
        context = "\n\n".join([
            f"[Source: {doc['source']}]\n{doc['text'][:500]}"
            for doc, score in results
        ])
        
        # Generate response
        response = self.llm.generate(question, context)
        
        return {
            'answer': response,
            'retrieved_docs': results,
            'num_sources': len(results)
        }

# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">💰 Financial RAG Assistant</h1>', unsafe_allow_html=True)
    st.markdown("### Advanced RAG with OCR, Embeddings & Local LLM")
    
    # Initialize pipeline
    if 'rag_pipeline' not in st.session_state:
        with st.spinner("Initializing RAG pipeline..."):
            st.session_state.rag_pipeline = RAGPipeline()
    
    pipeline = st.session_state.rag_pipeline
    
    # System Status
    with st.expander("🔧 System Status", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("**Embeddings:**")
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                st.markdown('<span class="status-success">✓ Sentence Transformers</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-warning">⚠ Hash-based</span>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("**Vector Store:**")
            if FAISS_AVAILABLE:
                st.markdown('<span class="status-success">✓ FAISS</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-warning">⚠ NumPy</span>', unsafe_allow_html=True)
        
        with col3:
            st.markdown("**OCR:**")
            if OCR_AVAILABLE:
                st.markdown('<span class="status-success">✓ Available</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-warning">⚠ Limited</span>', unsafe_allow_html=True)
        
        with col4:
            st.markdown("**LLM:**")
            if pipeline.llm.available:
                st.markdown('<span class="status-success">✓ Ollama</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-warning">⚠ Fallback</span>', unsafe_allow_html=True)
        
        # Vector store stats
        stats = pipeline.vector_store.get_stats()
        st.info(f"📊 Vector Store: {stats['num_documents']} chunks | Dimension: {stats['embedding_dim']} | Method: {stats['method']}")
    
    # Sidebar - Document Upload
    with st.sidebar:
        st.header("📚 Document Upload")
        
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=['pdf', 'csv', 'xlsx', 'txt'],
            accept_multiple_files=True,
            help="Supports PDF (with OCR), CSV, XLSX, and TXT files"
        )
        
        if uploaded_files:
            if st.button("Process Documents", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"Processing {file.name}...")
                    
                    # Determine file type
                    file_type = file.name.split('.')[-1].lower()
                    
                    # Process document
                    result = pipeline.process_document(file, file_type)
                    
                    if result['success']:
                        st.success(f"✓ {file.name}: {result['num_chunks']} chunks")
                    else:
                        st.error(f"✗ {file.name}: {result['error']}")
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.text("Processing complete!")
                st.balloons()
        
        # Knowledge base stats
        st.markdown("---")
        st.header("📊 Knowledge Base")
        stats = pipeline.vector_store.get_stats()
        st.metric("Total Chunks", stats['num_documents'])
        st.metric("Embedding Method", pipeline.embedding_engine.method)
        
        if st.button("Clear Knowledge Base"):
            st.session_state.rag_pipeline = RAGPipeline()
            st.rerun()
    
    # Main Query Interface
    st.header("🔍 Ask a Financial Question")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_area(
            "Enter your question:",
            placeholder="e.g., Analyze my quarterly expenses and suggest cost reduction strategies",
            height=120,
            key="query_input"
        )
    
    with col2:
        st.markdown("**Settings**")
        top_k = st.slider("Context Chunks", 1, 10, 5)
        show_sources = st.checkbox("Show Sources", value=True)
    
    col_a, col_b = st.columns([1, 1])
    with col_a:
        analyze_btn = st.button("🚀 Analyze", type="primary", use_container_width=True)
    with col_b:
        if st.button("Clear History", use_container_width=True):
            st.session_state.conversation_history = []
            st.rerun()
    
    # Process Query
    if analyze_btn and query:
        if pipeline.vector_store.get_stats()['num_documents'] == 0:
            st.warning("⚠️ Please upload documents first!")
        else:
            with st.spinner("🧠 Analyzing with RAG pipeline..."):
                # Query the system
                result = pipeline.query(query, top_k=top_k)
                
                # Display Results
                st.markdown("---")
                st.markdown('<div class="module-box">📊 Financial Analysis Report</div>', unsafe_allow_html=True)
                
                # Main Answer
                st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                st.markdown("### 💡 Analysis & Recommendations")
                st.markdown(result['answer'])
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Retrieved Sources
                if show_sources and result['retrieved_docs']:
                    st.markdown("### 📖 Retrieved Context")
                    
                    for i, (doc, score) in enumerate(result['retrieved_docs'][:3], 1):
                        with st.expander(f"Source {i}: {doc['source']} (Relevance: {score:.2%})"):
                            st.text(doc['text'][:500] + "...")
                            st.caption(f"Chunk ID: {doc['chunk_id']} | Word Count: {doc.get('word_count', 'N/A')}")
                
                # Metadata
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Sources Used", result['num_sources'])
                with col2:
                    st.metric("LLM Status", "Ollama" if pipeline.llm.available else "Fallback")
                with col3:
                    st.metric("Embedding Method", pipeline.embedding_engine.method)
                
                # Save to history
                st.session_state.conversation_history.append({
                    'query': query,
                    'answer': result['answer'],
                    'timestamp': datetime.now().isoformat(),
                    'num_sources': result['num_sources']
                })
    
    # Conversation History
    if st.session_state.conversation_history:
        st.markdown("---")
        st.header("📜 Recent Conversations")
        
        for i, conv in enumerate(reversed(st.session_state.conversation_history[-5:]), 1):
            with st.expander(f"Q{len(st.session_state.conversation_history) - i + 1}: {conv['query'][:60]}..."):
                st.markdown(f"**Question:** {conv['query']}")
                st.markdown(f"**Answer:** {conv['answer'][:300]}...")
                st.caption(f"Time: {conv['timestamp']} | Sources: {conv['num_sources']}")
    
    # Installation Guide
    with st.expander("📦 Installation Guide"):
        st.markdown("""
        ### Required Dependencies
        
        ```bash
        # Core dependencies
        pip install streamlit pandas numpy plotly openpyxl
        
        # Embeddings (recommended)
        pip install sentence-transformers
        
        # Vector store (recommended)
        pip install faiss-cpu
        
        # PDF & OCR support
        pip install PyPDF2 pytesseract pdf2image pillow
        
        # Install Tesseract (for OCR)
        # Ubuntu/Debian: sudo apt-get install tesseract-ocr
        # macOS: brew install tesseract
        # Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
        
        # Install Ollama
        curl -fsSL https://ollama.com/install.sh | sh
        ollama pull llama3.2:latest
        ```
        
        ### Optional: Poppler (for pdf2image)
        ```bash
        # Ubuntu/Debian: sudo apt-get install poppler-utils
        # macOS: brew install poppler
        ```
        """)

if __name__ == "__main__":
    main()