import streamlit as st
import pandas as pd
import numpy as np
import json
import io
import base64
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional
import hashlib

# Configuration
st.set_page_config(
    page_title="Financial Decision Assistant",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .module-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .insight-box {
        background: #f0f8ff;
        padding: 1.5rem;
        border-left: 5px solid #1f77b4;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'knowledge_base' not in st.session_state:
    st.session_state.knowledge_base = []
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'embeddings_cache' not in st.session_state:
    st.session_state.embeddings_cache = {}

# ============================================================================
# MODULE 1: MULTIMODAL PREPROCESSING
# ============================================================================

class MultimodalPreprocessor:
    """Handles file chunking, embedding, and feature extraction"""
    
    @staticmethod
    def process_csv(file) -> Dict[str, Any]:
        """Process CSV files"""
        df = pd.read_csv(file)
        
        # Extract features
        features = {
            'type': 'tabular',
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'numerical_summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {},
            'sample_data': df.head(10).to_dict('records'),
            'text_representation': df.to_string()
        }
        
        # Create chunks
        chunks = []
        chunk_size = 50
        for i in range(0, len(df), chunk_size):
            chunk_df = df.iloc[i:i+chunk_size]
            chunks.append({
                'chunk_id': i // chunk_size,
                'data': chunk_df.to_dict('records'),
                'text': chunk_df.to_string()
            })
        
        return {
            'features': features,
            'chunks': chunks,
            'raw_data': df
        }
    
    @staticmethod
    def process_excel(file) -> Dict[str, Any]:
        """Process Excel files"""
        df = pd.read_excel(file)
        return MultimodalPreprocessor.process_csv(io.StringIO(df.to_csv(index=False)))
    
    @staticmethod
    def process_text(text: str) -> Dict[str, Any]:
        """Process text input"""
        # Split into sentences for chunking
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        chunks = []
        chunk_size = 3
        for i in range(0, len(sentences), chunk_size):
            chunk_text = '. '.join(sentences[i:i+chunk_size])
            chunks.append({
                'chunk_id': i // chunk_size,
                'text': chunk_text
            })
        
        features = {
            'type': 'text',
            'length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(sentences)
        }
        
        return {
            'features': features,
            'chunks': chunks,
            'raw_data': text
        }
    
    @staticmethod
    def create_embedding(text: str) -> List[float]:
        """Create simple embedding using hash-based vectorization"""
        # Simple embedding: use character frequency and hash
        vector = [0.0] * 128
        
        # Character frequency
        for char in text.lower():
            idx = ord(char) % 128
            vector[idx] += 1
        
        # Normalize
        magnitude = sum(v**2 for v in vector) ** 0.5
        if magnitude > 0:
            vector = [v / magnitude for v in vector]
        
        # Add some hash-based features
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        for i in range(len(vector)):
            vector[i] += ((hash_val >> i) & 1) * 0.1
        
        return vector
    
    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a**2 for a in vec1) ** 0.5
        magnitude2 = sum(b**2 for b in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)

# ============================================================================
# MODULE 2: CONTEXT EVALUATION
# ============================================================================

class ContextEvaluator:
    """Evaluates whether query needs RAG lookup"""
    
    @staticmethod
    def evaluate_query(query: str, knowledge_base: List[Dict]) -> Dict[str, Any]:
        """Determine if query needs external context"""
        
        # Keywords that suggest need for data
        data_keywords = ['analyze', 'compare', 'calculate', 'show', 'data', 'report', 
                        'trend', 'forecast', 'budget', 'expense', 'revenue', 'profit']
        
        # Check if query contains data keywords
        needs_rag = any(keyword in query.lower() for keyword in data_keywords)
        
        # Check if we have relevant data in knowledge base
        has_relevant_data = len(knowledge_base) > 0
        
        return {
            'needs_rag': needs_rag and has_relevant_data,
            'confidence': 0.8 if needs_rag else 0.3,
            'reason': 'Query requires data analysis' if needs_rag else 'Simple query can be answered directly'
        }

# ============================================================================
# MODULE 3: RAG LOOKUP
# ============================================================================

class RAGRetriever:
    """Retrieves relevant context from knowledge base"""
    
    def __init__(self, preprocessor: MultimodalPreprocessor):
        self.preprocessor = preprocessor
    
    def retrieve(self, query: str, knowledge_base: List[Dict], top_k: int = 3) -> Dict[str, Any]:
        """Retrieve relevant chunks from knowledge base"""
        
        if not knowledge_base:
            return {
                'sufficient': False,
                'contexts': [],
                'scores': []
            }
        
        # Create query embedding
        query_embedding = self.preprocessor.create_embedding(query)
        
        # Calculate similarity scores for all chunks
        all_chunks = []
        for kb_item in knowledge_base:
            for chunk in kb_item['chunks']:
                chunk_text = chunk.get('text', str(chunk.get('data', '')))
                chunk_embedding = self.preprocessor.create_embedding(chunk_text)
                similarity = self.preprocessor.cosine_similarity(query_embedding, chunk_embedding)
                
                all_chunks.append({
                    'chunk': chunk,
                    'source': kb_item['source'],
                    'similarity': similarity,
                    'features': kb_item['features']
                })
        
        # Sort by similarity
        all_chunks.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Get top-k
        top_chunks = all_chunks[:top_k]
        
        # Evaluate sufficiency
        sufficient = len(top_chunks) > 0 and top_chunks[0]['similarity'] > 0.3
        
        return {
            'sufficient': sufficient,
            'contexts': top_chunks,
            'scores': [c['similarity'] for c in top_chunks]
        }

# ============================================================================
# MODULE 4: INFERENCE MODULE
# ============================================================================

class InferenceEngine:
    """Performs reasoning and answer generation"""
    
    @staticmethod
    def generate_insight(query: str, contexts: List[Dict], direct_answer: bool = False) -> Dict[str, Any]:
        """Generate financial insights based on query and context"""
        
        if direct_answer:
            return InferenceEngine._direct_inference(query)
        
        # Analyze retrieved contexts
        analysis = InferenceEngine._analyze_contexts(contexts)
        
        # Generate recommendations
        recommendations = InferenceEngine._generate_recommendations(query, analysis)
        
        # Create visualization data
        viz_data = InferenceEngine._prepare_visualizations(contexts)
        
        return {
            'analysis': analysis,
            'recommendations': recommendations,
            'visualizations': viz_data,
            'confidence': analysis.get('confidence', 0.7)
        }
    
    @staticmethod
    def _direct_inference(query: str) -> Dict[str, Any]:
        """Handle queries that don't need RAG"""
        
        # Simple rule-based responses for common financial queries
        response = {
            'analysis': {
                'summary': 'Based on general financial principles:',
                'key_points': []
            },
            'recommendations': [],
            'visualizations': None,
            'confidence': 0.6
        }
        
        query_lower = query.lower()
        
        if 'budget' in query_lower:
            response['analysis']['key_points'] = [
                'Create a detailed budget tracking income and expenses',
                'Follow the 50/30/20 rule: 50% needs, 30% wants, 20% savings',
                'Review and adjust monthly'
            ]
            response['recommendations'] = [
                'Start tracking all expenses',
                'Set specific savings goals',
                'Use budgeting tools or apps'
            ]
        
        elif 'invest' in query_lower:
            response['analysis']['key_points'] = [
                'Diversification reduces risk',
                'Consider your risk tolerance and time horizon',
                'Regular contributions benefit from dollar-cost averaging'
            ]
            response['recommendations'] = [
                'Start with index funds for broad market exposure',
                'Maintain an emergency fund first',
                'Consider consulting a financial advisor'
            ]
        
        elif 'save' in query_lower or 'saving' in query_lower:
            response['analysis']['key_points'] = [
                'Build an emergency fund (3-6 months expenses)',
                'Automate savings to make it consistent',
                'Use high-yield savings accounts'
            ]
            response['recommendations'] = [
                'Set up automatic transfers to savings',
                'Cut unnecessary subscriptions',
                'Look for better interest rates'
            ]
        
        else:
            response['analysis']['key_points'] = [
                'Financial planning requires understanding your current situation',
                'Set clear, measurable goals',
                'Regular review and adjustment is key'
            ]
            response['recommendations'] = [
                'Define your financial goals',
                'Gather all financial documents',
                'Consider seeking professional advice'
            ]
        
        return response
    
    @staticmethod
    def _analyze_contexts(contexts: List[Dict]) -> Dict[str, Any]:
        """Analyze retrieved contexts"""
        
        analysis = {
            'summary': '',
            'key_points': [],
            'metrics': {},
            'confidence': 0.0
        }
        
        if not contexts:
            return analysis
        
        # Extract numerical data
        numerical_data = []
        for ctx in contexts:
            if ctx['features']['type'] == 'tabular':
                numerical_data.append(ctx['features'].get('numerical_summary', {}))
        
        # Generate summary
        analysis['summary'] = f"Analyzed {len(contexts)} relevant data sources"
        
        # Extract key points
        for i, ctx in enumerate(contexts[:3]):
            score = ctx['similarity']
            analysis['key_points'].append(
                f"Source {i+1} (relevance: {score:.2%}): {ctx['source']}"
            )
        
        # Calculate confidence
        if contexts:
            analysis['confidence'] = sum(c['similarity'] for c in contexts) / len(contexts)
        
        # Aggregate metrics
        if numerical_data:
            analysis['metrics'] = numerical_data[0]  # Use first source's metrics
        
        return analysis
    
    @staticmethod
    def _generate_recommendations(query: str, analysis: Dict) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Based on confidence level
        if analysis['confidence'] > 0.7:
            recommendations.append("High confidence in analysis - recommendations are data-driven")
        else:
            recommendations.append("Moderate confidence - consider gathering more data")
        
        # Query-specific recommendations
        query_lower = query.lower()
        
        if 'expense' in query_lower or 'cost' in query_lower:
            recommendations.extend([
                "Review expense categories for optimization opportunities",
                "Identify top 3 expense categories for reduction",
                "Set spending limits for discretionary categories"
            ])
        
        elif 'revenue' in query_lower or 'income' in query_lower:
            recommendations.extend([
                "Analyze revenue trends for growth patterns",
                "Identify highest performing revenue streams",
                "Explore diversification opportunities"
            ])
        
        elif 'profit' in query_lower or 'margin' in query_lower:
            recommendations.extend([
                "Focus on improving high-margin products/services",
                "Review cost structure for optimization",
                "Consider pricing strategy adjustments"
            ])
        
        else:
            recommendations.extend([
                "Conduct comprehensive financial review",
                "Set measurable financial goals",
                "Implement regular monitoring and reporting"
            ])
        
        return recommendations
    
    @staticmethod
    def _prepare_visualizations(contexts: List[Dict]) -> Optional[Dict]:
        """Prepare data for visualizations"""
        
        viz_data = {
            'has_viz': False,
            'charts': []
        }
        
        for ctx in contexts:
            if ctx['features']['type'] == 'tabular':
                viz_data['has_viz'] = True
                viz_data['source_data'] = ctx['chunk'].get('data', [])
                break
        
        return viz_data if viz_data['has_viz'] else None

# ============================================================================
# MODULE 5: POST-PROCESSING
# ============================================================================

class PostProcessor:
    """Response refinement and optimization"""
    
    @staticmethod
    def refine_response(insight: Dict) -> Dict[str, Any]:
        """Refine and structure the final response"""
        
        refined = {
            'executive_summary': PostProcessor._create_executive_summary(insight),
            'detailed_analysis': insight['analysis'],
            'recommendations': PostProcessor._prioritize_recommendations(insight['recommendations']),
            'supporting_data': insight.get('visualizations'),
            'confidence_level': PostProcessor._interpret_confidence(insight['confidence']),
            'next_steps': PostProcessor._generate_next_steps(insight)
        }
        
        return refined
    
    @staticmethod
    def _create_executive_summary(insight: Dict) -> str:
        """Create concise executive summary"""
        
        analysis = insight['analysis']
        confidence = insight['confidence']
        
        summary = f"Financial Analysis Summary (Confidence: {confidence:.0%})\n\n"
        summary += analysis.get('summary', 'Analysis completed based on available data.')
        
        return summary
    
    @staticmethod
    def _prioritize_recommendations(recommendations: List[str]) -> List[Dict]:
        """Prioritize and categorize recommendations"""
        
        prioritized = []
        for i, rec in enumerate(recommendations):
            priority = 'High' if i < 2 else 'Medium' if i < 4 else 'Low'
            prioritized.append({
                'priority': priority,
                'action': rec,
                'timeframe': 'Immediate' if i < 2 else 'Short-term'
            })
        
        return prioritized
    
    @staticmethod
    def _interpret_confidence(confidence: float) -> str:
        """Interpret confidence score"""
        
        if confidence > 0.8:
            return "High - Strong data support"
        elif confidence > 0.6:
            return "Moderate - Reasonable data support"
        else:
            return "Low - Limited data available"
    
    @staticmethod
    def _generate_next_steps(insight: Dict) -> List[str]:
        """Generate actionable next steps"""
        
        return [
            "Review the detailed analysis and recommendations",
            "Prioritize high-priority actions for immediate implementation",
            "Set up monitoring metrics to track progress",
            "Schedule follow-up review in 30 days"
        ]

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">💰 Financial Decision Assistant</h1>', unsafe_allow_html=True)
    st.markdown("### Multimodal RAG-Powered Financial Analysis System")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Knowledge Base")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload financial data (CSV, XLSX, TXT)",
            type=['csv', 'xlsx', 'txt'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            preprocessor = MultimodalPreprocessor()
            
            for file in uploaded_files:
                file_id = f"{file.name}_{file.size}"
                
                # Check if already processed
                if file_id not in [kb['id'] for kb in st.session_state.knowledge_base]:
                    # Process file
                    if file.name.endswith('.csv'):
                        processed = preprocessor.process_csv(file)
                    elif file.name.endswith('.xlsx'):
                        processed = preprocessor.process_excel(file)
                    else:
                        content = file.read().decode('utf-8')
                        processed = preprocessor.process_text(content)
                    
                    # Add to knowledge base
                    st.session_state.knowledge_base.append({
                        'id': file_id,
                        'source': file.name,
                        'timestamp': datetime.now().isoformat(),
                        'features': processed['features'],
                        'chunks': processed['chunks']
                    })
        
        # Display knowledge base stats
        st.metric("Documents Loaded", len(st.session_state.knowledge_base))
        
        if st.session_state.knowledge_base:
            total_chunks = sum(len(kb['chunks']) for kb in st.session_state.knowledge_base)
            st.metric("Total Chunks", total_chunks)
        
        if st.button("Clear Knowledge Base"):
            st.session_state.knowledge_base = []
            st.rerun()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔍 Query Input")
        query = st.text_area(
            "Enter your financial question or request:",
            placeholder="e.g., Analyze my expenses and suggest where I can save money",
            height=100
        )
        
        col_a, col_b = st.columns(2)
        with col_a:
            analyze_btn = st.button("🚀 Analyze", type="primary", use_container_width=True)
        with col_b:
            if st.button("Clear History", use_container_width=True):
                st.session_state.conversation_history = []
                st.rerun()
    
    with col2:
        st.header("⚙️ Settings")
        top_k = st.slider("Context Chunks to Retrieve", 1, 5, 3)
        show_process = st.checkbox("Show Processing Steps", value=True)
    
    # Process query
    if analyze_btn and query:
        with st.spinner("Processing your query..."):
            # Initialize components
            preprocessor = MultimodalPreprocessor()
            evaluator = ContextEvaluator()
            retriever = RAGRetriever(preprocessor)
            inference_engine = InferenceEngine()
            post_processor = PostProcessor()
            
            # Step 1: Context Evaluation
            if show_process:
                st.markdown('<div class="module-box">📋 Step 1: Context Evaluation</div>', unsafe_allow_html=True)
            
            evaluation = evaluator.evaluate_query(query, st.session_state.knowledge_base)
            
            if show_process:
                st.info(f"**Decision:** {'Use RAG' if evaluation['needs_rag'] else 'Direct Answer'} | **Confidence:** {evaluation['confidence']:.0%}")
            
            # Step 2: RAG Lookup or Direct Inference
            if evaluation['needs_rag']:
                if show_process:
                    st.markdown('<div class="module-box">🔎 Step 2: RAG Retrieval</div>', unsafe_allow_html=True)
                
                retrieval_result = retriever.retrieve(query, st.session_state.knowledge_base, top_k=top_k)
                
                if show_process:
                    st.success(f"Retrieved {len(retrieval_result['contexts'])} relevant contexts")
                    if retrieval_result['contexts']:
                        for i, ctx in enumerate(retrieval_result['contexts'][:3]):
                            st.write(f"- Context {i+1}: {ctx['source']} (Score: {ctx['similarity']:.2%})")
                
                # Step 3: Inference
                if show_process:
                    st.markdown('<div class="module-box">🧠 Step 3: Inference & Analysis</div>', unsafe_allow_html=True)
                
                insight = inference_engine.generate_insight(query, retrieval_result['contexts'])
            else:
                if show_process:
                    st.markdown('<div class="module-box">🧠 Step 2: Direct Inference</div>', unsafe_allow_html=True)
                
                insight = inference_engine.generate_insight(query, [], direct_answer=True)
            
            # Step 4: Post-Processing
            if show_process:
                st.markdown('<div class="module-box">✨ Step 4: Response Refinement</div>', unsafe_allow_html=True)
            
            final_response = post_processor.refine_response(insight)
            
            # Display Results
            st.markdown("---")
            st.header("📊 Decision Insight Report")
            
            # Executive Summary
            st.markdown('<div class="insight-box">', unsafe_allow_html=True)
            st.subheader("Executive Summary")
            st.write(final_response['executive_summary'])
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Key Findings
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Confidence Level", final_response['confidence_level'])
            with col2:
                st.metric("Recommendations", len(final_response['recommendations']))
            with col3:
                st.metric("Data Sources", len(st.session_state.knowledge_base))
            
            # Detailed Analysis
            st.subheader("🔍 Detailed Analysis")
            analysis = final_response['detailed_analysis']
            st.write(analysis.get('summary', ''))
            
            if analysis.get('key_points'):
                st.write("**Key Findings:**")
                for point in analysis['key_points']:
                    st.write(f"• {point}")
            
            # Recommendations
            st.subheader("💡 Recommended Actions")
            for rec in final_response['recommendations']:
                priority_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
                st.write(f"{priority_color[rec['priority']]} **{rec['priority']} Priority** ({rec['timeframe']})")
                st.write(f"   {rec['action']}")
                st.write("")
            
            # Visualizations
            if final_response['supporting_data']:
                st.subheader("📈 Data Visualizations")
                viz_data = final_response['supporting_data']
                
                if viz_data.get('source_data'):
                    df = pd.DataFrame(viz_data['source_data'])
                    
                    # Auto-detect numeric columns for visualization
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    
                    if len(numeric_cols) >= 1:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Bar chart
                            if len(numeric_cols) >= 1:
                                fig = px.bar(df.head(10), y=numeric_cols[0], title=f"{numeric_cols[0]} Overview")
                                st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Line chart
                            if len(numeric_cols) >= 1:
                                fig = px.line(df.head(10), y=numeric_cols[0], title=f"{numeric_cols[0]} Trend")
                                st.plotly_chart(fig, use_container_width=True)
            
            # Next Steps
            st.subheader("🎯 Next Steps")
            for i, step in enumerate(final_response['next_steps'], 1):
                st.write(f"{i}. {step}")
            
            # Save to history
            st.session_state.conversation_history.append({
                'query': query,
                'response': final_response,
                'timestamp': datetime.now().isoformat()
            })
    
    # Conversation History
    if st.session_state.conversation_history:
        st.markdown("---")
        st.header("📜 Conversation History")
        
        for i, conv in enumerate(reversed(st.session_state.conversation_history[-5:]), 1):
            with st.expander(f"Query {len(st.session_state.conversation_history) - i + 1}: {conv['query'][:50]}..."):
                st.write(f"**Query:** {conv['query']}")
                st.write(f"**Timestamp:** {conv['timestamp']}")
                st.write(f"**Confidence:** {conv['response']['confidence_level']}")

if __name__ == "__main__":
    main()