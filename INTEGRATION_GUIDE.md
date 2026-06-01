"""
Integration Guide: How to Add Smart Retrieval & Response Validation
Step-by-step instructions to integrate new modules into your RAG pipeline
"""

# ============================================================================
# OPTION A: Minimal Integration (Just Copy-Paste into responder.py)
# ============================================================================

"""
In src/responder.py, add these imports at the top:

    from src.improved_prompts import select_prompt_template, SYSTEM_PROMPT_VI
    from src.smart_retriever import SmartContextRetriever
    from src.response_validator import ResponseValidator

Then update the ResponseGenerator class:
"""

# BEFORE (old code):
class ResponseGenerator:
    def __init__(self, model: 'LocalGGUFModel'):
        self.model = model
        self.prompt_template = PROMPT_TEMPLATE_VI  # ← Generic template
    
    def generate(self, question: str, retrieved_chunks: List[Dict]) -> Response:
        # Format context
        context = "\n\n".join([chunk['text'] for chunk in retrieved_chunks])
        
        # Build generic prompt
        prompt = self.prompt_template.format(
            context=context,
            question=question
        )
        
        # Generate answer
        answer = self.model.generate(prompt, max_tokens=128)
        return Response(answer=answer)


# AFTER (with smart modules):
class ResponseGenerator:
    def __init__(self, model: 'LocalGGUFModel'):
        self.model = model
        self.smart_retriever = SmartContextRetriever()  # ← NEW
        self.validator = ResponseValidator()             # ← NEW
    
    def generate(self, question: str, retrieved_chunks: List[Dict], 
                 semantic_scores: List[float] = None) -> Response:
        
        # [NEW] Smart ranking
        if semantic_scores is None:
            semantic_scores = [0.5] * len(retrieved_chunks)
        
        ranked_chunks = self.smart_retriever.rank_chunks(
            chunks=retrieved_chunks,
            question=question,
            semantic_scores=semantic_scores
        )
        
        # [NEW] Select best chunks
        best_chunks = self.smart_retriever.select_best_chunks(ranked_chunks, top_k=3)
        
        # [NEW] Format with ranking info
        context = self.smart_retriever.format_context_with_scores(best_chunks)
        
        # [NEW] Select smart prompt template
        template, template_type, reason = select_prompt_template(
            question=question,
            context=context,
            num_chunks=len(best_chunks)
        )
        
        # Build enhanced prompt
        prompt = template.format(
            context=context,
            question=question
        )
        
        # Generate answer
        answer = self.model.generate(prompt, max_tokens=128)
        
        # [NEW] Validate quality
        quality = self.validator.assess_overall(question, answer, context)
        
        return Response(
            answer=answer,
            quality_score=quality.overall,
            quality_issues=quality.issues
        )


# ============================================================================
# OPTION B: Full Integration into RAG Pipeline (rag_pipeline.py)
# ============================================================================

"""
Update the RAGPipeline class to use smart retriever:
"""

class RAGPipeline:
    def __init__(self, ...):
        # ... existing code ...
        self.smart_retriever = SmartContextRetriever()  # ← NEW
        self.validator = ResponseValidator()             # ← NEW
    
    def answer(self, question: str) -> Dict:
        # Normalize question
        normalized_q = self.normalizer.normalize(question)
        
        # Retrieve chunks with semantic scores
        retrieval_result = self.retriever.retrieve(normalized_q)
        chunks = retrieval_result.chunks
        
        # Extract semantic scores from retrieval
        semantic_scores = [chunk.get('score', 0.5) for chunk in chunks]
        
        # [NEW] Smart ranking
        ranked = self.smart_retriever.rank_chunks(
            chunks=chunks,
            question=normalized_q,
            semantic_scores=semantic_scores
        )
        
        # [NEW] Select best chunks with quality threshold
        best_chunks = self.smart_retriever.select_best_chunks(
            ranked,
            top_k=3,
            min_score=0.3  # Only use chunks scoring above 30%
        )
        
        # If no good chunks found, return "not found"
        if not best_chunks:
            return {
                'answer': 'Không tìm thấy thông tin liên quan trong cơ sở dữ liệu',
                'quality_score': 0.0,
                'sources': [],
                'confidence': 'low'
            }
        
        # Format context
        context_text = self.smart_retriever.format_context_with_scores(best_chunks)
        
        # Get source pages
        source_pages = [chunk.metadata.get('page_num', '?') for chunk in best_chunks]
        
        # Generate response
        raw_answer = self.responder.generate(
            question=normalized_q,
            retrieved_chunks=[
                {'text': c.text, 'metadata': c.metadata} 
                for c in best_chunks
            ],
            semantic_scores=[c.semantic_score for c in best_chunks]
        )
        
        # [NEW] Validate quality
        quality = self.validator.assess_overall(
            normalized_q,
            raw_answer.answer,
            context_text
        )
        
        # Log ranking explanation for debugging
        if self.verbose:
            print(self.smart_retriever.explain_ranking(ranked, top_k=3))
        
        return {
            'answer': raw_answer.answer,
            'quality_score': quality.overall,
            'quality_acceptable': self.validator.is_acceptable(quality),
            'sources': source_pages,
            'confidence': self._determine_confidence(quality.grounding),
            'issues': quality.issues,
            'suggestions': quality.suggestions
        }
    
    def _determine_confidence(self, grounding_score: float) -> str:
        """Determine confidence level based on grounding quality"""
        if grounding_score >= 0.8:
            return 'high'
        elif grounding_score >= 0.6:
            return 'medium'
        else:
            return 'low'


# ============================================================================
# OPTION C: Streamlit Integration (streamlit_app.py)
# ============================================================================

"""
Update the Streamlit app to display quality metrics:
"""

def main():
    st.set_page_config(page_title="HR Chatbot", layout="wide")
    st.title("🤖 HR Chatbot với AI")
    
    # Initialize components
    pipeline = initialize_components()
    
    # User input
    user_question = st.text_input("Hỏi bất kỳ điều gì về chính sách HR:")
    
    if user_question:
        with st.spinner("Đang xử lý..."):
            result = pipeline.answer(user_question)
        
        # Display main answer
        st.markdown("### 📝 Trả lời")
        st.write(result['answer'])
        
        # [NEW] Quality metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            quality_pct = result['quality_score'] * 100
            if quality_pct >= 70:
                st.success(f"✅ Chất lượng: {quality_pct:.0f}%")
            elif quality_pct >= 50:
                st.warning(f"⚠️ Chất lượng: {quality_pct:.0f}%")
            else:
                st.error(f"❌ Chất lượng: {quality_pct:.0f}%")
        
        with col2:
            confidence_emojis = {
                'high': '🔥 Cao',
                'medium': '⚡ Trung bình',
                'low': '❄️ Thấp'
            }
            st.info(f"Độ tin cậy: {confidence_emojis.get(result['confidence'], '?')}")
        
        with col3:
            sources = ', '.join(map(str, result['sources']))
            st.info(f"Nguồn: Trang {sources}")
        
        # [NEW] Show issues if quality is low
        if not result['quality_acceptable']:
            st.warning("⚠️ Phản hồi này có thể không hoàn chỉnh")
            if result['issues']:
                st.markdown("**Các vấn đề phát hiện:**")
                for issue in result['issues']:
                    st.markdown(f"- {issue}")
            if result['suggestions']:
                st.markdown("**Gợi ý cải thiện:**")
                for suggestion in result['suggestions']:
                    st.markdown(f"- {suggestion}")


# ============================================================================
# Testing the Integration
# ============================================================================

"""
Quick test to verify everything works:
"""

if __name__ == "__main__":
    from src.rag_pipeline import RAGPipeline
    
    # Initialize pipeline
    pipeline = RAGPipeline(
        model_name="qwen2.5-1.5b-instruct-q4_k_m.gguf",
        embedding_model="all-MiniLM-L6-v2"
    )
    
    # Test question
    test_question = "Tôi được hưởng bao nhiêu ngày nghỉ phép mỗi năm?"
    
    # Get answer
    result = pipeline.answer(test_question)
    
    # Display results
    print("=" * 60)
    print(f"Q: {test_question}")
    print(f"\nA: {result['answer']}")
    print(f"\nQuality Score: {result['quality_score']:.1%}")
    print(f"Confidence: {result['confidence']}")
    print(f"Sources: Page {', '.join(map(str, result['sources']))}")
    print(f"Acceptable: {result['quality_acceptable']}")
    
    if result['issues']:
        print(f"\nIssues: {result['issues']}")
    if result['suggestions']:
        print(f"Suggestions: {result['suggestions']}")
    print("=" * 60)


# ============================================================================
# Configuration Recommendations
# ============================================================================

"""
Optimal settings for smart retrieval:

# In rag_pipeline.py initialization:
pipeline = RAGPipeline(
    model_name="qwen2.5-1.5b-instruct-q4_k_m.gguf",
    n_ctx=1024,
    n_gpu_layers=-1,
    
    # Retriever settings
    top_k=5,  # Retrieve 5 chunks first
    
    # Smart retriever settings (will filter to top 3)
    smart_top_k=3,          # Select top 3 after ranking
    min_quality_score=0.3,  # Only use chunks > 30% quality
    
    # Ranking weights (optional - uses defaults if not specified)
    ranking_weights={
        'semantic': 0.4,
        'keyword': 0.35,
        'specificity': 0.15,
        'coherence': 0.1
    }
)

# These settings balance:
✅ Quality (select best chunks)
✅ Speed (don't rank too many)
✅ Diversity (retrieve 5, select 3 best)
✅ Coverage (get different perspectives)
"""


# ============================================================================
# Troubleshooting
# ============================================================================

"""
Q: Response quality score is still low?
A: Check:
   1. Context quality (are chunks relevant?)
   2. Question clarity (is question well-formed?)
   3. Model capacity (is model understanding the prompt?)
   
Q: Responses are now shorter than before?
A: That's expected - we're being more selective.
   If you need longer responses, adjust:
   - max_tokens in model generation
   - response length validation threshold

Q: Which prompt template is being selected?
A: Add debug logging:
   template, template_type, reason = select_prompt_template(...)
   print(f"Selected: {template_type} - {reason}")

Q: Want to use different ranking weights?
A: Pass custom weights to rank_chunks():
   ranked = smart_retriever.rank_chunks(
       chunks=chunks,
       question=question,
       semantic_scores=scores,
       weights={'semantic': 0.5, 'keyword': 0.3, ...}
   )

Q: How to disable validation?
A: Just don't call validator.assess_overall() - optional step
   Or set min_quality_score=0.0 to accept all chunks
"""
