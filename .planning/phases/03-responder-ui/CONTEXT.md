# Phase 3: Responder, UI & Integration

## Phase Goal
Build LLM responder module + Streamlit UI for end-to-end RAG chatbot. Users can ask Vietnamese HR questions, system retrieves relevant policies, and generates fluent Vietnamese answers. No hallucination, sourced from handbook only.

**Duration:** ~4-5 hours (Week 3)  
**Status:** Planning ready, implementation starting  
**Entry Criteria:** Phase 2 complete (100% validation passed)

---

## Architecture Decisions

### 1. Responder Module (Agent 2 – Phi-3-Mini)

**Module:** `src/responder.py`

**Design:**
- Phi-3-Mini LLM (3.8B parameters, GGUF format)
- Takes retrieved chunks + user question as input
- Generates fluent Vietnamese answers
- Sources answers from context only (no hallucination)
- Includes uncertainty handling (when context insufficient)

**Key Components:**
- `ResponseGenerator` class – Main responder
- Prompt engineering: Vietnamese instruction format
- Context formatting: Chunks with page references
- Source attribution: Which chunks were used
- Confidence scoring: How certain is the answer

**Model Integration:**
- Uses `LocalGGUFModel` from Phase 1
- Phi-3-Mini-4k-instruct (~2.3GB GGUF)
- Context window: 2048 tokens
- Temperature: 0.3 (low randomness for consistency)

**Performance Target:** <2 seconds per response on 8GB RAM, 4-core CPU

### 2. Streamlit UI

**Module:** `streamlit_app.py`

**Design:**
- Interactive web interface (no backend server needed)
- Single-page app: question input → answer display
- Three-column layout:
  - Left: Question input + submit button
  - Center: Answer display + sources
  - Right: Settings/debug info (optional)

**Key Features:**
- Question input box (Vietnamese support)
- Real-time answer streaming (if available)
- Source display with page references
- Latency metrics (for debugging)
- Error handling (graceful failures)
- Session state (chat history optional)

**Tech Stack:**
- Streamlit framework (lightweight, no backend)
- st.write() for Markdown formatting
- st.sidebar for settings
- st.columns for layout
- Session state for persistence

**Performance Target:** <3 seconds UI + response latency

### 3. End-to-End Integration

**Data Flow:**
```
User Question (Vietnamese)
  ↓
[Streamlit UI Input] 
  ↓
[Phase 2: Question Normalizer] → Normalized query
  ↓
[Phase 2: Retriever] → Top-3 chunks + distances
  ↓
[Phase 3: Responder] → Context prompt + question
  ↓
[Phi-3-Mini LLM] → Vietnamese answer
  ↓
[Streamlit UI Output] → Answer + sources + metrics
```

**Integration Points:**
- Question Normalizer: `from src.question_normalizer import QuestionNormalizer`
- Retriever: `from src.retriever import Retriever`
- Responder: `from src.responder import ResponseGenerator`
- Vector Store: `from src.embeddings import VectorStoreManager, LocalEmbedder`

### 4. Hallucination Prevention

**Strategy:**
1. **Context-Only Answers:** Explicitly instruct model to use only retrieved chunks
2. **Uncertainty Handling:** Detect when model lacks confidence, respond with "not in handbook"
3. **Source Attribution:** Always cite which chunk(s) were used
4. **Token Limits:** Cap response to 256 tokens (prevents rambling)
5. **Prompt Engineering:** Use Vietnamese instructions for clarity

**Example Prompt Template:**
```
Bạn là trợ lý HR của công ty. Trả lời câu hỏi CHỈ dựa vào thông tin dưới đây.
Nếu không tìm thấy, nói: "Theo tài liệu hiện có, không có thông tin về điều này."

Thông tin từ tài liệu:
{context}

Câu hỏi: {question}

Trả lời (ngắn gọn, ≤256 từ):
```

---

## Implementation Status

### Phase 3 Tasks

**Task 1: Phi-3-Mini Model Setup**
- Download/locate Phi-3-Mini GGUF (~2.3GB)
- Store in `./models/phi-3-mini.gguf`
- Verify loading with test script

**Task 2: Responder Module** (`src/responder.py`)
- Implement `ResponseGenerator` class
- Integrate with Phase 1 `LocalGGUFModel`
- Prompt engineering for Vietnamese HR context
- Source tracking (which chunks used)
- Confidence scoring

**Task 3: Response Formatting**
- Parse LLM output cleanly
- Extract sources from context
- Format for Streamlit display
- Handle errors gracefully

**Task 4: Streamlit UI** (`streamlit_app.py`)
- Initialize session state (vector store, embedder, responder)
- Question input interface
- Answer display with markdown formatting
- Source citations
- Performance metrics
- Error messages

**Task 5: Integration Testing** (`test_e2e.py`)
- End-to-end pipeline test
- Sample Vietnamese HR questions
- Verify answer quality (no hallucination)
- Latency benchmarking
- Error handling

**Task 6: Deployment & Documentation**
- README with installation instructions
- Usage examples
- Performance notes
- Known limitations

---

## Dependencies (New for Phase 3)

**Already Available (Phase 1-2):**
- PyMuPDF, sentence-transformers, chromadb, langchain, llama-cpp-python ✓

**New for Phase 3:**
- `streamlit` – Web UI framework
- `streamlit-option-menu` (optional) – Enhanced sidebar menus
- Phi-3-Mini GGUF model (~2.3GB, download from HuggingFace)

**Installation:**
```bash
pip install streamlit streamlit-option-menu
```

---

## Phase 3 Exit Criteria

### ✓ Functional Requirements
- [ ] Phi-3-Mini loads and generates text without error
- [ ] Responder generates fluent Vietnamese answers
- [ ] No hallucination (answers grounded in retrieved context)
- [ ] Source attribution works (shows which chunks used)
- [ ] Streamlit UI renders correctly
- [ ] End-to-end pipeline works (question → answer in <3s)

### ✓ Performance Requirements
- [ ] Response generation: <2 seconds per answer
- [ ] Streamlit UI latency: <1 second interaction response
- [ ] Total end-to-end: <3 seconds (retrieval + response)
- [ ] Memory usage: <8GB on 4-core CPU machine

### ✓ Quality Requirements
- [ ] 5+ integration test cases passing
- [ ] No Python crashes on edge case inputs
- [ ] Graceful error handling (invalid questions, network errors)
- [ ] Clear source citations
- [ ] Helpful error messages to users

### ✓ Documentation
- [ ] README.md with setup & usage
- [ ] Inline code docstrings complete
- [ ] Example questions and expected answers
- [ ] Known limitations documented

---

## Phi-3-Mini Model Details

**Model Specs:**
- Name: Phi-3-mini-4k-instruct
- Parameters: 3.8 billion
- Format: GGUF (quantized, CPU-friendly)
- File size: ~2.3GB
- Context window: 4096 tokens
- License: MIT

**Download Sources:**
- HuggingFace: `microsoft/Phi-3-mini-4k-instruct-gguf`
- Ollama: `ollama pull phi3:3.8b`
- Direct: [HF Model Card](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)

**Performance on CPU:**
- Inference: ~50-100ms per token
- Response time (256 tokens): ~12-25 seconds
- Memory: ~4-5GB (quantized)

---

## Known Constraints

### Hardware
- Target: 8GB RAM, 4 CPU cores (no GPU)
- Phi-3-Mini runs on CPU (slow but functional)
- Total models: Embedder (1.2GB) + Responder (2.3GB) = ~3.5GB loaded

### Latency
- Embedding: ~5ms
- Retrieval: ~8ms
- Response generation: ~12-25 seconds (Phi-3 on CPU)
- **Total: ~25-40 seconds per query** ⚠️ (trade-off: accuracy vs speed)

### Alternatives if Speed Critical
- Use smaller Phi-2 (~2.7B)
- Use faster embedder (e.g., FastText)
- Implement response caching
- Use Quantization (Q4 instead of Q8)

---

## Testing Strategy

**Unit Tests:**
- `ResponseGenerator` initialization
- Prompt construction
- Source extraction
- Error handling

**Integration Tests:**
- Full pipeline: question → answer
- Vietnamese input/output validation
- Source tracking
- Latency benchmarks

**UI Tests:**
- Streamlit component rendering
- Input/output flow
- Error display
- Session state persistence

**Manual Tests:**
- Sample HR questions (5+)
- Edge cases (very long questions, unclear intent)
- Performance on target hardware

---

## Next Steps

### Immediate (Today)
1. ✅ Create Phase 3 planning documents
2. Create `src/responder.py` module
3. Download/verify Phi-3-Mini model
4. Create `streamlit_app.py` UI

### Short Term (Next 2-3 hours)
5. Implement end-to-end integration test
6. Test on target hardware (8GB, 4-core)
7. Debug and optimize

### Finalization
8. Create README & documentation
9. Commit all code to GitHub
10. Generate final project summary

---

## Phase 3 Checklist

- [ ] Phase 3 planning documents (CONTEXT, PLAN, TODO)
- [ ] `src/responder.py` implemented & tested
- [ ] Phi-3-Mini model downloaded & verified
- [ ] `streamlit_app.py` UI built
- [ ] Integration test passing
- [ ] Latency benchmarks measured
- [ ] README.md created
- [ ] All code committed to GitHub
- [ ] Final project summary generated

---

## Success Criteria

**By End of Phase 3:**
- ✅ Working Vietnamese HR chatbot
- ✅ Fully local (no cloud APIs)
- ✅ <3 second query latency (retrieval + response)
- ✅ Zero hallucination (grounded answers)
- ✅ Production-ready code
- ✅ Deployable on 8GB CPU machine

---

**Phase 3 Status:** Planning Complete → Ready for Implementation

**Estimated Remaining Time:** 3-4 hours (accelerated from 4-5 hours due to Phase 2 performance gains)

**Target Completion:** May 11, 2026 (Today/Tomorrow)

---

*Last Updated: May 11, 2026*
