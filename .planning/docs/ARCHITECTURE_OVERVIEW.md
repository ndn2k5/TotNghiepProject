# 📊 RAG Pipeline Architecture Overview

## Current Architecture Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INPUT (Question)                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
        ┌─────────────────────────────┐
        │   Question Normalizer       │
        │  (Vietnamese text cleanup)  │
        └────────────┬────────────────┘
                     │
                     ▼
        ┌─────────────────────────────────────┐
        │  Semantic Search (Embeddings)       │
        │  - Embed normalized question        │
        │  - Search ChromaDB with top-2       │
        │  - Retrieve relevant chunks         │
        └────────────┬──────────────────────────┘
                     │
                     ▼
        ┌──────────────────────────────────────────┐
        │    Context Assembly & Ranking            │
        │  - Format chunks into context            │
        │  - Sort by relevance (cosine distance)   │
        │  - Limit to 200 chars per chunk          │
        └────────────┬───────────────────────────────┘
                     │
                     ▼
        ┌──────────────────────────────────────────┐
        │    Prompt Construction                    │
        │  - Add system instruction (Vietnamese)   │
        │  - Append retrieved context              │
        │  - Append user question                  │
        └────────────┬───────────────────────────────┘
                     │
                     ▼
        ┌──────────────────────────────────────────┐
        │    LLM Generation (Qwen2.5-1.5B)        │
        │  - temperature: 0.1 (deterministic)      │
        │  - max_tokens: 128 (concise)             │
        │  - GPU acceleration (CUDA)               │
        └────────────┬───────────────────────────────┘
                     │
                     ▼
        ┌──────────────────────────────────────────┐
        │    Response Post-Processing              │
        │  - Remove artifacts                      │
        │  - Extract sources                       │
        │  - Calculate confidence                  │
        └────────────┬───────────────────────────────┘
                     │
                     ▼
        ┌──────────────────────────────────────────┐
        │    Final Response + Metadata             │
        │  - Answer text                           │
        │  - Sources (pages)                       │
        │  - Confidence score                      │
        │  - Latency metrics                       │
        └──────────────────────────────────────────┘
```

---

## 📥 Document Ingestion Pipeline

```
┌─────────────────┐
│   PDF File      │
│  handbook.pdf   │
└────────┬────────┘
         │
         ▼
   ┌──────────────────┐
   │  PDFExtractor    │ ◄─── EXTRACT TEXT FROM PDF
   │  (PyMuPDF/fitz)  │      
   │  - Extract text  │      
   │  - Keep metadata │      
   │  - Per-page info │      
   └────────┬─────────┘
            │
            ▼
   ┌──────────────────┐
   │  Text Chunking   │ ◄─── SPLIT INTO CHUNKS
   │  (LangChain)     │      
   │  - chunk_size=400│      
   │  - overlap=50    │      
   └────────┬─────────┘
            │
            ▼
   ┌──────────────────┐
   │  Text Embedding  │ ◄─── CONVERT TO VECTORS
   │  (sentence-tf)   │      
   │  - all-MiniLM-L6 │      
   │  - 384 dimensions│      
   └────────┬─────────┘
            │
            ▼
   ┌──────────────────┐
   │  Vector Store    │ ◄─── PERSISTENT STORAGE
   │  (ChromaDB)      │      
   │  - chroma_db/    │      
   │  - indexed chunks│      
   └──────────────────┘
```

✅ **Already has PDF→Text step** via `PDFExtractor.extract_all_text()`

---

## 🎯 Current Issues & Solutions

### Issue 1: Generic Responses
**Problem**: Model trả lời quá ngắn hoặc chung chung  
**Current**: `max_tokens=128, temperature=0.1`  
**Solution**: Better prompt engineering + context weighting

### Issue 2: Wrong Context Selection
**Problem**: Lấy chunks không liên quan  
**Current**: Top-2 cosine similarity  
**Solution**: Multi-stage ranking + semantic validation

### Issue 3: Lost Information
**Problem**: Context bị cắt, mất chi tiết quan trọng  
**Current**: 200 chars per chunk → quá ngắn  
**Solution**: Adaptive context sizing based on relevance

### Issue 4: No Domain Understanding
**Problem**: Model không hiểu HR domain  
**Current**: Generic prompt  
**Solution**: Domain-specific system prompt + examples

---

## 🚀 Proposed Improvements

### 1. **Enhanced Question Understanding**
```python
# Current: Basic normalization
question → normalize → embed → search

# Proposed: Multi-layer understanding
question 
  → extract keywords (HR domain)
  → detect question type (salary? vacation? contract?)
  → expand variations
  → generate sub-queries
  → search all variations
  → ensemble results
```

### 2. **Smarter Context Selection**
```python
# Current: Top-2 by distance
chunks = retrieve_top_k(question, k=2)

# Proposed: Multi-criteria ranking
chunks = retrieve_candidates(question, k=10)
scores = {
    'semantic': cosine_similarity,
    'keyword_match': keyword_overlap,
    'domain_relevance': has_hr_terms,
    'recency': is_recent_policy,
}
ranked = weighted_ensemble(scores)
selected = top_k(ranked, k=3)
```

### 3. **Better Prompt Engineering**
```python
# Current: Generic Vietnamese prompt
# "Bạn là trợ lý HR..."

# Proposed: Domain-aware with examples + instructions
# - Add HR-specific context
# - Include format examples
# - Add fallback strategies
# - Add quality checks
```

### 4. **Response Quality Filtering**
```python
# Current: Just post-process
# response → clean → output

# Proposed: Validate before returning
# response 
#   → check_length (too short = regenerate)
#   → check_relevance (not answering question = flag)
#   → check_language (confirm Vietnamese)
#   → check_grounding (based on context only)
#   → auto-retry if fails
```

---

## 📋 Implementation Priority

| Priority | Improvement | Impact | Effort |
|----------|------------|--------|--------|
| 🔴 HIGH | Better prompts + system instructions | ⭐⭐⭐⭐⭐ | 30 min |
| 🔴 HIGH | Multi-layer context ranking | ⭐⭐⭐⭐ | 1 hour |
| 🟡 MEDIUM | Domain keyword extraction | ⭐⭐⭐ | 30 min |
| 🟡 MEDIUM | Response validation loop | ⭐⭐⭐ | 45 min |
| 🟢 LOW | Add examples to prompt | ⭐⭐ | 20 min |

---

## ✅ What Already Works Well

1. ✅ PDF extraction (PDFExtractor)
2. ✅ Text chunking (LangChain)
3. ✅ Embeddings (sentence-transformers)
4. ✅ Vector storage (ChromaDB)
5. ✅ CUDA acceleration
6. ✅ Question normalization
7. ✅ Basic RAG pipeline

---

## 🔧 Next Steps

Want me to implement:
1. **Better prompts** - More specific, HR-focused instructions
2. **Multi-stage ranking** - Better context selection
3. **Domain keyword extraction** - Understand HR terms better
4. **Response validation** - Ensure quality answers

Cái nào em muốn làm trước? 🎯
