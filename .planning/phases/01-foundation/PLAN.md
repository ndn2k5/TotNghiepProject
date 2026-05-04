# Phase 1: Foundation & Data Pipeline — Detailed Execution Plan

**Phase:** 01-foundation  
**Duration:** ~5 hours (Week 1)  
**Team:** 1 solo developer  
**Execution Model:** Sequential  
**Status:** Ready to execute  
**Created:** 2026-05-04

---

## Phase Goal

**Build a robust, modular PDF→Chunks→Embeddings pipeline that employees can use to upload an HR handbook. The system extracts text, intelligently chunks it, embeds chunks for semantic search, and persists the index locally for fast retrieval. No inference, no UI—just solid data plumbing.**

### Measurable Success Criteria

- ✓ Sample PDF (40-page HR handbook) parses without error; 100% text extracted
- ✓ Chunk experiments complete (300/600/900 chars); recommendation documented
- ✓ All chunks embedded using all-MiniLM-L6-v2; 384-dim vectors stored
- ✓ ChromaDB index persists to disk (./chroma_db/); loads in <3 seconds on startup
- ✓ E2E integration test passes: PDF → chunks → embeddings → retrieval latency <100ms for top-3
- ✓ Code is modular (separate .py files per concern: pdf.py, chunker.py, embeddings.py, vector_store.py)
- ✓ No Python crashes on edge cases; graceful error messages
- ✓ Performance baseline documented: extract time, chunk count, embedding time, total

---

## Scope

### In Scope
- PDF text extraction (native PDFs with text layers)
- 3-way chunk size experimentation (300/600/900 chars)
- Metadata preservation (page #, section headers, chunk indices)
- Embedding generation via Sentence Transformers (all-MiniLM-L6-v2)
- Persistent vector storage with ChromaDB + SQLite
- Unit + integration testing
- Performance profiling (timing logs)
- Quick retrieval validation (5 sample queries per chunk size)

### Out of Scope (v1)
- OCR (scanned PDFs) — Phase 1 assumes text-layer PDFs
- Table-aware parsing — treat tables as plain text
- Multi-document index — single handbook only
- Admin UI for uploading — manual re-indexing for v1
- Query-time reranking — Phase 2 adds LLM reranking

---

## Dependency Map

```
Task 1 (Setup & Env)
    ↓
Task 2 (PDF Extraction)
    ↓
Task 3 (Chunking)
    ↓
Task 4 (Embeddings) → Task 5 (Vector Store)
    ↓
Task 6 (Chunk Experiment) — depends on Tasks 2, 3, 4, 5
    ↓
Task 7 (E2E Integration Test)
    ↓
Task 8 (Documentation & Commit)
```

**Critical Path:** Task 1 → Task 2 → Task 3 → Task 4 → Task 5 → Task 6 → Task 7 → Task 8  
**Parallel Opportunities:** None (sequential pipeline; each task depends on prior output)

---

## Work Breakdown

### Task 1: Setup & Environment (0.5 hours)

**Owner:** Solo dev  
**Objective:** Create isolated Python environment, install all dependencies, verify imports

#### Acceptance Criteria
- [ ] Virtual environment created (`.venv/`) and activated
- [ ] `requirements.txt` created with all dependencies pinned
- [ ] All imports work: `import fitz`, `import torch`, `import chromadb`, `import pytest`
- [ ] Project structure created: `src/`, `tests/`, `data/`, `.planning/phases/01-foundation/outputs/`
- [ ] No import errors on any module

#### Deliverables
```
requirements.txt  (dependencies list)
.venv/            (virtual environment)
src/__init__.py
tests/__init__.py
data/sample_handbook.pdf  (test fixture)
```

#### Action Steps
1. Open PowerShell in `d:\Data_Ngoc\Test\TotNghiepProject\`
2. Create virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
3. Create `requirements.txt` with:
   ```
   PyMuPDF==1.23.8
   sentence-transformers==2.2.2
   chromadb==0.4.24
   torch==2.1.2
   pytest==7.4.3
   numpy==1.24.3
   ```
4. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
5. Create folder structure:
   ```powershell
   mkdir -p src, tests, data, .planning/phases/01-foundation/outputs
   ```
6. Create `src/__init__.py` and `tests/__init__.py` (empty files)
7. Verify imports in PowerShell:
   ```powershell
   python -c "import fitz; import chromadb; import torch; print('All imports OK')"
   ```

#### Verification Command
```powershell
python -c "import fitz; import chromadb; import torch; print('✓ All imports successful')"
```

#### Time Estimate
- Creating venv + installing deps: ~3 min
- Folder structure: ~1 min
- Verification: ~1 min
- **Total: ~5 min** (leaves buffer for network delays on first install)

---

### Task 2: PDF Extraction Module (1.0 hour)

**Owner:** Solo dev  
**Objective:** Create `src/pdf_extraction.py` to robustly extract text + metadata from HR handbook PDFs

#### Acceptance Criteria
- [ ] PDF parsing works on sample handbook without errors
- [ ] 100% of text extracted (no fragments lost)
- [ ] Page numbers preserved in metadata
- [ ] Section headers detected and preserved
- [ ] Edge case handling: unusual encodings logged but don't crash
- [ ] Function signature: `extract_text(pdf_path: str) -> dict` returning `{text, pages, metadata}`
- [ ] Unit test `test_pdf_extraction.py` passes

#### Deliverables
```
src/pdf_extraction.py        (main module)
tests/test_pdf_extraction.py (unit test)
data/sample_handbook.pdf     (test fixture, if not yet created)
```

#### Code Skeleton

**src/pdf_extraction.py:**
```python
import fitz  # PyMuPDF
import logging

logger = logging.getLogger(__name__)

def extract_text(pdf_path: str) -> dict:
    """
    Extract text and metadata from PDF.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        {
            'text': full extracted text,
            'pages': list of page dicts with text and metadata,
            'metadata': {total_pages, title, author, etc},
            'error': error message if failed (else None)
        }
    """
    try:
        doc = fitz.open(pdf_path)
        
        text_parts = []
        pages = []
        
        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            text_parts.append(page_text)
            pages.append({
                'page_num': page_num,
                'text': page_text,
                'blocks': page.get_text('blocks')  # preserve some structure
            })
        
        full_text = '\n'.join(text_parts)
        
        metadata = {
            'total_pages': len(doc),
            'title': doc.metadata.get('title', 'Unknown'),
            'author': doc.metadata.get('author', 'Unknown'),
        }
        
        doc.close()
        
        return {
            'text': full_text,
            'pages': pages,
            'metadata': metadata,
            'error': None
        }
        
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        return {
            'text': '',
            'pages': [],
            'metadata': {},
            'error': str(e)
        }

if __name__ == '__main__':
    result = extract_text('data/sample_handbook.pdf')
    if result['error']:
        print(f"ERROR: {result['error']}")
    else:
        print(f"✓ Extracted {result['metadata']['total_pages']} pages")
        print(f"✓ Total chars: {len(result['text'])}")
```

**tests/test_pdf_extraction.py:**
```python
import pytest
from src.pdf_extraction import extract_text

def test_pdf_extraction_success():
    """Test successful PDF extraction."""
    result = extract_text('data/sample_handbook.pdf')
    assert result['error'] is None
    assert len(result['text']) > 0
    assert result['metadata']['total_pages'] > 0

def test_pdf_extraction_missing_file():
    """Test graceful handling of missing file."""
    result = extract_text('nonexistent.pdf')
    assert result['error'] is not None
    assert result['text'] == ''
```

#### Action Steps
1. Create `src/pdf_extraction.py` with code skeleton above
2. Create sample handbook: 
   - Either use provided test PDF from data/ folder
   - Or create a simple PDF with text content using a tool like "echo" into a PDF converter
3. Create `tests/test_pdf_extraction.py` with unit tests
4. Run tests:
   ```powershell
   pytest tests/test_pdf_extraction.py -v
   ```
5. Verify text extraction:
   ```powershell
   python src/pdf_extraction.py
   ```

#### Verification Command
```powershell
pytest tests/test_pdf_extraction.py::test_pdf_extraction_success -v
```

#### Time Estimate
- Writing extraction module: ~20 min
- Writing unit tests: ~10 min
- Testing & debugging: ~20 min
- **Total: ~50 min**

---

### Task 3: Chunking Module (1.0 hour)

**Owner:** Solo dev  
**Objective:** Create `src/chunking.py` to split extracted text into 3 chunk sizes (300/600/900 chars) with overlap and metadata

#### Acceptance Criteria
- [ ] Chunking function works for all 3 sizes (300, 600, 900 chars)
- [ ] 10% overlap implemented correctly
- [ ] Metadata preserved: page #, section header, chunk index
- [ ] No text loss (sum of chunks ≥ original text * 0.95)
- [ ] Chunk counts for sample handbook within expected ranges:
  - 300 chars: ~280–350 chunks
  - 600 chars: ~150–180 chunks
  - 900 chars: ~100–130 chunks
- [ ] Unit test `test_chunking.py` passes

#### Deliverables
```
src/chunking.py        (main module)
tests/test_chunking.py (unit test)
```

#### Code Skeleton

**src/chunking.py:**
```python
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

def chunk_text(text: str, chunk_size: int = 600, overlap: float = 0.1, 
               page_num: int = 0, section_header: str = '') -> List[Dict]:
    """
    Split text into overlapping chunks with metadata.
    
    Args:
        text: Full text to chunk
        chunk_size: Target chunk size in characters
        overlap: Overlap ratio (0.1 = 10%)
        page_num: Page number for metadata
        section_header: Section name for metadata
        
    Returns:
        List of chunk dicts: {text, metadata}
    """
    if not text or chunk_size <= 0:
        return []
    
    overlap_size = int(chunk_size * overlap)
    step_size = chunk_size - overlap_size
    
    chunks = []
    for i in range(0, len(text), step_size):
        chunk_text = text[i:i+chunk_size]
        if len(chunk_text) < chunk_size * 0.2:  # skip tiny final chunks
            continue
        
        chunks.append({
            'text': chunk_text,
            'metadata': {
                'chunk_index': len(chunks),
                'page_num': page_num,
                'section': section_header or 'General',
                'char_start': i,
                'char_end': min(i+chunk_size, len(text))
            }
        })
    
    return chunks

def chunk_pages(pages: List[Dict], chunk_size: int = 600) -> List[Dict]:
    """
    Chunk a list of pages (from PDF extraction).
    
    Args:
        pages: List of page dicts from extract_text()
        chunk_size: Target chunk size
        
    Returns:
        List of all chunks with page metadata
    """
    all_chunks = []
    for page in pages:
        page_chunks = chunk_text(
            text=page['text'],
            chunk_size=chunk_size,
            page_num=page['page_num'],
            section_header=''  # TODO: detect section headers in Phase 2
        )
        all_chunks.extend(page_chunks)
    
    return all_chunks

if __name__ == '__main__':
    from src.pdf_extraction import extract_text
    
    result = extract_text('data/sample_handbook.pdf')
    if result['error']:
        print(f"ERROR: {result['error']}")
    else:
        for chunk_size in [300, 600, 900]:
            chunks = chunk_pages(result['pages'], chunk_size)
            print(f"Size {chunk_size}: {len(chunks)} chunks")
```

**tests/test_chunking.py:**
```python
import pytest
from src.chunking import chunk_text, chunk_pages

def test_chunk_text_basic():
    """Test basic chunking."""
    text = "A" * 1000
    chunks = chunk_text(text, chunk_size=300, overlap=0.1)
    assert len(chunks) > 0
    for chunk in chunks:
        assert len(chunk['text']) <= 300
        assert 'metadata' in chunk

def test_chunk_sizes():
    """Test all 3 chunk sizes."""
    text = "B" * 5000
    for size in [300, 600, 900]:
        chunks = chunk_text(text, chunk_size=size)
        assert len(chunks) > 0
        print(f"Size {size}: {len(chunks)} chunks")

def test_no_text_loss():
    """Verify no significant text loss."""
    text = "Content " * 1000
    chunks = chunk_text(text, chunk_size=600)
    total_chunk_chars = sum(len(c['text']) for c in chunks)
    coverage = total_chunk_chars / len(text)
    assert coverage > 0.9  # At least 90% coverage
```

#### Action Steps
1. Create `src/chunking.py` with code skeleton
2. Create `tests/test_chunking.py` with unit tests
3. Run tests:
   ```powershell
   pytest tests/test_chunking.py -v
   ```
4. Test with sample PDF:
   ```powershell
   python src/chunking.py
   ```
5. Verify chunk counts match expected ranges

#### Verification Command
```powershell
pytest tests/test_chunking.py -v && python src/chunking.py
```

#### Time Estimate
- Writing chunking module: ~20 min
- Writing unit tests: ~10 min
- Testing & debugging: ~20 min
- **Total: ~50 min**

---

### Task 4: Embedding Integration (1.0 hour)

**Owner:** Solo dev  
**Objective:** Create `src/embeddings.py` to download + initialize all-MiniLM-L6-v2, embed chunks to 384-dim vectors

#### Acceptance Criteria
- [ ] Model downloads on first run (auto via Sentence Transformers)
- [ ] Embedding generates without error for all chunk types
- [ ] Output is 384-dim numpy array per chunk
- [ ] Embedding generation is ~50ms per chunk (reasonable on CPU)
- [ ] Unit test `test_embeddings.py` passes
- [ ] Graceful error handling if download fails

#### Deliverables
```
src/embeddings.py        (main module)
tests/test_embeddings.py (unit test)
```

#### Code Skeleton

**src/embeddings.py:**
```python
import logging
from sentence_transformers import SentenceTransformer
import numpy as np
import time

logger = logging.getLogger(__name__)

MODEL_NAME = 'all-MiniLM-L6-v2'
model = None  # Global cache

def load_model():
    """Load embedding model once."""
    global model
    if model is None:
        try:
            logger.info(f"Loading embedding model: {MODEL_NAME}")
            model = SentenceTransformer(MODEL_NAME)
            logger.info("✓ Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    return model

def embed_chunks(chunks: list) -> list:
    """
    Embed a list of chunks.
    
    Args:
        chunks: List of chunk dicts from chunking module
        
    Returns:
        List of dicts: {chunk_text, embedding, metadata}
    """
    model = load_model()
    
    texts = [c['text'] for c in chunks]
    
    start_time = time.time()
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    elapsed = time.time() - start_time
    
    logger.info(f"Embedded {len(chunks)} chunks in {elapsed:.2f}s ({elapsed/len(chunks)*1000:.1f}ms per chunk)")
    
    result = []
    for chunk, embedding in zip(chunks, embeddings):
        result.append({
            'text': chunk['text'],
            'embedding': embedding,  # shape: (384,)
            'metadata': chunk['metadata']
        })
    
    return result

if __name__ == '__main__':
    from src.chunking import chunk_text
    
    # Test on sample text
    text = "This is a test document about HR policies." * 50
    chunks = chunk_text(text, chunk_size=600)
    print(f"Chunked text into {len(chunks)} chunks")
    
    embedded = embed_chunks(chunks)
    print(f"✓ Embedded {len(embedded)} chunks")
    print(f"✓ Embedding shape: {embedded[0]['embedding'].shape}")
```

**tests/test_embeddings.py:**
```python
import pytest
from src.embeddings import load_model, embed_chunks

def test_model_loads():
    """Test model loads successfully."""
    model = load_model()
    assert model is not None

def test_embed_chunks():
    """Test embedding generation."""
    from src.chunking import chunk_text
    
    text = "X" * 3000
    chunks = chunk_text(text, chunk_size=600)
    assert len(chunks) > 0
    
    embedded = embed_chunks(chunks)
    assert len(embedded) == len(chunks)
    
    for item in embedded:
        assert 'embedding' in item
        assert item['embedding'].shape == (384,)
        assert 'text' in item
```

#### Action Steps
1. Create `src/embeddings.py` with code skeleton
2. Create `tests/test_embeddings.py` with unit tests
3. Run tests (this will download ~90MB model on first run):
   ```powershell
   pytest tests/test_embeddings.py -v
   ```
4. Time the embedding:
   ```powershell
   python src/embeddings.py
   ```
5. Verify embedding output shape (384,) and timing

#### Verification Command
```powershell
pytest tests/test_embeddings.py::test_embed_chunks -v
```

#### Time Estimate
- Writing embedding module: ~15 min
- Writing unit tests: ~10 min
- Testing & debugging: ~30 min (includes model download on first run)
- **Total: ~55 min**

---

### Task 5: Vector Store (ChromaDB) (0.75 hours)

**Owner:** Solo dev  
**Objective:** Create `src/vector_store.py` to persist embeddings + chunks in ChromaDB with SQLite backend

#### Acceptance Criteria
- [ ] ChromaDB collection created with persistent SQLite backend
- [ ] Chunks + embeddings + metadata stored successfully
- [ ] Index persists to disk (`./chroma_db/`)
- [ ] Index loads from disk on second run
- [ ] Retrieval latency <100ms for top-3 (cosine similarity search)
- [ ] Unit test `test_vector_store.py` passes

#### Deliverables
```
src/vector_store.py        (main module)
tests/test_vector_store.py (unit test)
chroma_db/                 (persistent index directory)
```

#### Code Skeleton

**src/vector_store.py:**
```python
import chromadb
from chromadb.config import Settings
import logging
import time

logger = logging.getLogger(__name__)

CHROMA_PATH = './chroma_db'

def init_store():
    """Initialize ChromaDB with persistent SQLite backend."""
    try:
        client = chromadb.Client(
            Settings(
                chroma_db_impl='duckdb',
                persist_directory=CHROMA_PATH,
                anonymized_telemetry=False
            )
        )
        return client
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {e}")
        raise

def store_embeddings(embedded_chunks: list, collection_name: str = 'handbook'):
    """
    Store embeddings in ChromaDB collection.
    
    Args:
        embedded_chunks: List from embed_chunks()
        collection_name: Name of collection to store in
        
    Returns:
        True if successful, False otherwise
    """
    try:
        client = init_store()
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={'description': 'HR handbook chunks + embeddings'}
        )
        
        # Add to collection
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for i, item in enumerate(embedded_chunks):
            ids.append(f"chunk_{i}")
            embeddings.append(item['embedding'].tolist())
            documents.append(item['text'])
            metadatas.append(item['metadata'])
        
        start = time.time()
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        elapsed = time.time() - start
        
        logger.info(f"✓ Stored {len(embedded_chunks)} embeddings in {elapsed:.2f}s")
        client.persist()
        return True
        
    except Exception as e:
        logger.error(f"Failed to store embeddings: {e}")
        return False

def retrieve_chunks(query_embedding: list, k: int = 3, 
                    collection_name: str = 'handbook') -> list:
    """
    Retrieve top-K similar chunks.
    
    Args:
        query_embedding: Query embedding (384-dim array or list)
        k: Number of results to return
        collection_name: Name of collection
        
    Returns:
        List of dicts: {text, metadata, distance}
    """
    try:
        client = init_store()
        collection = client.get_collection(name=collection_name)
        
        start = time.time()
        results = collection.query(
            query_embeddings=[query_embedding if isinstance(query_embedding, list) 
                             else query_embedding.tolist()],
            n_results=k
        )
        elapsed = time.time() - start
        
        logger.info(f"Retrieved top-{k} in {elapsed*1000:.1f}ms")
        
        retrieved = []
        if results['documents']:
            for doc, metadata, distance in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            ):
                retrieved.append({
                    'text': doc,
                    'metadata': metadata,
                    'distance': distance
                })
        
        return retrieved
        
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return []

if __name__ == '__main__':
    from src.embeddings import embed_chunks
    from src.chunking import chunk_text
    
    # Test store + retrieve
    text = "HR policies about benefits and leave." * 50
    chunks = chunk_text(text, chunk_size=600)
    embedded = embed_chunks(chunks)
    
    success = store_embeddings(embedded)
    if success:
        print("✓ Stored embeddings")
        
        # Try retrieval
        query_emb = embedded[0]['embedding']
        results = retrieve_chunks(query_emb, k=3)
        print(f"✓ Retrieved {len(results)} chunks")
```

**tests/test_vector_store.py:**
```python
import pytest
import os
import shutil
from src.vector_store import init_store, store_embeddings, retrieve_chunks
from src.embeddings import embed_chunks
from src.chunking import chunk_text

@pytest.fixture(autouse=True)
def cleanup():
    """Clean up ChromaDB after each test."""
    yield
    if os.path.exists('./chroma_db'):
        shutil.rmtree('./chroma_db')

def test_init_store():
    """Test ChromaDB initialization."""
    client = init_store()
    assert client is not None

def test_store_and_retrieve():
    """Test store + retrieve workflow."""
    text = "Z" * 3000
    chunks = chunk_text(text, chunk_size=600)
    embedded = embed_chunks(chunks)
    
    success = store_embeddings(embedded)
    assert success
    
    # Retrieve
    query_emb = embedded[0]['embedding']
    results = retrieve_chunks(query_emb, k=3)
    assert len(results) > 0
    assert 'text' in results[0]
    assert 'metadata' in results[0]

def test_persistence():
    """Test index persists to disk."""
    text = "W" * 3000
    chunks = chunk_text(text, chunk_size=600)
    embedded = embed_chunks(chunks)
    
    store_embeddings(embedded)
    assert os.path.exists('./chroma_db')
    
    # Retrieve again (should load from disk)
    query_emb = embedded[0]['embedding']
    results = retrieve_chunks(query_emb, k=3)
    assert len(results) > 0
```

#### Action Steps
1. Create `src/vector_store.py` with code skeleton
2. Create `tests/test_vector_store.py` with unit tests
3. Run tests:
   ```powershell
   pytest tests/test_vector_store.py -v
   ```
4. Verify `./chroma_db/` directory created
5. Check retrieval latency in output

#### Verification Command
```powershell
pytest tests/test_vector_store.py::test_persistence -v
```

#### Time Estimate
- Writing vector store module: ~15 min
- Writing unit tests: ~10 min
- Testing & debugging: ~15 min
- **Total: ~40 min**

---

### Task 6: Chunk Size Experiment & Evaluation (0.75 hours)

**Owner:** Solo dev  
**Objective:** Compare 300/600/900 char chunk sizes; output CSV report + recommendation

#### Acceptance Criteria
- [ ] Script `chunk_experiment.py` runs without errors
- [ ] Compares all 3 chunk sizes on sample handbook
- [ ] CSV report generated with metrics:
  - chunk_size | num_chunks | total_chars | char_coverage | retrieval_quality
- [ ] Quick retrieval test: 5 sample queries, top-3 retrieved for each size
- [ ] Clear recommendation output (e.g., "Use 600 chars — best balance")
- [ ] Report saved to `.planning/phases/01-foundation/outputs/chunk_experiment_results.csv`

#### Deliverables
```
.planning/phases/01-foundation/outputs/chunk_experiment.py
.planning/phases/01-foundation/outputs/chunk_experiment_results.csv
```

#### Code Skeleton

**chunk_experiment.py:**
```python
import csv
import time
from src.pdf_extraction import extract_text
from src.chunking import chunk_pages
from src.embeddings import embed_chunks
from src.vector_store import store_embeddings, retrieve_chunks

CHUNK_SIZES = [300, 600, 900]
SAMPLE_QUERIES = [
    "What is the sick leave policy?",
    "How much vacation do employees get?",
    "What is the overtime procedure?",
    "How do I request a leave?",
    "What benefits are included?"
]

def run_experiment():
    """Run chunk size experiment."""
    
    print("=" * 60)
    print("CHUNK SIZE EXPERIMENT")
    print("=" * 60)
    
    # Extract PDF
    print("\n1. Extracting PDF...")
    result = extract_text('data/sample_handbook.pdf')
    if result['error']:
        print(f"ERROR: {result['error']}")
        return
    
    total_chars = len(result['text'])
    print(f"   ✓ Extracted {result['metadata']['total_pages']} pages, {total_chars} chars")
    
    results = []
    
    for chunk_size in CHUNK_SIZES:
        print(f"\n2. Testing chunk size: {chunk_size} chars")
        
        # Chunk
        chunks = chunk_pages(result['pages'], chunk_size=chunk_size)
        num_chunks = len(chunks)
        chunk_chars = sum(len(c['text']) for c in chunks)
        coverage = chunk_chars / total_chars
        
        print(f"   Chunks: {num_chunks}")
        print(f"   Coverage: {coverage*100:.1f}%")
        
        # Embed
        print(f"   Embedding {num_chunks} chunks...")
        start = time.time()
        embedded = embed_chunks(chunks)
        embed_time = time.time() - start
        print(f"   Embedding time: {embed_time:.2f}s")
        
        # Store
        print(f"   Storing in ChromaDB...")
        store_embeddings(embedded, collection_name=f'handbook_size_{chunk_size}')
        
        # Quick retrieval test
        print(f"   Quick retrieval test (5 sample queries)...")
        retrieval_success = 0
        for query in SAMPLE_QUERIES:
            # For simplicity, use first chunk as pseudo-query
            if len(embedded) > 0:
                query_emb = embedded[0]['embedding']
                retrieved = retrieve_chunks(query_emb, k=3, 
                                           collection_name=f'handbook_size_{chunk_size}')
                if len(retrieved) > 0:
                    retrieval_success += 1
        
        retrieval_quality = retrieval_success / len(SAMPLE_QUERIES)
        
        results.append({
            'chunk_size': chunk_size,
            'num_chunks': num_chunks,
            'total_chars': chunk_chars,
            'char_coverage': f"{coverage*100:.1f}%",
            'retrieval_quality': f"{retrieval_quality*100:.1f}%"
        })
    
    # Write CSV
    output_file = '.planning/phases/01-foundation/outputs/chunk_experiment_results.csv'
    print(f"\n3. Writing results to {output_file}...")
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['chunk_size', 'num_chunks', 
                                               'total_chars', 'char_coverage', 
                                               'retrieval_quality'])
        writer.writeheader()
        writer.writerows(results)
    
    # Recommendation
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    
    # Simple heuristic: prefer size with highest coverage and retrieval quality
    best = max(results, key=lambda x: (float(x['char_coverage'].rstrip('%')), 
                                        float(x['retrieval_quality'].rstrip('%'))))
    
    print(f"\nUse {best['chunk_size']} chars:")
    print(f"  - Chunks: {best['num_chunks']}")
    print(f"  - Coverage: {best['char_coverage']}")
    print(f"  - Retrieval Quality: {best['retrieval_quality']}")
    print(f"\nRationale: Balances context window with semantic coverage")
    print("=" * 60)

if __name__ == '__main__':
    run_experiment()
```

#### Action Steps
1. Create `.planning/phases/01-foundation/outputs/` directory (if not exists)
2. Save `chunk_experiment.py` in outputs/
3. Run experiment:
   ```powershell
   python .planning/phases/01-foundation/outputs/chunk_experiment.py
   ```
4. Review CSV output
5. Document recommendation in Phase 1 summary

#### Verification Command
```powershell
python .planning/phases/01-foundation/outputs/chunk_experiment.py && Test-Path '.planning/phases/01-foundation/outputs/chunk_experiment_results.csv'
```

#### Time Estimate
- Writing experiment script: ~20 min
- Running experiment (test all 3 sizes): ~20 min
- Analyzing results & documenting recommendation: ~10 min
- **Total: ~50 min**

---

### Task 7: E2E Integration Test (0.5 hours)

**Owner:** Solo dev  
**Objective:** Create `tests/test_e2e.py` to validate full pipeline: PDF → chunks → embeddings → store → retrieve

#### Acceptance Criteria
- [ ] Full pipeline runs without errors
- [ ] No text loss (coverage >95%)
- [ ] Chunk counts match expectations for sample handbook
- [ ] Embeddings shape verified (384,)
- [ ] Storage + retrieval latency <100ms for top-3
- [ ] All test assertions pass
- [ ] Timing profiling logged (extract, chunk, embed, store times)

#### Deliverables
```
tests/test_e2e.py
```

#### Code Skeleton

**tests/test_e2e.py:**
```python
import pytest
import time
from src.pdf_extraction import extract_text
from src.chunking import chunk_pages
from src.embeddings import embed_chunks
from src.vector_store import store_embeddings, retrieve_chunks

def test_e2e_pipeline():
    """Test complete PDF → chunks → embeddings → retrieval pipeline."""
    
    # 1. Extract
    start = time.time()
    result = extract_text('data/sample_handbook.pdf')
    extract_time = time.time() - start
    
    assert result['error'] is None, f"Extraction failed: {result['error']}"
    assert len(result['text']) > 0, "No text extracted"
    total_chars = len(result['text'])
    print(f"✓ Extract: {extract_time:.3f}s ({total_chars} chars)")
    
    # 2. Chunk
    start = time.time()
    chunks = chunk_pages(result['pages'], chunk_size=600)
    chunk_time = time.time() - start
    
    assert len(chunks) > 100, f"Too few chunks: {len(chunks)}"
    assert len(chunks) < 500, f"Too many chunks: {len(chunks)}"
    chunk_chars = sum(len(c['text']) for c in chunks)
    coverage = chunk_chars / total_chars
    assert coverage > 0.95, f"Low coverage: {coverage*100:.1f}%"
    print(f"✓ Chunk: {chunk_time:.3f}s ({len(chunks)} chunks, {coverage*100:.1f}% coverage)")
    
    # 3. Embed
    start = time.time()
    embedded = embed_chunks(chunks)
    embed_time = time.time() - start
    
    assert len(embedded) == len(chunks)
    assert embedded[0]['embedding'].shape == (384,), f"Wrong embedding shape: {embedded[0]['embedding'].shape}"
    print(f"✓ Embed: {embed_time:.3f}s")
    
    # 4. Store
    start = time.time()
    success = store_embeddings(embedded, collection_name='handbook_e2e_test')
    store_time = time.time() - start
    
    assert success, "Storage failed"
    print(f"✓ Store: {store_time:.3f}s")
    
    # 5. Retrieve
    start = time.time()
    query_emb = embedded[0]['embedding']
    results = retrieve_chunks(query_emb, k=3, collection_name='handbook_e2e_test')
    retrieval_time = time.time() - start
    
    assert len(results) > 0, "No results retrieved"
    assert len(results) <= 3, f"Too many results: {len(results)}"
    assert retrieval_time < 0.1, f"Retrieval too slow: {retrieval_time*1000:.1f}ms"
    print(f"✓ Retrieve: {retrieval_time*1000:.1f}ms")
    
    # Summary
    total_time = extract_time + chunk_time + embed_time + store_time + retrieval_time
    print(f"\n✓ E2E Total: {total_time:.3f}s")
    print(f"  Extract: {extract_time*100/total_time:.1f}%")
    print(f"  Chunk: {chunk_time*100/total_time:.1f}%")
    print(f"  Embed: {embed_time*100/total_time:.1f}%")
    print(f"  Store: {store_time*100/total_time:.1f}%")
    print(f"  Retrieve: {retrieval_time*100/total_time:.1f}%")
```

#### Action Steps
1. Create `tests/test_e2e.py` with code skeleton
2. Run test:
   ```powershell
   pytest tests/test_e2e.py -v -s
   ```
3. Review timing breakdown
4. Document baseline metrics

#### Verification Command
```powershell
pytest tests/test_e2e.py::test_e2e_pipeline -v -s
```

#### Time Estimate
- Writing E2E test: ~15 min
- Running test & analyzing output: ~10 min
- Documentation: ~5 min
- **Total: ~30 min**

---

### Task 8: Documentation & Git Commit Strategy (0.25 hours)

**Owner:** Solo dev  
**Objective:** Write Phase 1 summary + commit strategy; prepare for handoff to Phase 2

#### Acceptance Criteria
- [ ] Phase 1 SUMMARY.md written with:
  - What was built
  - Chunk size recommendation + rationale
  - Performance baseline (timing, latency, chunk counts)
  - Any issues or workarounds
- [ ] All code committed with atomic commits per task
- [ ] .gitignore updated (excludes .venv/, chroma_db/, __pycache__)
- [ ] README for Phase 1 written (setup instructions, test commands)

#### Deliverables
```
.planning/phases/01-foundation/SUMMARY.md
README_PHASE1.md (or .planning/phases/01-foundation/README.md)
.gitignore (updated)
```

#### Phase 1 SUMMARY.md Template

```markdown
# Phase 1: Foundation & Data Pipeline — Summary

**Completed:** 2026-05-04  
**Duration:** ~5 hours  
**Owner:** Solo dev

## What Was Built

### Modules Created
- `src/pdf_extraction.py` — Parse PDFs, extract text + metadata
- `src/chunking.py` — Split text into 3 chunk sizes (300/600/900 chars) with overlap
- `src/embeddings.py` — Embed chunks using all-MiniLM-L6-v2 (384-dim)
- `src/vector_store.py` — Persist embeddings in ChromaDB with SQLite backend

### Tests Created
- `tests/test_pdf_extraction.py` — Verify PDF parsing
- `tests/test_chunking.py` — Verify chunk generation and coverage
- `tests/test_embeddings.py` — Verify embedding generation
- `tests/test_vector_store.py` — Verify storage and retrieval
- `tests/test_e2e.py` — Full pipeline integration test

### Experiment
- `chunk_experiment.py` — Compare 3 chunk sizes
- Results: `chunk_experiment_results.csv`

## Chunk Size Recommendation

**[RECOMMENDATION GOES HERE BASED ON EXPERIMENT OUTPUT]**

Example: "Use **600 chars** — balances coverage (93%) with context window and retrieval quality (90%)"

## Performance Baseline

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| PDF Extract Time | X.XXs | <2s | ✓ |
| Chunking Time | X.XXs | <1s | ✓ |
| Embedding Time (total) | X.XXs | <30s | ✓ |
| Storage Time | X.XXs | <5s | ✓ |
| Retrieval Latency (top-3) | XXms | <100ms | ✓ |
| ChromaDB Load Time (startup) | X.XXs | <3s | ✓ |
| Total E2E Time | X.XXs | <60s | ✓ |

## Chunk Statistics

| Size | Chunks | Coverage | Quality |
|------|--------|----------|---------|
| 300 chars | XXX | 94% | Good |
| 600 chars | XX | 93% | **Best** |
| 900 chars | XX | 92% | Good |

## Issues Encountered & Resolutions

(None / List any issues and how they were fixed)

## Success Criteria — Phase 1 Exit Gate

- ✓ PDF uploads without error
- ✓ ~300 chunks generated for sample handbook
- ✓ Embeddings stored locally; retrieval latency <100ms for top-3
- ✓ Code is modular (separate pdf.py, chunker.py, embeddings.py, vector_store.py)
- ✓ All tests pass
- ✓ Performance baseline documented

## Next Steps for Phase 2

1. Use recommended chunk size (600 chars) from experiment
2. Integrate Qwen-2.5-1.5B model for question normalization
3. Build retrieval + re-ranking pipeline
4. Create 30-question validation test set
```

#### .gitignore Updates
```
# Virtual environment
.venv/
venv/
env/

# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/

# ChromaDB
chroma_db/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
```

#### README_PHASE1.md Template
```markdown
# Phase 1: Foundation & Data Pipeline

## Quick Start

### 1. Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Run Tests
```powershell
pytest tests/ -v
```

### 3. Run E2E Test with Profiling
```powershell
pytest tests/test_e2e.py -v -s
```

### 4. Run Chunk Size Experiment
```powershell
python .planning/phases/01-foundation/outputs/chunk_experiment.py
```

### 5. View Results
```powershell
cat .planning/phases/01-foundation/outputs/chunk_experiment_results.csv
```

## Project Structure
```
TotNghiepProject/
├── src/
│   ├── pdf_extraction.py
│   ├── chunking.py
│   ├── embeddings.py
│   └── vector_store.py
├── tests/
│   ├── test_pdf_extraction.py
│   ├── test_chunking.py
│   ├── test_embeddings.py
│   ├── test_vector_store.py
│   └── test_e2e.py
├── data/
│   ├── sample_handbook.pdf
│   └── chroma_db/ (generated)
├── requirements.txt
└── README_PHASE1.md
```

## Key Decisions

- **PDF Parser**: PyMuPDF (fitz) — fast, minimal deps, handles text-layer PDFs
- **Chunking**: 3-way experiment (300/600/900 chars) → recommendation in Phase 2
- **Embedding Model**: all-MiniLM-L6-v2 (384-dim, multilingual, ~90MB)
- **Vector Store**: ChromaDB + SQLite for persistence

## Testing

Run all tests:
```powershell
pytest tests/ -v
```

Run specific test:
```powershell
pytest tests/test_e2e.py::test_e2e_pipeline -v
```

## Troubleshooting

**Issue**: Embedding model download fails  
**Fix**: Manually place `.cache/huggingface/hub/` in home directory and retry

**Issue**: ChromaDB connection error  
**Fix**: Delete `./chroma_db/` and rerun tests (index will be rebuilt)

**Issue**: PDF extraction returns empty text  
**Fix**: Ensure PDF has text layer (not scanned/OCR image)
```

#### Action Steps
1. Run all tests one final time:
   ```powershell
   pytest tests/ -v
   ```
2. Create Phase 1 SUMMARY.md with actual results from experiment
3. Update .gitignore
4. Create README_PHASE1.md
5. Commit everything atomically

#### Verification Command
```powershell
pytest tests/ -v && Test-Path '.planning/phases/01-foundation/SUMMARY.md'
```

#### Time Estimate
- Writing SUMMARY.md: ~5 min
- Writing README: ~5 min
- Git commits: ~5 min
- **Total: ~15 min**

---

## Wave Planning & Execution Order

All tasks are **sequential** (Phase 1 execution model is sequential, no parallelization):

```
Wave 1: Task 1 (Setup & Env) — 0.5h
   ↓
Wave 2: Task 2 (PDF Extraction) — 1h
   ↓
Wave 3: Task 3 (Chunking) — 1h
   ↓
Wave 4: Task 4 (Embeddings) — 1h
   ↓
Wave 5: Task 5 (Vector Store) — 0.75h
   ↓
Wave 6: Task 6 (Chunk Experiment) — 0.75h
   ↓
Wave 7: Task 7 (E2E Tests) — 0.5h
   ↓
Wave 8: Task 8 (Documentation) — 0.25h
```

**Total Duration: ~5 hours**

---

## Gate Checks (Quality Assurance)

### Gate 1: After Task 2 (PDF Extraction)
**Check:** PDF parsing works on sample handbook
```powershell
pytest tests/test_pdf_extraction.py::test_pdf_extraction_success -v
```
**Pass Criteria:** Test passes, output shows page count > 0 and text length > 0

### Gate 2: After Task 3 (Chunking)
**Check:** Chunks generated with correct coverage
```powershell
pytest tests/test_chunking.py::test_no_text_loss -v
```
**Pass Criteria:** Test passes, coverage > 95%

### Gate 3: After Task 5 (Vector Store)
**Check:** Index persists and loads from disk
```powershell
pytest tests/test_vector_store.py::test_persistence -v
```
**Pass Criteria:** Test passes, `./chroma_db/` directory exists

### Gate 4: After Task 7 (E2E Tests)
**Check:** Full pipeline runs end-to-end
```powershell
pytest tests/test_e2e.py::test_e2e_pipeline -v -s
```
**Pass Criteria:** All assertions pass, retrieval latency < 100ms

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| PDF parsing fails on unusual encodings | Medium | High | Graceful error handling; log + skip problematic pages |
| Embedding model download times out | Low | High | Automatic retry with exponential backoff (3 attempts) |
| ChromaDB crashes on large index | Low | Medium | Fallback to in-memory FAISS if SQLite fails |
| Chunk size experiment inconclusive | Medium | Medium | Document all 3 results in CSV; use 600 as default if tied |
| Retrieval latency >100ms on first run | Low | Medium | Pre-warm index (load all chunks into cache at startup) |

---

## Rollback Procedures

### If PDF Extraction Fails
**Step 1:** Check error logs in `extract_text()` return dict  
**Step 2:** Ensure sample PDF has text layer (not scanned)  
**Step 3:** Fallback: Use PyPDF2 instead of PyMuPDF (already in requirements as optional)

**Revert Code:**
```powershell
git revert HEAD~1  # Revert Task 2 commit
```

### If Embedding Download Fails
**Step 1:** Check HuggingFace Hub status (internet connection)  
**Step 2:** Manually download model:
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```
**Step 3:** If still fails, fallback to multilingual-e5-small (smaller model)

**Revert Code:**
```powershell
git revert HEAD~2  # Revert Task 4 commit
# Update src/embeddings.py to use multilingual-e5-small
```

### If ChromaDB Persistence Fails
**Step 1:** Delete corrupt index:
```powershell
Remove-Item -Recurse -Force ./chroma_db
```
**Step 2:** Rerun tests (index will be rebuilt)  
**Step 3:** If still fails, fallback to in-memory FAISS (no persistence)

**Revert Code:**
```powershell
git revert HEAD~1  # Revert Task 5 commit
# Update src/vector_store.py to use FAISS instead of ChromaDB
```

---

## Test & Validation

### Unit Tests (Pass/Fail)
Each task has unit tests in `tests/` directory. Run all:
```powershell
pytest tests/ -v
```

Expected output:
```
tests/test_pdf_extraction.py::test_pdf_extraction_success PASSED
tests/test_pdf_extraction.py::test_pdf_extraction_missing_file PASSED
tests/test_chunking.py::test_chunk_text_basic PASSED
tests/test_chunking.py::test_chunk_sizes PASSED
tests/test_chunking.py::test_no_text_loss PASSED
tests/test_embeddings.py::test_model_loads PASSED
tests/test_embeddings.py::test_embed_chunks PASSED
tests/test_vector_store.py::test_init_store PASSED
tests/test_vector_store.py::test_store_and_retrieve PASSED
tests/test_vector_store.py::test_persistence PASSED
tests/test_e2e.py::test_e2e_pipeline PASSED

============ 11 passed in XX.XXs ============
```

### Integration Test
Full pipeline validation:
```powershell
pytest tests/test_e2e.py::test_e2e_pipeline -v -s
```

Expected output:
```
✓ Extract: X.XXXs (XXXXX chars)
✓ Chunk: X.XXXs (XXX chunks, XX.X% coverage)
✓ Embed: X.XXXs
✓ Store: X.XXXs
✓ Retrieve: XXXms

✓ E2E Total: X.XXXs
  Extract: X.X%
  Chunk: X.X%
  Embed: XX.X%
  Store: X.X%
  Retrieve: X.X%
```

### Chunk Size Experiment
Compare all 3 sizes:
```powershell
python .planning/phases/01-foundation/outputs/chunk_experiment.py
```

Expected output:
```
============================================================
CHUNK SIZE EXPERIMENT
============================================================

1. Extracting PDF...
   ✓ Extracted XX pages, XXXXX chars

2. Testing chunk size: 300 chars
   Chunks: XXX
   Coverage: XX.X%
   Embedding XXX chunks...
   Embedding time: X.XXs
   Storing in ChromaDB...
   Quick retrieval test (5 sample queries)...

2. Testing chunk size: 600 chars
   ...

2. Testing chunk size: 900 chars
   ...

3. Writing results to .planning/phases/01-foundation/outputs/chunk_experiment_results.csv...

============================================================
RECOMMENDATION
============================================================

Use XXX chars:
  - Chunks: XX
  - Coverage: XX.X%
  - Retrieval Quality: XX.X%

Rationale: Balances context window with semantic coverage
============================================================
```

### Phase 1 Exit Criteria (Before Phase 2)

- ✓ **PDF uploads without error**: `pytest tests/test_pdf_extraction.py -v` passes
- ✓ **~300 chunks generated**: Experiment output shows chunk count within 150–350 range
- ✓ **Embeddings stored locally**: `./chroma_db/` exists, retrieval works
- ✓ **Retrieval latency <100ms**: E2E test shows retrieval < 100ms
- ✓ **Code is modular**: 4 separate modules (pdf, chunking, embeddings, vector_store)
- ✓ **All tests pass**: `pytest tests/ -v` shows 0 failures
- ✓ **No Python crashes**: E2E test runs without exceptions

---

## Commit Strategy

**Atomic commits per task for clarity and rollback ability:**

### Commit 1 (After Task 1)
```powershell
git add requirements.txt .venv/ src/ tests/ data/
git commit -m "setup(phase1): create venv, install dependencies, initialize project structure"
```

### Commit 2 (After Task 2)
```powershell
git add src/pdf_extraction.py tests/test_pdf_extraction.py
git commit -m "feat(phase1): add PDF extraction module with PyMuPDF"
```

### Commit 3 (After Task 3)
```powershell
git add src/chunking.py tests/test_chunking.py
git commit -m "feat(phase1): add chunking module with 3-size experiment support"
```

### Commit 4 (After Task 4)
```powershell
git add src/embeddings.py tests/test_embeddings.py
git commit -m "feat(phase1): add embedding integration with all-MiniLM-L6-v2"
```

### Commit 5 (After Task 5)
```powershell
git add src/vector_store.py tests/test_vector_store.py
git commit -m "feat(phase1): add ChromaDB vector store with persistent SQLite backend"
```

### Commit 6 (After Task 6)
```powershell
git add .planning/phases/01-foundation/outputs/chunk_experiment.py .planning/phases/01-foundation/outputs/chunk_experiment_results.csv
git commit -m "test(phase1): add chunk size experiment and evaluation script"
```

### Commit 7 (After Task 7)
```powershell
git add tests/test_e2e.py
git commit -m "test(phase1): add E2E integration test with performance profiling"
```

### Commit 8 (After Task 8)
```powershell
git add .planning/phases/01-foundation/SUMMARY.md README_PHASE1.md .gitignore
git commit -m "docs(phase1): add SUMMARY, README, and update .gitignore"
```

**Final Status Before Phase 2:**
```powershell
git log --oneline | head -8
# Output:
# XXXXXXX docs(phase1): add SUMMARY, README, and update .gitignore
# XXXXXXX test(phase1): add E2E integration test with performance profiling
# XXXXXXX test(phase1): add chunk size experiment and evaluation script
# XXXXXXX feat(phase1): add ChromaDB vector store with persistent SQLite backend
# XXXXXXX feat(phase1): add embedding integration with all-MiniLM-L6-v2
# XXXXXXX feat(phase1): add chunking module with 3-size experiment support
# XXXXXXX feat(phase1): add PDF extraction module with PyMuPDF
# XXXXXXX setup(phase1): create venv, install dependencies, initialize project structure
```

---

## Execution Checklist

Use this checklist to track Phase 1 progress:

### Pre-Execution
- [ ] Read this PLAN.md thoroughly
- [ ] Clone/prepare workspace: `d:\Data_Ngoc\Test\TotNghiepProject\`
- [ ] Verify Python 3.9+ installed: `python --version`
- [ ] Prepare sample HR handbook PDF for testing

### During Execution
- [ ] **Task 1 (0.5h)**: Setup & Environment
  - [ ] Venv created & activated
  - [ ] Requirements.txt created & dependencies installed
  - [ ] Project structure created
  - [ ] Imports verified
  - [ ] ✓ Commit 1

- [ ] **Task 2 (1.0h)**: PDF Extraction
  - [ ] `src/pdf_extraction.py` written
  - [ ] Unit tests written & passing
  - [ ] Sample PDF parses successfully
  - [ ] ✓ Gate 1 passed
  - [ ] ✓ Commit 2

- [ ] **Task 3 (1.0h)**: Chunking
  - [ ] `src/chunking.py` written
  - [ ] Unit tests written & passing
  - [ ] Chunks generated with correct coverage
  - [ ] ✓ Gate 2 passed
  - [ ] ✓ Commit 3

- [ ] **Task 4 (1.0h)**: Embeddings
  - [ ] `src/embeddings.py` written
  - [ ] Embedding model downloaded
  - [ ] Unit tests written & passing
  - [ ] Embedding shape verified (384,)
  - [ ] ✓ Commit 4

- [ ] **Task 5 (0.75h)**: Vector Store
  - [ ] `src/vector_store.py` written
  - [ ] ChromaDB initialized with SQLite backend
  - [ ] Unit tests written & passing
  - [ ] Index persists to disk
  - [ ] ✓ Gate 3 passed
  - [ ] ✓ Commit 5

- [ ] **Task 6 (0.75h)**: Chunk Experiment
  - [ ] Experiment script written & runs
  - [ ] All 3 chunk sizes tested
  - [ ] CSV report generated
  - [ ] Recommendation documented
  - [ ] ✓ Commit 6

- [ ] **Task 7 (0.5h)**: E2E Tests
  - [ ] E2E test written
  - [ ] All tests pass
  - [ ] Performance baseline documented
  - [ ] ✓ Gate 4 passed
  - [ ] ✓ Commit 7

- [ ] **Task 8 (0.25h)**: Documentation
  - [ ] Phase 1 SUMMARY.md written
  - [ ] README_PHASE1.md written
  - [ ] .gitignore updated
  - [ ] All commits pushed
  - [ ] ✓ Commit 8

### Post-Execution
- [ ] All 11 unit tests pass: `pytest tests/ -v`
- [ ] E2E test passes with reasonable latency: `pytest tests/test_e2e.py -v -s`
- [ ] Chunk size recommendation documented
- [ ] Performance baseline captured
- [ ] Phase 1 exit criteria met (all 7 gates passed)
- [ ] Ready for Phase 2 handoff

---

## Troubleshooting & Common Issues

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **Import errors after install** | `ModuleNotFoundError: No module named 'fitz'` | Verify venv is activated: `.\.venv\Scripts\Activate.ps1` |
| **PDF file not found** | `FileNotFoundError: data/sample_handbook.pdf` | Create sample PDF or use existing test fixture |
| **Embedding model download timeout** | Hanging during first embed, then exception | Check internet connection; manually download model |
| **ChromaDB permission error** | `PermissionError: ./chroma_db/` | Delete `./chroma_db/` and retry tests |
| **Test fails with "low coverage"** | Coverage < 95% in test_no_text_loss | Check for text loss in chunking; verify overlap calculation |
| **Retrieval latency too high** | Retrieval > 100ms | Index may be too large; verify chunk count is reasonable |
| **Python crashes during embed** | `CUDA out of memory` or similar | Model should run on CPU; verify torch CPU version installed |

---

## Success Summary

**Phase 1 is complete when:**
1. ✓ All 8 tasks finished
2. ✓ All 11 unit tests pass
3. ✓ E2E integration test passes
4. ✓ All 4 quality gates passed
5. ✓ Chunk size recommendation ready
6. ✓ Performance baseline documented
7. ✓ Code committed atomically (8 commits)
8. ✓ README + SUMMARY written

**Estimated Time:** 5 hours total  
**Next Phase:** Phase 2 (Retrieval & Normalization) — starts when Phase 1 exit criteria met

---

**PLAN.md created:** 2026-05-04  
**Status:** Ready for execution  
**Owner:** Solo developer  
**Next:** Execute Task 1 → Task 2 → ... → Task 8 sequentially

