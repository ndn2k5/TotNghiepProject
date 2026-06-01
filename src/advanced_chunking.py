"""
Advanced Chunking Strategies
- Semantic Chunking (chunk by meaning, not fixed size)
- Small-to-Big Retrieval (retrieve small chunks, expand to big chunks)
"""

import logging
from typing import List, Dict, Optional
import re

logger = logging.getLogger(__name__)


class SemanticChunker:
    """
    Chunks text by semantic boundaries (sentences, paragraphs)
    instead of fixed character count
    """
    
    def __init__(self, chunk_size: int = 300, overlap: int = 50):
        """
        Initialize semantic chunker
        
        Args:
            chunk_size: Target size for chunks (in characters)
            overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Vietnamese sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Chunk text semantically (by sentences/paragraphs)
        
        Args:
            text: Text to chunk
            metadata: Metadata to attach to chunks
            
        Returns:
            List of chunks with metadata
        """
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # Add sentence if it fits in current chunk
            if current_size + sentence_size <= self.chunk_size:
                current_chunk.append(sentence)
                current_size += sentence_size + 1  # +1 for space
            else:
                # Save current chunk and start new one
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append({
                        'text': chunk_text,
                        'metadata': metadata or {},
                        'size': len(chunk_text),
                        'type': 'semantic'
                    })
                
                # Start new chunk (with overlap)
                overlap_sentences = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences + [sentence]
                current_size = sum(len(s) for s in current_chunk) + len(current_chunk)
        
        # Add last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'metadata': metadata or {},
                'size': len(chunk_text),
                'type': 'semantic'
            })
        
        logger.info(f"Semantic chunking produced {len(chunks)} chunks")
        return chunks
    
    def _get_overlap_sentences(self, sentences: List[str], ratio: float = 0.3) -> List[str]:
        """Get last sentences for overlap"""
        count = max(1, int(len(sentences) * ratio))
        return sentences[-count:]


class SmallToBigRetrieval:
    """
    Small-to-Big retrieval strategy:
    - Index small chunks for search (precise)
    - But return parent chunks (more context)
    """
    
    def __init__(self, small_chunk_size: int = 150, big_chunk_size: int = 500):
        """
        Initialize Small-to-Big retrieval
        
        Args:
            small_chunk_size: Size for search index
            big_chunk_size: Size for context retrieval
        """
        self.small_chunk_size = small_chunk_size
        self.big_chunk_size = big_chunk_size
    
    def _chunk_small(self, text: str) -> List[str]:
        """Create small chunks for indexing"""
        chunker = SemanticChunker(chunk_size=self.small_chunk_size, overlap=30)
        chunks = chunker.chunk(text)
        return [c['text'] for c in chunks]
    
    def _chunk_big(self, text: str) -> List[str]:
        """Create big chunks for context"""
        chunker = SemanticChunker(chunk_size=self.big_chunk_size, overlap=50)
        chunks = chunker.chunk(text)
        return [c['text'] for c in chunks]
    
    def process(self, text: str, metadata: Dict = None) -> Dict:
        """
        Process text into small and big chunks
        
        Returns:
            {
                'small_chunks': [for search],
                'big_chunks': [for context],
                'mapping': {small_idx: big_idx}
            }
        """
        small_chunks = self._chunk_small(text)
        big_chunks = self._chunk_big(text)
        
        # Create mapping from small to big chunks
        mapping = {}
        text_lower = text.lower()
        
        for i, small_chunk in enumerate(small_chunks):
            small_lower = small_chunk.lower()
            
            # Find which big chunk contains this small chunk
            for j, big_chunk in enumerate(big_chunks):
                big_lower = big_chunk.lower()
                if small_lower in big_lower:
                    mapping[i] = j
                    break
        
        return {
            'small_chunks': [
                {
                    'text': chunk,
                    'metadata': metadata or {},
                    'type': 'small',
                    'size': len(chunk)
                }
                for chunk in small_chunks
            ],
            'big_chunks': [
                {
                    'text': chunk,
                    'metadata': metadata or {},
                    'type': 'big',
                    'size': len(chunk)
                }
                for chunk in big_chunks
            ],
            'mapping': mapping
        }
    
    def retrieve_with_context(
        self,
        retrieved_small_indices: List[int],
        processed_chunks: Dict
    ) -> List[Dict]:
        """
        Given retrieved small chunk indices, return big chunks with context
        
        Args:
            retrieved_small_indices: Indices of retrieved small chunks
            processed_chunks: Output from process()
            
        Returns:
            List of big chunks with context
        """
        mapping = processed_chunks['mapping']
        big_chunks = processed_chunks['big_chunks']
        
        # Get unique big chunk indices
        big_indices = set()
        for small_idx in retrieved_small_indices:
            if small_idx in mapping:
                big_indices.add(mapping[small_idx])
        
        # Return big chunks
        return [big_chunks[idx] for idx in sorted(big_indices)]


class AdvancedChunkingStrategy:
    """
    Combined strategy: Semantic + Small-to-Big
    """
    
    def __init__(self):
        """Initialize advanced chunking"""
        self.semantic_chunker = SemanticChunker()
        self.small_to_big = SmallToBigRetrieval()
    
    def chunk_document(self, text: str, metadata: Dict = None) -> Dict:
        """
        Chunk document using advanced strategy
        """
        processed = self.small_to_big.process(text, metadata)
        
        logger.info(
            f"Advanced chunking: {len(processed['small_chunks'])} small chunks, "
            f"{len(processed['big_chunks'])} big chunks"
        )
        
        return processed


if __name__ == "__main__":
    print("Advanced Chunking Module Ready")
    print("- Semantic Chunking ✓")
    print("- Small-to-Big Retrieval ✓")
    print("- Combined Strategy ✓")
