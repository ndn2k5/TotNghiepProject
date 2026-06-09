import pytest
from unittest.mock import patch, MagicMock
from src.rag_pipeline import RAGPipeline

@pytest.fixture
def mock_model_path(tmp_path):
    model_file = tmp_path / "mock_model.gguf"
    model_file.write_bytes(b"mock data")
    return str(model_file)

class TestRAGPipelineUnit:
    @patch("src.rag_pipeline.LocalEmbedder")
    @patch("src.rag_pipeline.VectorStoreManager")
    @patch("src.rag_pipeline.LocalGGUFModel")
    def test_pipeline_init_and_stats(self, mock_llm_cls, mock_vdb_cls, mock_embed_cls, mock_model_path):
        # Setup mocks
        mock_vdb = mock_vdb_cls.return_value
        mock_vdb.count.return_value = 10
        mock_embed = mock_embed_cls.return_value
        mock_embed.dimension = 384
        
        # Setup LLM mock specifically for model_name
        mock_llm = mock_llm_cls.return_value
        mock_llm.model_name = "mock_model.gguf"
        
        pipeline = RAGPipeline(mock_model_path)
        
        stats = pipeline.get_stats()
        assert stats["documents_in_store"] == 10
        assert stats["embedding_dimension"] == 384
        # Check that we at least get a string or mock object that we can check
        assert "mock" in str(stats["model"]).lower()

    @patch("src.rag_pipeline.LocalEmbedder")
    @patch("src.rag_pipeline.VectorStoreManager")
    @patch("src.rag_pipeline.LocalGGUFModel")
    def test_language_switching(self, mock_llm_cls, mock_vdb_cls, mock_embed_cls, mock_model_path):
        pipeline_vi = RAGPipeline(mock_model_path, language="vi")
        assert "tiếng Việt" in pipeline_vi.prompt_template
        
        pipeline_en = RAGPipeline(mock_model_path, language="en")
        assert "accurately and concisely" in pipeline_en.prompt_template
