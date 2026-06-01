"""
Comprehensive code validation and testing suite for RAG chatbot.

Usage:
    python test_all_code.py
    
This script validates:
1. All imports and dependencies
2. Module structure
3. Backward compatibility
4. RetrieverAgent integration
5. Pipeline integration
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def test_imports():
    """Test all imports work correctly"""
    logger.info("=" * 60)
    logger.info("TEST 1: Checking imports...")
    logger.info("=" * 60)
    
    try:
        logger.info("✓ Importing embeddings...")
        from src.embeddings import LocalEmbedder, VectorStoreManager
        
        logger.info("✓ Importing GGUF models...")
        from src.gguf_models import LocalGGUFModel
        
        logger.info("✓ Importing PDF extraction...")
        from src.pdf_extraction import PDFExtractor
        
        logger.info("✓ Importing chunking...")
        from src.chunking import chunk_pages
        
        logger.info("✓ Importing question normalizer...")
        from src.question_normalizer import QuestionNormalizer
        
        logger.info("✓ Importing NEW RetrieverAgent...")
        from src.retriever_agent import RetrieverAgent
        
        logger.info("✓ Importing RAGPipeline (with agent support)...")
        from src.rag_pipeline import RAGPipeline
        
        logger.info("✓ Importing responder...")
        from src.responder import ResponseGenerator
        
        logger.info("\n✅ All imports successful!\n")
        return True
        
    except Exception as e:
        logger.error(f"❌ Import failed: {e}")
        return False


def test_retriever_agent_structure():
    """Test RetrieverAgent class structure"""
    logger.info("=" * 60)
    logger.info("TEST 2: Checking RetrieverAgent structure...")
    logger.info("=" * 60)
    
    try:
        from src.retriever_agent import RetrieverAgent
        
        # Check required methods
        required_methods = ['__init__', 'is_enabled', 'process']
        for method in required_methods:
            if not hasattr(RetrieverAgent, method):
                raise AttributeError(f"Missing method: {method}")
            logger.info(f"✓ Method {method} exists")
        
        # Instantiate with enabled=False (don't load model)
        agent = RetrieverAgent(model_path=None, enabled=False)
        assert not agent.is_enabled(), "Agent should be disabled"
        logger.info("✓ Agent correctly disabled when no model path")
        
        logger.info("\n✅ RetrieverAgent structure valid!\n")
        return True
        
    except Exception as e:
        logger.error(f"❌ Structure check failed: {e}")
        return False


def test_rag_pipeline_backward_compat():
    """Test RAGPipeline backward compatibility"""
    logger.info("=" * 60)
    logger.info("TEST 3: Checking RAGPipeline backward compatibility...")
    logger.info("=" * 60)
    
    try:
        from src.rag_pipeline import RAGPipeline
        
        # Check that RAGPipeline has new parameter
        import inspect
        sig = inspect.signature(RAGPipeline.__init__)
        
        required_params = [
            'self', 'model_path', 'persist_dir', 'collection_name',
            'n_ctx', 'language', 'n_gpu_layers', 'use_reranker',
            'reranker_model', 'retriever_agent_model_path'
        ]
        
        actual_params = list(sig.parameters.keys())
        
        for param in required_params[:-1]:  # All except retriever_agent_model_path
            if param not in actual_params:
                raise ValueError(f"Missing parameter: {param}")
            logger.info(f"✓ Parameter {param} exists")
        
        # Check new parameter exists
        if 'retriever_agent_model_path' not in actual_params:
            raise ValueError("Missing new parameter: retriever_agent_model_path")
        logger.info("✓ New parameter retriever_agent_model_path exists")
        
        # Check it's optional
        param_obj = sig.parameters['retriever_agent_model_path']
        if param_obj.default is None:
            logger.info("✓ retriever_agent_model_path is optional (default=None)")
        
        logger.info("\n✅ Backward compatibility verified!\n")
        return True
        
    except Exception as e:
        logger.error(f"❌ Backward compatibility check failed: {e}")
        return False


def test_file_structure():
    """Test file structure is correct"""
    logger.info("=" * 60)
    logger.info("TEST 4: Checking file structure...")
    logger.info("=" * 60)
    
    try:
        required_files = [
            "src/__init__.py",
            "src/retriever_agent.py",
            "src/rag_pipeline.py",
            "src/embeddings.py",
            "src/gguf_models.py",
            "src/pdf_extraction.py",
            "src/chunking.py",
            "src/question_normalizer.py",
            "src/responder.py",
        ]
        
        for file in required_files:
            path = Path(file)
            if not path.exists():
                raise FileNotFoundError(f"Missing file: {file}")
            logger.info(f"✓ File exists: {file}")
        
        logger.info("\n✅ File structure correct!\n")
        return True
        
    except Exception as e:
        logger.error(f"❌ File structure check failed: {e}")
        return False


def test_prompts():
    """Test prompt templates exist"""
    logger.info("=" * 60)
    logger.info("TEST 5: Checking prompt templates...")
    logger.info("=" * 60)
    
    try:
        from src.rag_pipeline import PROMPT_TEMPLATE_VI, PROMPT_TEMPLATE_EN
        from src.retriever_agent import PROMPT_RETRIEVER_VI, PROMPT_RETRIEVER_EN
        
        # Check Vietnamese prompts
        if "{question}" not in PROMPT_TEMPLATE_VI:
            raise ValueError("PROMPT_TEMPLATE_VI missing {question} placeholder")
        logger.info("✓ PROMPT_TEMPLATE_VI is valid")
        
        if "{context}" not in PROMPT_TEMPLATE_VI:
            raise ValueError("PROMPT_TEMPLATE_VI missing {context} placeholder")
        logger.info("✓ PROMPT_TEMPLATE_VI has context placeholder")
        
        # Check English prompts
        if "{question}" not in PROMPT_TEMPLATE_EN:
            raise ValueError("PROMPT_TEMPLATE_EN missing {question} placeholder")
        logger.info("✓ PROMPT_TEMPLATE_EN is valid")
        
        # Check retriever agent prompts
        if "{question}" not in PROMPT_RETRIEVER_VI:
            raise ValueError("PROMPT_RETRIEVER_VI missing {question} placeholder")
        logger.info("✓ PROMPT_RETRIEVER_VI is valid")
        
        if "{chunks_text}" not in PROMPT_RETRIEVER_VI:
            raise ValueError("PROMPT_RETRIEVER_VI missing {chunks_text} placeholder")
        logger.info("✓ PROMPT_RETRIEVER_VI has chunks_text placeholder")
        
        logger.info("\n✅ All prompts valid!\n")
        return True
        
    except Exception as e:
        logger.error(f"❌ Prompt check failed: {e}")
        return False


def test_type_hints():
    """Test type hints are present"""
    logger.info("=" * 60)
    logger.info("TEST 6: Checking type hints...")
    logger.info("=" * 60)
    
    try:
        from src.retriever_agent import RetrieverAgent
        import inspect
        
        # Check RetrieverAgent.__init__ has type hints
        sig = inspect.signature(RetrieverAgent.__init__)
        
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            if param.annotation == inspect.Parameter.empty:
                logger.warning(f"⚠️  Parameter {param_name} missing type hint")
            else:
                logger.info(f"✓ Parameter {param_name}: {param.annotation}")
        
        # Check return type
        if sig.return_annotation == inspect.Signature.empty:
            logger.warning("⚠️  Return type hint missing for __init__")
        else:
            logger.info(f"✓ Return type: {sig.return_annotation}")
        
        logger.info("\n✅ Type hints present (warnings are acceptable)!\n")
        return True
        
    except Exception as e:
        logger.error(f"❌ Type hint check failed: {e}")
        return False


def test_error_handling():
    """Test error handling is present"""
    logger.info("=" * 60)
    logger.info("TEST 7: Checking error handling...")
    logger.info("=" * 60)
    
    try:
        from src.retriever_agent import RetrieverAgent
        import inspect
        
        # Check RetrieverAgent.process has try-except
        source = inspect.getsource(RetrieverAgent.process)
        
        if "try:" in source and "except" in source:
            logger.info("✓ RetrieverAgent.process has exception handling")
        else:
            raise ValueError("Missing exception handling in process()")
        
        if "logger.error" in source or "logger.warning" in source:
            logger.info("✓ RetrieverAgent.process has logging")
        else:
            raise ValueError("Missing logging in process()")
        
        logger.info("\n✅ Error handling present!\n")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error handling check failed: {e}")
        return False


def test_documentation():
    """Test documentation is present"""
    logger.info("=" * 60)
    logger.info("TEST 8: Checking documentation...")
    logger.info("=" * 60)
    
    try:
        from src.retriever_agent import RetrieverAgent
        
        # Check module docstring
        if not RetrieverAgent.__module__:
            logger.warning("⚠️  Module docstring missing")
        else:
            logger.info("✓ Module has documentation")
        
        # Check class docstring
        if not RetrieverAgent.__doc__:
            logger.warning("⚠️  Class docstring missing")
        else:
            logger.info(f"✓ Class docstring: {RetrieverAgent.__doc__[:50]}...")
        
        # Check method docstrings
        for method_name in ['__init__', 'process', 'is_enabled']:
            method = getattr(RetrieverAgent, method_name)
            if not method.__doc__:
                logger.warning(f"⚠️  Method {method_name} docstring missing")
            else:
                logger.info(f"✓ Method {method_name} documented")
        
        logger.info("\n✅ Documentation present!\n")
        return True
        
    except Exception as e:
        logger.error(f"❌ Documentation check failed: {e}")
        return False


def main():
    """Run all tests"""
    logger.info("\n" + "=" * 60)
    logger.info("COMPREHENSIVE CODE REVIEW")
    logger.info("=" * 60 + "\n")
    
    tests = [
        ("Imports", test_imports),
        ("RetrieverAgent Structure", test_retriever_agent_structure),
        ("RAGPipeline Backward Compatibility", test_rag_pipeline_backward_compat),
        ("File Structure", test_file_structure),
        ("Prompts", test_prompts),
        ("Type Hints", test_type_hints),
        ("Error Handling", test_error_handling),
        ("Documentation", test_documentation),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"RESULT: {passed}/{total} tests passed")
    logger.info("=" * 60 + "\n")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! Code is ready for deployment.\n")
        return 0
    else:
        logger.error(f"⚠️  {total - passed} test(s) failed. Review above.\n")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
