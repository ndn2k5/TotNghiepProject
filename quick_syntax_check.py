"""
Fast syntax and structure validation without heavy imports.

This test:
1. Checks Python syntax of all files
2. Validates file structure
3. Checks for required classes/functions
4. Verifies backward compatibility
"""

import ast
import sys
from pathlib import Path


def validate_python_syntax(file_path):
    """Validate Python syntax by parsing with ast"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)


def check_file_has_class(file_path, class_name):
    """Check if file defines a specific class"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                return True
        return False
    except Exception as e:
        return False


def check_file_has_function(file_path, func_name):
    """Check if file defines a specific function"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                return True
        return False
    except Exception as e:
        return False


def main():
    print("\n" + "=" * 70)
    print("FAST SYNTAX & STRUCTURE VALIDATION")
    print("=" * 70 + "\n")
    
    all_passed = True
    
    # Test 1: File existence and syntax
    print("TEST 1: Python syntax validation")
    print("-" * 70)
    
    files_to_check = [
        "src/retriever_agent.py",
        "src/rag_pipeline.py",
        "src/embeddings.py",
        "src/gguf_models.py",
        "src/pdf_extraction.py",
        "src/chunking.py",
        "src/question_normalizer.py",
        "src/responder.py",
    ]
    
    for file_path in files_to_check:
        p = Path(file_path)
        if not p.exists():
            print(f"❌ File not found: {file_path}")
            all_passed = False
            continue
        
        valid, error = validate_python_syntax(file_path)
        if valid:
            print(f"✓ {file_path}")
        else:
            print(f"❌ {file_path}: {error}")
            all_passed = False
    
    # Test 2: Key classes exist
    print("\nTEST 2: Required classes")
    print("-" * 70)
    
    class_checks = [
        ("src/retriever_agent.py", "RetrieverAgent"),
        ("src/rag_pipeline.py", "RAGPipeline"),
        ("src/embeddings.py", "LocalEmbedder"),
        ("src/embeddings.py", "VectorStoreManager"),
        ("src/gguf_models.py", "LocalGGUFModel"),
        ("src/pdf_extraction.py", "PDFExtractor"),
        ("src/responder.py", "ResponseGenerator"),
    ]
    
    for file_path, class_name in class_checks:
        if check_file_has_class(file_path, class_name):
            print(f"✓ {class_name} in {file_path}")
        else:
            print(f"❌ {class_name} NOT FOUND in {file_path}")
            all_passed = False
    
    # Test 3: Key methods exist in RetrieverAgent
    print("\nTEST 3: RetrieverAgent methods")
    print("-" * 70)
    
    with open("src/retriever_agent.py", 'r', encoding='utf-8') as f:
        tree = ast.parse(f.read())
    
    retriever_agent_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "RetrieverAgent":
            retriever_agent_class = node
            break
    
    if retriever_agent_class:
        methods = [node.name for node in retriever_agent_class.body if isinstance(node, ast.FunctionDef)]
        required_methods = ["__init__", "is_enabled", "process"]
        
        for method in required_methods:
            if method in methods:
                print(f"✓ Method {method} exists")
            else:
                print(f"❌ Method {method} NOT FOUND")
                all_passed = False
    else:
        print("❌ RetrieverAgent class not found")
        all_passed = False
    
    # Test 4: RAGPipeline has retriever_agent parameter
    print("\nTEST 4: RAGPipeline integration")
    print("-" * 70)
    
    with open("src/rag_pipeline.py", 'r', encoding='utf-8') as f:
        content = f.read()
    
    if "retriever_agent_model_path" in content:
        print("✓ RAGPipeline has retriever_agent_model_path parameter")
    else:
        print("❌ RAGPipeline missing retriever_agent_model_path parameter")
        all_passed = False
    
    if "self.retriever_agent" in content:
        print("✓ RAGPipeline initializes self.retriever_agent")
    else:
        print("❌ RAGPipeline not initializing self.retriever_agent")
        all_passed = False
    
    if "from src.retriever_agent import RetrieverAgent" in content:
        print("✓ RAGPipeline imports RetrieverAgent")
    else:
        print("❌ RAGPipeline NOT importing RetrieverAgent")
        all_passed = False
    
    if "self.retriever_agent.process" in content:
        print("✓ RAGPipeline calls self.retriever_agent.process()")
    else:
        print("❌ RAGPipeline NOT calling self.retriever_agent.process()")
        all_passed = False
    
    if "retriever_agent_used" in content:
        print("✓ RAGPipeline tracks retriever_agent_used")
    else:
        print("❌ RAGPipeline NOT tracking retriever_agent_used")
        all_passed = False
    
    # Test 5: Backward compatibility
    print("\nTEST 5: Backward compatibility")
    print("-" * 70)
    
    if "retriever_agent_model_path: Optional[str] = None" in content:
        print("✓ retriever_agent_model_path has default value (backward compatible)")
    else:
        print("⚠️  Check retriever_agent_model_path default value")
    
    if "retriever_agent_model_path is not None" in content:
        print("✓ Agent enabled only when model path provided (optional)")
    else:
        print("⚠️  Check agent enabling logic")
    
    # Test 6: Prompts defined
    print("\nTEST 6: Prompt templates")
    print("-" * 70)
    
    retriever_file = Path("src/retriever_agent.py")
    rag_file = Path("src/rag_pipeline.py")
    
    retriever_content = retriever_file.read_text(encoding='utf-8')
    rag_content = rag_file.read_text(encoding='utf-8')
    
    prompts_to_check = [
        ("PROMPT_RETRIEVER_VI", retriever_content),
        ("PROMPT_RETRIEVER_EN", retriever_content),
        ("PROMPT_TEMPLATE_VI", rag_content),
        ("PROMPT_TEMPLATE_EN", rag_content),
    ]
    
    for prompt_name, file_content in prompts_to_check:
        if f"{prompt_name} =" in file_content or f"{prompt_name}=" in file_content:
            print(f"✓ {prompt_name} defined")
        else:
            print(f"❌ {prompt_name} NOT FOUND")
            all_passed = False
    
    # Test 7: Logging
    print("\nTEST 7: Logging statements")
    print("-" * 70)
    
    if "import logging" in retriever_content:
        print("✓ RetrieverAgent imports logging")
    else:
        print("⚠️  RetrieverAgent missing logging import")
    
    if "logger.debug" in retriever_content or "logger.info" in retriever_content:
        print("✓ RetrieverAgent has logging statements")
    else:
        print("⚠️  RetrieverAgent missing logging")
    
    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL CHECKS PASSED! Code structure is valid.")
        print("=" * 70 + "\n")
        return 0
    else:
        print("❌ SOME CHECKS FAILED! Review above for details.")
        print("=" * 70 + "\n")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
