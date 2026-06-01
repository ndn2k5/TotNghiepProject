"""
Clone 3 GitHub handbooks, extract Markdown, chunk, save to data/raw_chunks.jsonl
"""
import subprocess
import os
import json
import re
from pathlib import Path
from langchain_text_splitters import MarkdownTextSplitter

HANDBOOKS = [
    {"name": "hshadab", "url": "https://github.com/hshadab/handbook"},
    {"name": "cuesoftinc", "url": "https://github.com/cuesoftinc/handbook"},
    {"name": "ultralytics", "url": "https://github.com/ultralytics/handbook"},
]

RAW_DIR = Path("data/raw")
OUTPUT_FILE = Path("data/raw_chunks.jsonl")
CHUNK_SIZE = 500  # chars
CHUNK_OVERLAP = 50  # chars

def clone_or_pull(repo_url: str, target_dir: Path):
    if target_dir.exists():
        print(f"  Already exists: {target_dir}, skipping clone")
        return
    subprocess.run(["git", "clone", "--depth=1", repo_url, str(target_dir)], check=True)

def extract_markdown_files(repo_dir: Path) -> list[tuple[str, str]]:
    """Return list of (filename, content) for all .md files."""
    results = []
    for md_file in repo_dir.rglob("*.md"):
        try:
            content = md_file.read_text(encoding="utf-8", errors="ignore")
            results.append((str(md_file.relative_to(repo_dir)), content))
        except Exception as e:
            print(f"  Warning: could not read {md_file}: {e}")
    return results

def chunk_text_with_langchain(text: str, source: str, filename: str) -> list[dict]:
    """Split text into overlapping chunks using MarkdownTextSplitter."""
    splitter = MarkdownTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = splitter.create_documents([text])
    
    chunks = []
    for doc in docs:
        chunk = doc.page_content.strip()
        # Clean up any remaining artifacts if needed, but MarkdownTextSplitter handles markdown.
        # Removing raw HTML tags, URLs that are too long, or code blocks may still be needed,
        # but let's keep it simple and just strip.
        
        # Strip long code blocks as they are not good QA material
        chunk = re.sub(r"```[\s\S]*?```", "", chunk).strip()
        
        if len(chunk) > 100:  # skip tiny chunks
            chunks.append({
                "text": chunk,
                "source": source,
                "filename": filename,
            })
    return chunks

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    all_chunks = []

    for handbook in HANDBOOKS:
        name = handbook["name"]
        url = handbook["url"]
        repo_dir = RAW_DIR / name

        print(f"\n[{name}] Cloning from {url}...")
        clone_or_pull(url, repo_dir)

        md_files = extract_markdown_files(repo_dir)
        print(f"[{name}] Found {len(md_files)} markdown files")

        for filename, content in md_files:
            if len(content.strip()) < 200:
                continue  # skip too-short files
            file_chunks = chunk_text_with_langchain(content, source=name, filename=filename)
            all_chunks.extend(file_chunks)

        print(f"[{name}] Extracted chunks so far: {len(all_chunks)}")

    print(f"\nTotal chunks: {len(all_chunks)}")
    print(f"Total words (approx): {sum(len(c['text'].split()) for c in all_chunks)}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\n✓ Saved {len(all_chunks)} chunks to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
