"""
Crawl and download Vietnamese HR policy PDFs for training data.

Usage:
    python scripts/crawl_hr_pdfs.py                  # download up to 10 PDFs
    python scripts/crawl_hr_pdfs.py --limit 20       # download up to 20
    python scripts/crawl_hr_pdfs.py --dry-run        # show URLs only, no download
"""
import argparse
import hashlib
import io
import sys
import time
from pathlib import Path

import httpx

# Fix Windows console encoding for Vietnamese text
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

OUTPUT_DIR = Path("data/raw/pdf")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Vietnamese HR search queries — no filetype: operator (DDG ignores it)
# Filter for .pdf links in results instead
QUERIES = [
    '"sổ tay nhân viên" site:tailieu.vn OR site:123doc.net OR site:vndoc.com',
    '"nội quy công ty" nhân sự site:tailieu.vn OR site:123doc.net',
    '"quy chế nhân sự" lao động site:tailieu.vn OR site:vndoc.com',
    '"chính sách phúc lợi" nhân viên site:tailieu.vn',
    '"sổ tay nhân viên" filetype:pdf',
    '"nội quy lao động" filetype:pdf',
    '"handbook" "nhân viên" "công ty" filetype:pdf',
    'sổ tay nhân viên công ty Việt Nam pdf download',
    'nội quy công ty nhân sự pdf tải về',
    '"hướng dẫn nhân viên" "quy trình" pdf',
]

# Direct PDF URLs from known Vietnamese document hosts — seed list
SEED_URLS = [
    "https://thuvienphapluat.vn/van-ban/Lao-dong-Tien-luong/Nghi-dinh-145-2020-ND-CP-huong-dan-Bo-luat-Lao-dong-ve-dieu-kien-lao-dong-456454.aspx",
]

MIN_PDF_BYTES = 50_000   # skip tiny/corrupt PDFs (<50 KB)
MAX_PDF_BYTES = 20_000_000  # skip huge files (>20 MB)
REQUEST_TIMEOUT = 30


def url_to_filename(url: str) -> str:
    slug = url.split("/")[-1].split("?")[0]
    if not slug.lower().endswith(".pdf"):
        slug = hashlib.md5(url.encode()).hexdigest()[:12] + ".pdf"
    return slug


def already_downloaded(url: str) -> bool:
    fname = url_to_filename(url)
    return (OUTPUT_DIR / fname).exists()


def download_pdf(url: str, dry_run: bool = False) -> bool:
    fname = url_to_filename(url)
    dest = OUTPUT_DIR / fname

    if dest.exists():
        print(f"  skip (exists): {fname}")
        return False

    if dry_run:
        print(f"  [dry-run] would download: {url}")
        return True

    try:
        print(f"  downloading: {fname} ...", end=" ", flush=True)
        with httpx.Client(follow_redirects=True, timeout=REQUEST_TIMEOUT, verify=False) as client:
            resp = client.get(url, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")
        if "pdf" not in content_type and not url.lower().endswith(".pdf"):
            print(f"skip (not PDF: {content_type})")
            return False

        size = len(resp.content)
        if size < MIN_PDF_BYTES:
            print(f"skip (too small: {size//1024}KB)")
            return False
        if size > MAX_PDF_BYTES:
            print(f"skip (too large: {size//1024//1024}MB)")
            return False

        dest.write_bytes(resp.content)
        print(f"OK ({size//1024}KB)")
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        return False


def extract_pdf_links_from_page(url: str) -> list[str]:
    """Fetch a page and pull out all .pdf hrefs."""
    try:
        resp = httpx.get(url, timeout=15, follow_redirects=True,
                         verify=False,
                         headers={"User-Agent": "Mozilla/5.0"})
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, "html.parser")
        base = "/".join(url.split("/")[:3])
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.lower().endswith(".pdf"):
                full = href if href.startswith("http") else base + "/" + href.lstrip("/")
                links.append(full)
        return links
    except Exception as e:
        print(f"  page-scrape error {url}: {e}")
        return []


def search_pdf_urls(queries: list[str], max_results_per_query: int = 15) -> list[str]:
    seen = set()
    urls = []

    with DDGS() as ddgs:
        for query in queries:
            print(f"\nSearching: {query[:60]}")
            try:
                results = list(ddgs.text(query, max_results=max_results_per_query))
                for r in results:
                    url = r.get("href") or r.get("url", "")
                    if not url:
                        continue
                    # Direct PDF link
                    if url.lower().endswith(".pdf") and url not in seen:
                        seen.add(url)
                        urls.append(url)
                        print(f"  direct pdf: {url}")
                    # Page that may contain PDF links — scrape it
                    elif any(host in url for host in
                             ("tailieu.vn", "123doc.net", "vndoc.com", "slideshare.net",
                              "doc.edu.vn", "luatvietnam.vn")):
                        for pdf_url in extract_pdf_links_from_page(url):
                            if pdf_url not in seen:
                                seen.add(pdf_url)
                                urls.append(pdf_url)
                                print(f"  scraped pdf: {pdf_url}")
                time.sleep(1.5)
            except Exception as e:
                print(f"  search error: {e}")

    return urls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10, help="Max PDFs to download")
    parser.add_argument("--dry-run", action="store_true", help="Show URLs only, no download")
    args = parser.parse_args()

    print(f"Target: {args.limit} PDFs -> {OUTPUT_DIR}")
    print(f"Already have: {len(list(OUTPUT_DIR.glob('*.pdf')))} PDFs\n")

    urls = search_pdf_urls(QUERIES, max_results_per_query=8)
    print(f"\nFound {len(urls)} candidate PDF URLs")

    downloaded = 0
    for url in urls:
        if downloaded >= args.limit:
            break
        ok = download_pdf(url, dry_run=args.dry_run)
        if ok and not args.dry_run:
            downloaded += 1
        time.sleep(0.5)

    existing = list(OUTPUT_DIR.glob("*.pdf"))
    print(f"\nDone. PDFs in {OUTPUT_DIR}: {len(existing)}")
    for p in existing:
        print(f"  {p.name}  ({p.stat().st_size//1024}KB)")


if __name__ == "__main__":
    main()
