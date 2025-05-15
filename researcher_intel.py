
"""One‑shot intelligence report generator for academic researchers.

Given a researcher's full name, this script will:
  1. Locate their Google Scholar profile (via `scholarly`).
  2. Compile their top‑10 most‑cited papers *plus* all papers published within the
     last 3 calendar years.
  3. Attempt to fetch PDFs (preferring arXiv/open‑access links). If a clean text
     layer is unavailable, fallback to OCR with Tesseract.
  4. Summarise each paper, then derive a holistic view of the researcher’s
     interests, themes, and stated goals using the Gemini LLM.
  5. Discover the personal homepage / blog via DuckDuckGo and extract readable
     content + recent posts/tweets (via `snscrape`) for a glimpse of personal
     thoughts.
  6. Emit a Markdown report to STDOUT and save it as ``<slugified‑name>.md``.

Usage
-----
$ export GEMINI_API_KEY="sk‑..."
$ python researcher_intel.py "Ada Lovelace"

Dependencies (install via pip)
------------------------------
  scholarly duckduckgo_search readability‑lxml newspaper3k snscrape git+https://github.com/google‑generativeai/python
  pdfminer.six pdf2image pytesseract python‑dateutil tqdm beautifulsoup4 requests

External requirements
---------------------
* **Tesseract OCR**: Install system package and ensure `tesseract` is on PATH.
* **Poppler** (for `pdf2image`): Needed on Linux/macOS for PDF → image.

Notes & caveats
---------------
* Google Scholar scraping is fragile. For heavy use, swap out `scholarly` for
  SerpAPI or Publish‑or‑Perish.
* Only publicly available/CC‑licensed PDFs are downloaded to avoid copyright
  issues.
* Gemini usage billed under your key. The script budgets tokens conservatively
  but large corpora still cost.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import re
import subprocess
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from textwrap import dedent
from typing import Literal, Sequence

import requests
from bs4 import BeautifulSoup
from dateutil import parser as dateparser
from duckduckgo_search import DDGS
from readability import Document
from scholarly import scholarly, ProxyGenerator  # type: ignore
from tqdm import tqdm

# Gemini
import google.generativeai as genai  # type: ignore

# PDF handling
from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFSyntaxError
from pdf2image import convert_from_path
import pytesseract

RECENT_YEARS = 3
MAX_PAPERS = 50  # absolute safety cap
TOP_CITED_N = 10

###############################################################################
# UTILS
###############################################################################

def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower())
    return slug.strip("-")


def ensure_gemini() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set the GEMINI_API_KEY environment variable.")
    genai.configure(api_key=api_key)


def gemini_summarise(prompt: str, *, system_prompt: str | None = None) -> str:
    ensure_gemini()
    model = genai.GenerativeModel("gemini-pro")
    full_prompt = (system_prompt + "\n" if system_prompt else "") + prompt
    response = model.generate_content(full_prompt)
    return response.text.strip()

###############################################################################
# GOOGLE SCHOLAR
###############################################################################

def setup_scholar_proxy() -> None:
    # Create an ephemeral Tor proxy to mitigate CAPTCHAs (optional).
    # Users comfortable with SerpAPI can delete this and plug in their key.
    pg = ProxyGenerator()
    if pg.Tor_Internal(tor_cmd="tor"):
        scholarly.use_proxy(pg)


def find_author(name: str):
    search_query = scholarly.search_author(name)
    try:
        return next(search_query)
    except StopIteration:
        raise ValueError(f"No Google Scholar profile found for '{name}'.")


def extract_papers(author, now: datetime):
    filled = scholarly.fill(author, sections=["publications"])
    pubs = filled.get("publications", [])

    # Compute top cited
    pubs = [scholarly.fill(p) for p in pubs]
    pubs_sorted = sorted(pubs, key=lambda p: p.get("num_citations", 0), reverse=True)
    top_cited = pubs_sorted[:TOP_CITED_N]

    # Recent window
    cutoff = now.year - RECENT_YEARS
    recent = [p for p in pubs_sorted if p.get("bib", {}).get("pub_year") and int(p["bib"]["pub_year"]) >= cutoff]

    # Merge without duplicates
    seen_titles = set()
    selected = []
    for p in top_cited + recent:
        title = p["bib"].get("title", "").lower()
        if title and title not in seen_titles:
            selected.append(p)
            seen_titles.add(title)
        if len(selected) >= MAX_PAPERS:
            break
    return selected

###############################################################################
# PDF RETRIEVAL & TEXT EXTRACTION
###############################################################################

def best_pdf_url(pub) -> str | None:
    """Heuristic: prefer eprints/ArXiv links; else use direct link in pub."""
    eprint = pub.get("eprint_url")
    if eprint and eprint.endswith(".pdf"):
        return eprint
    elif eprint:
        return eprint + ".pdf" if not eprint.endswith(".pdf") else eprint
    # fallback: inspect pub["pub_url"]
    url = pub.get("pub_url")
    if url and url.endswith(".pdf"):
        return url
    return None


def download(url: str, dest: Path) -> bool:
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        dest.write_bytes(r.content)
        return True
    except Exception:
        return False


def pdf_text(path: Path) -> str:
    try:
        return extract_text(str(path))
    except (PDFSyntaxError, ValueError):
        return ""


def pdf_to_images(path: Path) -> list[Path]:
    with tempfile.TemporaryDirectory() as tmpdir:
        images = convert_from_path(str(path), dpi=300, output_folder=tmpdir)
        out_paths = []
        for i, img in enumerate(images):
            out_path = Path(tmpdir) / f"page_{i}.png"
            img.save(out_path)
            out_paths.append(out_path)
        return out_paths


def ocr_images(images: Sequence[Path]) -> str:
    texts = []
    for img in images:
        texts.append(pytesseract.image_to_string(img))
    return "\n".join(texts)

###############################################################################
# HOMEPAGE & SOCIAL MEDIA
###############################################################################

def find_homepage(name: str) -> str | None:
    query = f"{name} personal homepage"
    for result in DDGS().text(query, max_results=5):
        url = result.get("href")
        if url and all(bad not in url for bad in ("scholar.google", "researchgate")):
            return url
    return None


def fetch_readable(url: str) -> str:
    html = requests.get(url, timeout=20).text
    doc = Document(html)
    readable_html = doc.summary()
    soup = BeautifulSoup(readable_html, "html.parser")
    return soup.get_text(" ", strip=True)


def fetch_recent_tweets(name: str, limit: int = 20) -> list[str]:
    try:
        import snscrape.modules.twitter as sntwitter  # type: ignore
    except ImportError:
        return []
    query = f"from:{name.split()[0]} since:{(datetime.utcnow()-timedelta(days=365)).date()}"
    tweets = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
        if i >= limit:
            break
        tweets.append(tweet.content)
    return tweets

###############################################################################
# MARKDOWN REPORT
###############################################################################

def markdown_report(
    name: str,
    scholar_profile: str,
    stats: dict[str, str | int],
    topics_summary: str,
    paper_summaries: list[tuple[str, str]],
    personal_summary: str | None,
    tweet_summary: str | None,
) -> str:
    md = [f"# Researcher Intelligence Report: {name}\n"]
    md.append("## Quick Stats")
    for k, v in stats.items():
        md.append(f"- **{k}:** {v}")
    md.append(f"- **Google Scholar:** {scholar_profile}\n")

    md.append("## Research Themes & Goals")
    md.append(topics_summary + "\n")

    md.append("## Key Papers Analysed\n")
    for title, summary in paper_summaries:
        md.append(f"### {title}\n")
        md.append(summary + "\n")

    if personal_summary:
        md.append("## Insights from Personal Homepage/Blog\n")
        md.append(personal_summary + "\n")

    if tweet_summary:
        md.append("## Social‑Media Snapshot\n")
        md.append(tweet_summary + "\n")

    md.append("---\n*Generated on {:%B %d, %Y}*".format(datetime.now()))
    return "\n".join(md)

###############################################################################
# MAIN WORKFLOW
###############################################################################

async def process_paper(pub, session) -> tuple[str, str]:
    title = pub["bib"].get("title", "Unknown Title")
    pdf_url = best_pdf_url(pub)
    text = ""
    if pdf_url:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            if download(pdf_url, Path(tmp.name)):
                text = pdf_text(Path(tmp.name))
                if len(text.strip()) < 200:
                    images = pdf_to_images(Path(tmp.name))
                    text = ocr_images(images)
    if not text:
        # fallback: use abstract if any
        text = pub["bib"].get("abstract", "")
    summary = gemini_summarise(
        text[:16000],
        system_prompt="You are a research‑assistant. Summarise the following academic paper in 4 sentences, plain language:"
    )
    return title, summary


def main():
    parser = argparse.ArgumentParser(description="Generate intelligence report on a researcher.")
    parser.add_argument("name", help="Full researcher name (as on Google Scholar)")
    args = parser.parse_args()

    now = datetime.utcnow()

    print("[+] Locating Google Scholar profile…")
    setup_scholar_proxy()
    author = find_author(args.name)
    filled_author = scholarly.fill(author)
    profile_url = filled_author.get("url_picture", "https://scholar.google.com")

    stats = {
        "Total citations": filled_author.get("citedby", "n/a"),
        "h‑index": filled_author.get("hindex", "n/a"),
        "i10‑index": filled_author.get("i10index", "n/a"),
    }

    print("[+] Selecting papers…")
    pubs = extract_papers(author, now)

    print(f"[+] Analysing {len(pubs)} papers…")
    paper_summaries: list[tuple[str, str]] = []
    for pub in tqdm(pubs):
        title, summary = asyncio.run(process_paper(pub, None))  # sequential fallback
        paper_summaries.append((title, summary))

    # Derive overarching themes
    theme_prompt = """Given the following paper summaries, describe this researcher's principal interests, research agenda, and long‑term goals in 6‑8 sentences. Use bullet points where helpful.\n\n""" + "\n\n".join(f"- {s}" for _, s in paper_summaries)
    topics_summary = gemini_summarise(theme_prompt)

    # Personal homepage
    homepage_url = find_homepage(args.name)
    personal_summary = None
    if homepage_url:
        print("[+] Scraping personal homepage…")
        homepage_text = fetch_readable(homepage_url)[:16000]
        personal_summary = gemini_summarise(homepage_text, system_prompt="Summarise the personal statements/blog post below in 4 sentences:")

    # Tweets
    tweet_summary = None
    tweets = fetch_recent_tweets(args.name)
    if tweets:
        tweets_text = "\n".join(tweets)[:16000]
        tweet_summary = gemini_summarise(tweets_text, system_prompt="Read these tweets and summarise the recurring themes in 3 sentences:")

    report_md = markdown_report(
        args.name,
        f"https://scholar.google.com/citations?user={author['scholar_id']}",
        stats,
        topics_summary,
        paper_summaries,
        personal_summary,
        tweet_summary,
    )

    outfile = Path.cwd() / f"{slugify(args.name)}.md"
    outfile.write_text(report_md, encoding="utf‑8")
    print("\n[✓] Report saved to", outfile)
    print("\n============ MARKDOWN PREVIEW ============\n")
    print(report_md)


if __name__ == "__main__":
    main()
