# SearchResearcher
Use to search the recent publication, personal homepage, blog posts of a given scholar and summarize


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
