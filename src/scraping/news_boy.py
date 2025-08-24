from newspaper import Article
import requests
import time
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright


from newspaper import Article
import requests
import time
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright


def get_news_data(url, page_delay = 3000, overall_timeout = 30000):
    """
    Fetch article text from a URL.
    1. Tries newspaper3k first.
    2. If 3k fails or returns empty, falls back to Playwright (JS rendering).
    """

    # 1️⃣ Try newspaper3k
    try:
        article = Article(url)
        article.download()
        article.parse()
        full_text = article.text
        if full_text and full_text.strip():
            if len(full_text) > 1000:
                return full_text
        print(f"newspaper3k returned empty for {url}, trying Playwright...")
    except Exception as e:
        print(f"newspaper3k failed for {url}: {e}, trying Playwright...")

    # 2️⃣ Fallback: Playwright for JS-rendered content
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=overall_timeout)
            page.wait_for_timeout(page_delay)  # small wait for JS content

            # Grab all paragraphs and list items
            paragraphs = page.query_selector_all("p, li")
            text_blocks = [
                p.text_content().strip()
                for p in paragraphs
                if p.text_content() and len(p.text_content().strip()) > 30
            ]

            browser.close()

            full_text = "\n".join(text_blocks)
            if full_text.strip():
                return full_text
            else:
                print(f"Playwright returned empty text for {url}")
                return None
    except Exception as e:
        print(f"Playwright failed for {url}: {e}")
        return None

def testing():
    url = "https://www.aljazeera.com/news/2025/8/9/india-says-six-pakistani-aircraft-shot-down-during-kashmir-conflict"

    print("\n--- Testing newspaper3k ---\n")
    text_3k = get_news_data(url)
    print(text_3k[:1000] if text_3k else "No text found.")

if __name__ == "__main__":
    testing()
