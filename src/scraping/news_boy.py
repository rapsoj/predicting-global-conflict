from newspaper import Article
import requests
import time
from bs4 import BeautifulSoup

def get_news_data(url,agent="3k", retries=2, delay=2, use_divs=False):
    """
    Downloads and parses the article from the given URL.
    Returns the article object with parsed content.
    """
    if agent == "3k":
        try:
            article = Article(url)
            article.download()
            article.parse()
            # article.nlp()

            # keywords = article.keywords
            # summary = article.summary
            full_text = article.text

            # return keywords, summary, full_text
            return full_text
        
        except Exception as e:
            print(f"An error occurred while processing the article: {e}")
            return None, None, None

    if agent == "req":
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

        for attempt in range(1, retries + 1):
            try:
                r = requests.get(url, headers=headers, timeout=10)
                r.raise_for_status()
                soup = BeautifulSoup(r.text, "html.parser")

                # Extract <p> and <li> tags
                paragraphs = [p.get_text().strip() for p in soup.find_all("p") if p.get_text().strip()]
                list_items = [li.get_text().strip() for li in soup.find_all("li") if li.get_text().strip()]

                text_blocks = paragraphs + list_items

                # Optionally include divs (careful: may include ads/menus)
                if use_divs:
                    div_text = [d.get_text().strip() for d in soup.find_all("div") if d.get_text().strip()]
                    text_blocks += div_text

                full_text = "\n".join(text_blocks)
                if full_text.strip():
                    return full_text

                # If empty, raise an exception to retry
                raise ValueError("No text extracted from page.")

            except Exception as e:
                print(f"Attempt {attempt} failed for {url}: {e}")
                if attempt < retries:
                    time.sleep(delay)
                else:
                    print(f"All {retries} attempts failed for {url}.")
                    return None
    
def testing():
    url = 'https://www.aljazeera.com/news/2025/8/9/india-says-six-pakistani-aircraft-shot-down-during-kashmir-conflict'
    article = Article(url)
    article.download()
    article.parse()
    print(f"Text: {article.text[:500]}...")  # Print first 500 characters of the article text
if __name__ == "__main__":
    testing()