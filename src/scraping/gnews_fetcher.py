from gnews import GNews
from datetime import datetime


class GNewsFetcher:
    def __init__(self, country="ZA", max_results=100, language="en", start_date=(2000,1,1), end_date=(2025,1,1)):
        self.gnews = GNews(language=language, country=country, max_results=max_results, start_date=start_date, end_date=end_date)
        self.country = country

    def get_news(self, search_query):
        self.search_query = search_query
        self.search_result = self.gnews.get_news(search_query)
        return self.format_news()
       
    def format_news(self):
        formatted_news = []
        for article in self.search_result:
            formatted_news.append({
                "title": article["title"],
                "description": article["description"],
                "published date": article["published date"],
                "url": article["url"],
                "country": self.country,
            })
        self.formatted_search = formatted_news
        return self.formatted_search  

def testing():
    news = GNewsFetcher(max_results=3)
    articles = news.get_news("protests in south africa")  
    for a in articles:
        print(f"Title: {a['title']}")
        print(f"Description: {a['description']}")
        print(f"Published Date: {a['published date']}")
        print(f"URL: {a['url']}\n")
        print(f"Query: {a['query']}\n")

if __name__ == "__main__":
    testing()
