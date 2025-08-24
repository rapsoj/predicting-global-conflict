from gnews import GNews
from datetime import datetime


class GNewsFetcher:
    def __init__(self, country : str ="ZA", max_results : int =20, language : str ="en", start_date : datetime | None = datetime(2000,1,1), end_date : datetime | None =datetime(2025,1,1)) -> None:
        '''
        # Output
        Initiates gnews search object with desired country, max search results, and relevant dates
        '''
        self.country = country
        self.max_results = max_results
        self.start_date = start_date
        self.end_date = end_date
        self.gnews = GNews(language=language, country=country, max_results=max_results, start_date=start_date, end_date=end_date)
    
    def update_config(self, country : str | None = None, max_results : int | None = None, start_date : datetime | None = None, end_date : datetime | None = None) -> None:
        '''
        # Output
        Updates desired search object initiated in __init__
        '''
        if country:
            self.country = country
            self.gnews.country = country
        if max_results:
            self.max_results = max_results
            self.gnews.max_results = max_results
        if start_date:
            self.start_date = start_date
            self.gnews.start_date = start_date
        if end_date:
            self.end_date = end_date
            self.gnews.end_date = end_date
        self.gnews = GNews(language="en", country=self.country, max_results=self.max_results, start_date=self.start_date, end_date=self.end_date)
        return None
    
    def get_single_search(self, search_country_query : dict[str, str]) -> list[dict[str, str]]:
        '''
        # Output
        Gets news articles for a single search-country query pair
        '''
        search_result = self.gnews.get_news(search_country_query["search"])
        for article in search_result:
            self.add_metadata(article, search_country_query)
        return search_result

    def get_bundle_search(self, search_country_queries : list[dict[str, str]], visited_urls : list[str]) -> list[dict[str, str]]:
        '''
        # Output
        Gets news articles for multiple search-country query pairs
        '''
        filtered_results = []
        for query in search_country_queries:
            if query["country"] != self.country:
                self.update_config(country=query["country"])
            search_result = self.gnews.get_news(query["search"])
            for article in search_result:
                if article["url"] not in visited_urls:
                    self.add_metadata(article, query)
                    filtered_results.append(article)
                    visited_urls.append(article["url"])
        return filtered_results

    def add_metadata(self, article : dict[str, str], search_country_query : dict[str, str]) -> None:
        '''
        # Output
        Adds metadata to article dictionary
        '''
        article["country"] = search_country_query["country"]
        article["search"] = search_country_query["search"]

def testing():
    pass

if __name__ == "__main__":
    testing()
