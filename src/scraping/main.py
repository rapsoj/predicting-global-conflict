# LOADING AND IMPORTING LIBRARIES

from gnews_fetcher import GNewsFetcher
import logic_parser as logic
from news_boy import get_news_data
from . import utils

import os, json

print("loading configurations...")
# LOADING CONFIGURATIONS
io = json.load(open(os.path.join(os.path.dirname(__file__), "io.json")))["testing"] #TESTING
metrics = io["metrics"]    
queries = io["queries"]
searches = io["searches"]
countries = io["countries"]
countries_names = list(countries.keys())
countries_codes = list(countries.values())
years = io["years"]

config = json.load(open(os.path.join(os.path.dirname(__file__), "config.json")))
gnews_filter = config['gnews_filter']
max_results = config['max results']
page_timeout = config['page timeout']
overall_timeout = config['overall timeout']

prompts = json.load(open(os.path.join(os.path.dirname(__file__), "prompts.json")))
instruction_prompt = prompts["general"]["instruction"]
gnews_filter_prompt = prompts["queries"]["gnews_filter"]
news_query_prompt = prompts["queries"]["news_query"]

print("generating searches...")
# GENERATING SEARCHES and initialising url tracker for efficiency
gnews_searches = utils.generate_search_queries(
    google_search_templates=searches,
    country_names=countries_names,
    search_metrics=metrics,
    years=years
)
print(f"Generated {len(gnews_searches)} searches.")
visited_urls = []

print("fetching news articles...")
# FETCHING URLS
news_agent = GNewsFetcher(country=countries[gnews_searches[0]["country"]], max_results=max_results)
google_news_articles = news_agent.get_bundle_search(queries=gnews_searches, visited_urls=visited_urls)
print(f"Fetched a whole {len(google_news_articles)} articles...")

print("generating prompts...")
# GENERATE PROMPTS FOR PARSING AND FILTERING

gnews_filter_prompt = utils.generate_instructions(gnews_filter_prompt, metrics)
news_query_prompt = utils.generate_instructions(news_query_prompt, metrics)

print("parsing articles...")
parsed_articles = [] # MAKE MORE EFFICIENT
# PARSE ARTICLES
for result in results:
    for article in result:
        url = article["url"]
        if url in visited_urls:
            continue
        visited_urls.append(article["url"])
        
        full_text = get_news_data(article["url"], page_delay=page_timeout, overall_timeout=overall_timeout)
        if full_text is None:
            print(f"Failed to fetch article at {url}, skipping...")
            continue
            
        article["full_text"] = full_text
        print(f"Summary: {full_text[:500]}...")  # Print first 500 characters of the article text

        response = filtering_agent.get_chatgpt_response(
            instruction_prompt, 
            f"{news_query_prompt.replace('[country]', article['country'])} {article['full_text'][:min(len(article['full_text']), 4000*4)]}" 
        )
        if response.lower().strip() == "no":
            continue
        
        formatted_response = [row.split(',') for row in response.strip().split("\n")[1:]]
        article["response"] = formatted_response
        print(f"Response: {response}")
        parsed_articles.append(article)

#SAVE TO CSV
utils.save_to_csv(parsed_articles)