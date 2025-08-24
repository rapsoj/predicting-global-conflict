# LOADING AND IMPORTING LIBRARIES
from datetime import datetime
from dateutil import parser

from gnews_fetcher import GNewsFetcher
import logic_parser as logic
from news_boy import get_news_data
from . import utils

import pandas as pd

import os, json, re, csv

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
results = []
prev_country = None
total_results = 0
for search in gnews_searches:
    if search["country"] != prev_country:
        news_agent = GNewsFetcher(country=countries[search["country"]], max_results=max_results)
    prev_country = search["country"]
    result = news_agent.get_news(search["search"])
    total_results += len(result)
    results.append(result)
    
print(f"Fetched a whole {total_results} articles...")
print("generating prompts...")

# GENERATE PROMPTS FOR PARSING AND FILTERING
def gen_prompts(any_text : str, metrics : list):
    return any_text.replace("[all metrics]", ", ".join(metrics))

gnews_filter_prompt = gen_prompts(gnews_filter_prompt, metrics)
news_query_prompt = gen_prompts(news_query_prompt, metrics)

# FILTER
filtering_agent = logic.TextParser(model="gpt-3.5-turbo")
if gnews_filter:
    print("filtering articles...")
    def filter_results():
        filtered_articles = []
        for result in results:
            for article in result:
                response = filtering_agent.get_chatgpt_response(
                    instruction_prompt, 
                    f"{gnews_filter_prompt.replace('[country]',article['country'])} Title:{article['title']} Description:{article['description']} Published Date:{article['published date']}"
                    )
                if response.strip().lower() == "yes":
                    filtered_articles.append(article)
        return filtered_articles
    articles = filter_results()
    results = [articles] 

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