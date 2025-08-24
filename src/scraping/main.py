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

# DISPLAY ARTICLES
def show_articles():
    for article in parsed_articles:
        print(f"Title: {article['title']}")
        print(f"Description: {article['description']}")
        print(f"Published Date: {article['published date']}")
        print(f"URL: {article['url']}")
        print(f"Parsed: {article['parsed_response']}\n")
show_articles()

# SAVE RESULTS TO CSV
def save_to_csv(filtered_articles):
    print("saving results to CSV...")

    data = parsed_articles

    # Parse all dates and determine full month range
    all_dates = []
    for row in data:
        for date_str in row[2:]:
            try:
                dt = parser.parse(date_str)
                all_dates.append(dt.replace(day=1))
            except Exception as e:
                print(f"Skipping unparseable date {date_str}: {e}")

    if not all_dates:
        raise ValueError("No valid dates found!")

    start = min(all_dates)
    end = max(all_dates)

    # Generate full list of month-year strings
    all_months = pd.date_range(start=start, end=end, freq='MS')
    all_months_str = [d.strftime("%B-%Y") for d in all_months]

    # Create output directory
    output_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Generate one DataFrame per metric and save
    for metric in metrics:
        # Initialize DataFrame
        df_metric = pd.DataFrame(0, index=countries, columns=all_months_str)

        # Fill DataFrame
        for row in data:
            country, row_metric, *dates = row
            if row_metric != metric:
                continue
            for date_str in dates:
                try:
                    dt = parser.parse(date_str)
                    month_str = dt.strftime("%B-%Y")
                    if month_str in df_metric.columns:
                        df_metric.loc[country, month_str] = 1
                except Exception as e:
                    print(f"Skipping unparseable date {date_str}: {e}")

        # Reset index for saving
        df_metric = df_metric.reset_index().rename(columns={"index": "country"})

        # Save to CSV
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(output_dir, f"{metric.replace(' ', '_')}_{timestamp}.csv")
        df_metric.to_csv(filename, index=False)
        print(f"Saved metric '{metric}' to {filename}")

save_to_csv(parsed_articles)