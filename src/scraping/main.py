# LOADING AND IMPORTING LIBRARIES
from datetime import datetime
from dateutil import parser as date_parser

from gnews_fetcher import GNewsFetcher
import logic_parser as logic
from news_boy import get_news_data

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

prompts = json.load(open(os.path.join(os.path.dirname(__file__), "prompts.json")))
instruction_prompt = prompts["general"]["instruction"]
gnews_filter_prompt = prompts["queries"]["gnews_filter"]
news_query_prompt = prompts["queries"]["news_query"]

print("generating searches...")
# GENERATING SEARCHES and initialising url tracker for efficiency
def gen_searches():
    search_array = []
    for search in searches:
        for country in countries_names:
            search_c = search.replace("[country]", country)
            for metric in metrics:
                search_m = search_c.replace("[metric]", metric)
                for year in years:
                    search_y = search_m.replace("[year]", year)
                    search_array.append({"search": search_y,
                                         "country": country
                                        })
    return search_array

gnews_searches = gen_searches()
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
        
        full_text = get_news_data(article["url"], agent="3k")
        if full_text is None:
            print(f"Failed to fetch article at {url}, skipping...")
            continue
        if full_text.strip() == "":
            print(f"Empty article at {url}, skipping...")
            continue
            
        article["full_text"] = full_text
        print(f"Summary: {full_text[:500]}...")  # Print first 500 characters of the article text

        response = filtering_agent.get_chatgpt_response(
            instruction_prompt, 
            f"{news_query_prompt.replace('[country]', article['country'])} {article['full_text']}"
        )
        if response.lower().strip() == "no":
            continue
        
        article["parsed_response"] = response
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
    if not filtered_articles:
        print("No articles to save.")
        return

    output_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(output_dir, f"{timestamp} scraping_results.csv")

    headers = ["year", "month", "country"] + [metric for metric in metrics]

    with open(filename, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()

        for article in filtered_articles:
            try:
                pub_date = date_parser.parse(article["published date"])
            except Exception as e:
                print(f"Error parsing date: {article['published date']} — {e}")
                continue

            query = article["query"]
    

            for metric in metrics:
                row[metric] = int(metric.lower() in query.lower())

            writer.writerow(row)

    print(f"Saved {len(filtered_articles)} articles to {filename}")


# save_to_csv(articles)