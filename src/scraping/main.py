# LOADING AND IMPORTING LIBRARIES
from gnews_fetcher import GNewsFetcher
import logic_parser as logic
from news_boy import BrowserSim
import utils

import os, json

print("loading configurations...")
# LOADING CONFIGURATIONS
io = json.load(open(os.path.join(os.path.dirname(__file__), "io.json")))["testing"] #TESTING
metrics = io["metrics"]    
searches = io["searches"]
countries = io["countries"]
countries_names = list(countries.keys())
countries_codes = list(countries.values())
years = io["years"]

config = json.load(open(os.path.join(os.path.dirname(__file__), "config.json")))
gnews_filter = config['gnews_filter']
max_results = config['max results']
page_timeout = config['page timeout']
min_page_text_length = config['min page text length']

prompts = json.load(open(os.path.join(os.path.dirname(__file__), "prompts.json")))
news_instruction = prompts["instructions"]["news_instruction"]
news_reminder = prompts["queries"]["news_reminder"]

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

print("generating prompts...")
# GENERATE PROMPTS FOR PARSING AND FILTERING
news_instruction = utils.generate_instructions(news_instruction, metrics)
news_reminder = utils.generate_instructions(news_reminder, metrics)

print("fetching news articles...")
# FETCHING URLS
news_agent = GNewsFetcher(country=countries[gnews_searches[0]["country"]], max_results=max_results)
google_news_articles = news_agent.get_bundle_search(search_country_queries=gnews_searches, visited_urls=visited_urls)
print(f"Fetched a whole {len(google_news_articles)} articles...")

print("fetching website data...")
# BROWSING AND GETTING FULL TEXT
accessed_articles = []
browser = BrowserSim(page_wait=page_timeout, min_text_length=min_page_text_length)
browser.start()
for article in google_news_articles:
    full_text = browser.get_page(article["url"])
    if full_text is None:
        continue
    article["full_text"] = full_text
    accessed_articles.append(article)
browser.end()
print(f"Accessed {len(accessed_articles)} articles with full text.")

#TODO testing functionality of saving and loading json
utils.save_articles_json(accessed_articles, filename="accessed_articles.json")

# print("parsing articles...")
# parsed_articles = [] # MAKE MORE EFFICIENT
# # PARSE ARTICLES
# for article in accessed_articles:
#     # url = article["url"]
#     # if url in visited_urls:
#     #     continue
#     # visited_urls.append(article["url"])
    
#     full_text = get_news_data(article["url"], page_delay=page_timeout, overall_timeout=overall_timeout)
#     if full_text is None:
#         print(f"Failed to fetch article at {url}, skipping...")
#         continue
        
#     article["full_text"] = full_text
#     print(f"Summary: {full_text[:500]}...")  # Print first 500 characters of the article text

#     response = filtering_agent.get_chatgpt_response(
#         instruction_prompt, 
#         f"{news_query_prompt.replace('[country]', article['country'])} {article['full_text'][:min(len(article['full_text']), 4000*4)]}" 
#     )
#     if response.lower().strip() == "no":
#         continue
    
#     formatted_response = [row.split(',') for row in response.strip().split("\n")[1:]]
#     article["response"] = formatted_response
#     print(f"Response: {response}")
#     parsed_articles.append(article)

# #SAVE TO CSV
# utils.save_to_csv(parsed_articles)