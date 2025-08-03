from datetime import datetime
from dateutil import parser as date_parser

from gnews_fetcher import GNewsFetcher
import logic_parser as logic

import os, json, re, csv

io = json.load(open(os.path.join(os.path.dirname(__file__), "io.json")))["testing"] #TESTING
metrics = io["metrics"]    
queries = io["queries"]
countries = io["countries"]
years = io["years"]

prompts = json.load(open(os.path.join(os.path.dirname(__file__), "prompts.json")))
instruction = prompts["gnews_filter"]["instruction"]

def gen_searches():
    searches = []
    for query in queries:
        for metric in metrics:
            for year in years:
                mets = re.findall(r"\[([^\]]+)\]", query)
                search = query.replace("[metric]", metric).replace("[country]", "Kenya").replace("[year]", year)
                searches.append({"metrics": mets, "search": search})
    return searches

searches = gen_searches()

news_agent = GNewsFetcher(country=countries["Kenya"], max_results=3)
results = []
for search in searches:
    results.append(
        news_agent.get_news(search["search"], search["metrics"])
    )

filtering_agent = logic.TextParser(model="gpt-3.5-turbo")
def filter_results():
    filtered_results = []
    for result in results:
        for article in result:
            response = filtering_agent.get_chatgpt_response(
                instruction, 
                f"Title:{article['title']} Description: {article['description']} Published Date:{article['published date']} Intended Query:{article['query']}"
                )
            if response.strip().lower() == "yes":
                filtered_results.append(article)
    return filtered_results

filtered_articles = filter_results()
def show_articles():
    if not filtered_articles:
        print("No articles matched the criteria.")
        return
    for article in filtered_articles:
        print(f"Title: {article['title']}")
        print(f"Description: {article['description']}")
        print(f"Published Date: {article['published date']}")
        print(f"URL: {article['url']}")
        print(f"Query: {article['query']}\n")
# show_articles()

def save_to_csv(filtered_articles):
    if not filtered_articles:
        print("No articles to save.")
        return

    # Ensure outputs folder exists
    output_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(output_dir, f"news_results_{timestamp}.csv")

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
            row = {
                "year": pub_date.year,
                "month": pub_date.month,
                "country": "Kenya",
            }

            for metric in metrics:
                row[metric] = int(metric.lower() in query.lower())

            writer.writerow(row)

    print(f"✅ Saved {len(filtered_articles)} articles to {filename}")


save_to_csv(filtered_articles)