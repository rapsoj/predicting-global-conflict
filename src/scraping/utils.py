import os
from datetime import datetime
from dateutil import parser
import pandas as pd

def generate_search_queries(google_search_templates : list[str], country_names : list[str], search_metrics : list[str], years : list[str]) -> list[dict]:
    '''
    # Outputs
    Array of dictionary with search query and desired country
    '''
    queries_to_search = []
    for search in google_search_templates:
        for country in country_names:
            search_c = search.replace("[country]", country)
            for metric in search_metrics:
                search_m = search_c.replace("[metric]", metric)
                for year in years:
                    search_y = search_m.replace("[year]", year)
                    queries_to_search.append({"search": search_y,
                                         "country": country
                                        })
    return queries_to_search

def display_article_results(articles : list[dict]):
    for article in articles:
        print(f"Title: {article['title']}")
        print(f"Description: {article['description']}")
        print(f"Published Date: {article['published date']}")
        print(f"URL: {article['url']}")
        print(f"Parsed: {article.get('parsed_response', 'N/A')}\n")

def generate_instructions(any_text : str, metrics : list[str]) -> str:
    return any_text.replace("[all metrics]", ", ".join(metrics))

def save_to_csv( # TODO
    data: list[list[str]],
    metrics: list[str],
    countries: list[str],
    output_dir: str = "outputs",
    date_format: str = "%m-%Y",  # default numeric month-year
) -> None:
    """
    Saves parsed article data into CSV files, one per metric.

    Parameters
    ----------
    data : list[list[str]]
        List of rows. Each row should look like:
        [country, metric, date1, date2, ...]
    metrics : list[str]
        List of known metrics to create separate CSVs.
    countries : list[str]
        List of country names (used as DataFrame index).
    output_dir : str, optional
        Directory to save CSVs into (default: "outputs").
    date_format : str, optional
        Format string for month-year labels (default: "%m-%Y").
        Examples: "%m-%Y" → "08-2022", "%B-%Y" → "August-2022"
    """
    print("Saving results to CSV...")

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
    all_months = pd.date_range(start=start, end=end, freq="MS")
    all_months_str = [d.strftime(date_format) for d in all_months]

    # Ensure output directory exists
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
                    month_str = dt.strftime(date_format)
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
