import pandas as pd
from dateutil import parser
import os
from datetime import datetime

# Example input
data = [
    ["Kenya", "coup", "08-2022", "September-2023"],
    ["Kenya", "election", "August-2022"],
    ["Kenya", "economic crisis", "09-2023", "December-2023"],
    ["Uganda", "coup", "January-2023"],
    ["Uganda", "economic crisis", "March-2023", "07-2023"]
]

# Known metrics
metrics = ["coup", "election", "economic crisis"]

# Collect all countries
countries = sorted(set(row[0] for row in data))

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
all_months_str = [d.strftime("%m-%Y") for d in all_months]

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
                month_str = dt.strftime("%m-%Y")
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
