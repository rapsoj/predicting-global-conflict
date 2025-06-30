# Predicting Global Conflict

> Forecasting the risk of political instability and violence using augmented ACLED data and machine learning.

## Motivation

Political instability and large-scale violence continue to threaten international peace and security. By leveraging machine learning and publicly available data, we aim to build predictive tools that can help anticipate conflict events before they escalate — enabling timely interventions from policymakers, NGOs, and global institutions.

## Goal

Build an end-to-end machine learning pipeline that:

- Collects and integrates multi-source open data on conflict indicators
- Performs web scraping to gather news and reports relevant to regional tensions
- Engineers features related to political, social, and economic variables
- Trains machine learning models to forecast the likelihood of conflict events across different regions

## Project Components

```

predicting-global-conflict/
│
├── data/               # Data directory (linked externally via Google Drive)
├── notebooks/          # Jupyter notebooks for EDA, modeling, and exploration
├── src/                # Core source code: data loading, scraping, ML pipeline
├── reports/            # Generated reports, visualizations, and figures
├── references/         # Background papers, research summaries
├── tests/              # Unit tests for code components
└── requirements.txt    # Python dependencies

```

## 📁 Data Access

Due to the size of datasets, data is hosted externally on Google Drive. Access them via the links provided in:

```

data/external\_links.txt

````

Make sure to download and place the data in the appropriate `data/` subfolders before running notebooks or code.

## ⚙️ Getting Started

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/predicting-global-conflict.git
cd predicting-global-conflict
````

2. **Set up a virtual environment:**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. **Download the data** from the provided Drive links.

4. **Start exploring:** Use the Jupyter notebooks or run the pipeline from `src/`.

## Interests

This project sits at the intersection of:

* Political Science
* International Development
* Security Studies
* Data Science for Peace

## Skills Applied

* Research and literature review
* Data scraping and integration
* Data manipulation and analysis
* Machine learning modelling
* Visualisation and evaluation

## References

See `references/papers_summary.md` for a curated list of relevant academic and policy papers.

## Contributing

We welcome contributions from researchers, data scientists, and peacebuilders.

* Open an issue to suggest improvements
* Submit a pull request
* Respect the project structure and documentation

## License

MIT License. See `LICENSE` file for details.