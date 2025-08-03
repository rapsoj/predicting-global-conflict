Hello.

# First Things First
Please give the following documents a read and just look over them:
1. prompts.json
2. io.json
3. logic_parser.py
4. gnews_fetcher.py

## prompts.json
This is a work in progress file dedicated to instructions or supplementary things to be thrown at the OpenAI API. So far it just deals with the preliminary scraping of gnews, and not full articles themselves.

## io.json
This file is dedicated to all IO from and between the internet and the OpenAI API, both for testing and official use.

## logic_parser.py
The simplest OpenAI API.

## gnews_fetcher.py
Gets news from the gnews API and formats it slightly.

---

Now that you have done that, you can have a proper look at:
# main.py
1. It extracts the relevant information from io.json and prompts.json
2. It generates searches and searches gnews.
3. It filters them using the OpenAI API based on the data received at the provided query.
4. The filtered searches then get converted to a binary csv with date-time and country data.

# Running
1. Run the file install_requirements.bat to download the dependencies.
2. Run the file main.py to get data.