@echo off
echo Installing required Python packages...

:: Upgrade pip
pip install --upgrade pip

:: Core packages
pip install openai python-dotenv gnews python-dateutil

:: Newspaper3k and parsing dependencies
pip install newspaper3k lxml html5lib beautifulsoup4

:: NLTK + Punkt tokenizer data
pip install nltk
python -m nltk.downloader punkt punkt_tab

echo ✅ All packages installed successfully.
pause
