@echo off
echo Installing required Python packages...

pip install --upgrade pip
pip install openai python-dotenv gnews python-dateutil

echo ✅ All packages installed successfully.
pause
