import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment and initialize OpenAI
load_dotenv()

class TextParser:
    def __init__(self, model="gpt-3.5-turbo"):
        self.ai= OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def get_chatgpt_response(self, instruction, prompt):
        response = self.ai.responses.create(
            model=self.model,
            instructions=instruction,
            input=prompt,
        )
        return response.output_text
