import os
from dotenv import load_dotenv
import google.generativeai as genai

# Set your Gemini API key as an environment variable:
# export GEMINI_API_KEY="your_api_key"
# or set it in your system environment variables
load_dotenv()  # Load environment variables from .env file
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Error: GEMINI_API_KEY not found.")
    exit()

# Configure Gemini API
genai.configure(api_key=api_key) 

# Load Gemini model
model = genai.GenerativeModel("gemini-2.5-flash")

# Get topic from user
topic = input("Enter a topic: ").strip()

# Input validation
if not topic:
    print("Error: Topic cannot be empty.")
    exit()

# Prompt for Study Buddy
prompt = (
    f"Explain '{topic}' in simple terms in under 100 words. "
    "Make the explanation easy for beginners to understand."
)

try:
    # Generate response
    response = model.generate_content(prompt)

    print("\nStudy Buddy:")
    print(response.text)

except Exception as e:
    print(f"An error occurred: {e}")