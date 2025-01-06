import os
from dotenv import load_dotenv
import openai

# Load environment variables from .env file
load_dotenv()

def correct_sentence_with_gpt(text):
    # Set your OpenAI API key
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    # Send the prompt to GPT
    messages = [
        {"role": "system", "content": "Jesteś pomocnym asystentem poprawiającym polskie zdania."},
        {"role": "user", "content": f"Popraw błędy w zdaniu: {text}"}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Use "gpt-4" or "gpt-3.5-turbo"
        messages=messages
    )

    # Extract the corrected sentence
    corrected_text = response["choices"][0]["message"]["content"]

    return corrected_text

if __name__ == "__main__":
    # Example sentence with errors
    sentence = "mieryła sią takka kobiecina że już me ma"
    corrected = correct_sentence_with_gpt(sentence)

    print("Original sentence:", sentence)
    print("Corrected sentence:", corrected)
