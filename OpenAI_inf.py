"""
Script for running inference using OpenAI's GPT models.
This script processes text data and generates emotion predictions using GPT models.
It supports both language-specific and country-specific evaluations.
"""

from inference_utils import process_file, write_json
from openai import OpenAI
import os


# Configuration for the evaluation
TSV_FILE_PATH = "data/test/eng.tsv"
# Either LANG or COUNTRY must be set to None
LANG = "English"     # "English", "Arabic", "Spanish", "German", "Amharic", "Hindi", None
COUNTRY = None      # "Ethiopia", "United Arab Emirates", "Germany", "India", "Mexico", None
MODEL_NAME = "gpt-4"
OUTPUT_JSON_PATH = "gpt4_outputs/eng_gpt-4.json"

# Get API key from environment variable
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

# Initialize OpenAI client
client = OpenAI(api_key=API_KEY)


def get_prediction(model: str, prompt: str) -> tuple[str, str]:
    """
    Gets a prediction from the GPT model.
    
    Args:
        model (str): Name of the GPT model to use (e.g., "gpt-4")
        prompt (str): The prompt to send to the model
        
    Returns:
        tuple[str, str]: A tuple containing (prompt, model_response)
        
    Note:
        The model is configured to return a single emotion word from the allowed set:
        'anger', 'fear', 'sadness', 'joy', 'guilt', or 'neutral'
    """
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    prediction = completion.choices[0].message.content
    return prompt, prediction


if __name__ == "__main__":
    # Process the TSV file and generate predictions
    output_data = process_file(
        tsv_file=TSV_FILE_PATH,
        model=MODEL_NAME,
        get_prediction=get_prediction,
        language=LANG,
        country=COUNTRY
    )
    
    # Save the results to a JSON file
    write_json(output_data, OUTPUT_JSON_PATH)


