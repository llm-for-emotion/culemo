"""
Script for running inference using Google's Gemini models.
This script processes text data and generates emotion predictions using Gemini models.
It supports both language-specific and country-specific evaluations.
"""

import google.generativeai as genai
from inference_utils import process_file, write_json
import os


# Configuration for the evaluation
TSV_FILE_PATH = "data/test/spn.tsv"
# Either LANG or COUNTRY must be set to None, but not both
LANG = None     # "English", "Arabic", "Spanish", "German", "Amharic", "Hindi", None
COUNTRY = "Mexico"      # "Ethiopia", "United Arab Emirates", "Germany", "India", "Mexico", None
MODEL_NAME = "gemini-1.5-flash"
OUTPUT_JSON_PATH = "gemini1.5_flash/countries/mex-eng_gemini1.5_flash.json"

# Get API key from environment variable
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable")

# Initialize Gemini API
genai.configure(api_key=API_KEY)


def get_prediction(model_name: str, prompt: str) -> tuple[str, str]:
    """
    Gets a prediction from the Gemini model.
    
    Args:
        model_name (str): Name of the Gemini model to use (e.g., "gemini-1.5-flash")
        prompt (str): The prompt to send to the model
        
    Returns:
        tuple[str, str]: A tuple containing (prompt, model_response)
        
    Note:
        The model is configured to return a single emotion word from the allowed set:
        'anger', 'fear', 'sadness', 'joy', 'guilt', or 'neutral'
    """
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    return prompt, response.text


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
    # The output JSON will contain:
    write_json(output_data, OUTPUT_JSON_PATH)


