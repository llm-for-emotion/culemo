"""
Script for running inference using Anthropic's Claude models.
This script processes text data and generates emotion predictions using Claude models.
It supports both language-specific and country-specific evaluations.
"""

from inference_utils import process_file, write_json
from anthropic import Anthropic
import os


# Configuration for the evaluation
TSV_FILE_PATH = "data/test/spn.tsv"
# Either LANG or COUNTRY must be set to None
LANG = None     # "English", "Arabic", "Spanish", "German", "Amharic", "Hindi", None
COUNTRY = "Mexico"      # "Ethiopia", "United Arab Emirates", "Germany", "India", "Mexico", None
MODEL_NAME = "claude-3-opus-20240229"
OUTPUT_JSON_PATH = "claude3_opus_outputs/countries/mex-eng_claude3_opus.json"

# Get API key from environment variable
API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not API_KEY:
    raise ValueError("Please set the ANTHROPIC_API_KEY environment variable")

# Initialize Anthropic client
client = Anthropic(
    api_key=API_KEY
)


def get_prediction(model: str, prompt: str) -> tuple[str, str]:
    """
    Gets a prediction from the Claude model.
    
    Args:
        model (str): Name of the Claude model to use (e.g., "claude-3-opus-20240229")
        prompt (str): The prompt to send to the model
        
    Returns:
        tuple[str, str]: A tuple containing (prompt, model_response)
        
    Note:
        The model is configured to return a single emotion word from the allowed set:
        'anger', 'fear', 'sadness', 'joy', 'guilt', or 'neutral'
    """
    response = client.messages.create(
        model=model,
        max_tokens=1000,  # Maximum number of tokens in the response
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    prediction = response.content[0].text
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


