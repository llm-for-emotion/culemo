"""
Script for running inference using Ollama models.
This script processes text data and generates emotion predictions using locally hosted Ollama models.
It supports both language-specific and country-specific evaluations.
"""

from inference_utils import process_file, write_json
from ollama import chat, ChatResponse

# Configuration for the evaluation
TSV_FILE_PATH = "data/test/test2.tsv"
# Either LANG or COUNTRY must be set to None
LANG = "Arabic"     # "English", "Arabic", "Spanish", "German", "Amharic", "Hindi", None
COUNTRY = None      # "Ethiopia", "United Arab Emirates", "Germany", "India", "Mexico", None
MODEL_NAME = "llama3.2:1b-instruct-q8_0"  # Local Llama model to use
OUTPUT_JSON_PATH = "test.json"


def get_prediction(model: str, prompt: str) -> tuple[str, str]:
    """
    Gets a prediction from the local Llama model using Ollama.
    
    Args:
        model (str): Name of the Llama model to use
        prompt (str): The prompt to send to the model
        
    Returns:
        tuple[str, str]: A tuple containing (prompt, model_response)
    """
    response: ChatResponse = chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )
    print(response["message"]["content"])
    return prompt, response["message"]["content"]


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


