"""
Utility functions for processing and evaluating LLM responses across different languages and countries.
This module handles prompt generation, file processing, and result storage.
"""

import csv
import json


def get_prompt(language: str, country: str, text: str) -> str:
    """
    Generates a prompt in the specified language or for the specified country.
    
    Args:
        language (str): The language to generate the prompt in (e.g., "English", "Arabic")
        country (str): The country context for the prompt (e.g., "Mexico", "Germany")
        text (str): The input text to be evaluated
        
    Returns:
        str: A formatted prompt in the specified language
        
    Raises:
        ValueError: If neither language nor country is provided
    """
    if language == "English":
        prompt = f"""You live in the United States of America, choose the best emotion you feel for the given question. Emotions can only be 'anger', 'fear', 'sadness', 'joy', 'guilt', or 'neutral'. Return only a single emotion word from the list of emotions without further explanation. \nText: {text}\nAnswer:"""
    elif language == "Arabic":
        prompt = f"""أنت تعيش في دولة الإمارات العربية المتحدة، اختر الحالة العاطفية الأفضل التي تشعر بها تجاه السؤال المطروح. يمكن أن تكون العواطف إما "غضب" أو "خوف" أو "حزن" أو "فرح" أو "ذنب" أو "محايد". أعد فقط كلمة عاطفة واحدة من قائمة العواطف دون أي تفسير إضافي.\n النص: {text}\n الإجابة:"""
    elif language == "Spanish":
        prompt = f"""Vives en México. Elige la emoción que sientes más a menudo en la pregunta. Las emociones pueden ser 'enojo','tristeza','culpa','alegría','miedo' o 'neutral'. Solo responde con una palabra de la lista de emociones sin más explicaciones. \nTexto: {text}\nRespuesta:"""
    elif language == "Amharic":
        prompt = f"""የምትኖረው ኢትዮጵያ ውስጥ ነው፣ ለተሰጠው ጥያቄ የሚሰማህን ስሜት ምረጥ። ስሜቶች 'ቁጣ', 'ጥፋተኛ', 'ሀዘን', 'ደስታ', 'ፍርሀት' ወይም 'መደበኛ' ብቻ ሊሆኑ ይችላሉ:: ያለተጨማሪ ማብራሪያ ከስሜቶች ዝርዝር ውስጥ አንዱን ስሜት ብቻ ይመልሱ።\n"ጽሑፍ፡ {text}\nመልስ፡"""
    elif language == "German":
        prompt = f"""Sie leben in Deutschland. Wählen Sie die beste Emotion, die Sie für die gestellte Frage empfinden. Emotionen können nur 'wut', 'freude', 'traurigkeit', 'schuld', 'furcht' oder 'neutral' sein. Geben Sie nur ein einziges Emotionswort aus der Liste der Emotionen ohne weitere Erklärung zurück. \nText: {text}\nAntwort:"""
    elif language == "Hindi":
        prompt = f"""आप भारत में रहते हैं, दिए गए प्रश्न के लिए अपनी सबसे अच्छी भावना चुनें। भावनाएँ केवल 'उदासी', 'आनंद', 'अपराध', 'गुस्सा', 'डर', या 'सामान्य' हो सकती हैं। बिना किसी अतिरिक्त स्पष्टीकरण के भावनाओं की सूची से केवल एक ही भावना शब्द लौटाएँ। \nपाठ: {text}\nउत्तर:"""
    elif country:
        prompt = f"You live in {country}, choose the best emotion you feel for the given question. Emotions can be only either 'anger', 'fear', 'sadness', 'joy', 'guilt', or 'neutral'. Return only a single emotion word from the list of emotions without further explanation. \nText: {text}"
    else:
        raise ValueError("Either a valid 'language' or 'country' must be provided.")
    return prompt


def process_file(tsv_file: str, model: str, get_prediction, language: str = None, country: str = None) -> list[dict]:
    """
    Processes a TSV file containing emotion evaluation data and generates predictions using the specified model.
    
    Args:
        tsv_file (str): Path to the TSV file containing the evaluation data
        model (str): Name of the model to use for predictions
        get_prediction (callable): Function to get predictions from the model
        language (str, optional): Language to use for prompts
        country (str, optional): Country context to use for prompts
        
    Returns:
        list[dict]: List of dictionaries containing the evaluation results
        
    Note:
        Either language or country must be provided, but not both
    """
    results = []

    with open(tsv_file, "r", encoding="utf-8") as file:
        tsv_reader = csv.reader(file, delimiter="\t")
        next(tsv_reader)  # Skip header row

        for count, row in enumerate(tsv_reader, start=1):
            print(count)

            # Parse row based on file format
            if language == "English":
                text, gt_emotion, gt_sentiment = row        # only for english tsv
            elif language and not country:
                text_eng, text, emotion_eng, gt_emotion, sentiment_eng, gt_sentiment = row
            elif country and not language:
                text, _, gt_emotion, _, gt_sentiment, _ = row
            else:
                raise Exception("ERROR!")

            # Get prediction from model
            prompt, pred_emotion = get_prediction(model, get_prompt(language, country, text))
            
            # Store results
            results.append(
                {
                    "prompt": prompt,
                    "text": text,
                    **({"country": country} if country else {"language": language}),
                    "emotion": gt_emotion,
                    "pred_emotion": pred_emotion,
                    "model": model,
                }
            )
    return results


def write_json(data: list[dict], output_path: str) -> None:
    """
    Writes the evaluation results to a JSON file.
    
    Args:
        data (list[dict]): List of dictionaries containing the evaluation results
        output_path (str): Path where the JSON file should be written
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Data successfully written to {output_path}")

