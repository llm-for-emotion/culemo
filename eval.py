"""
Script for evaluating model predictions against ground truth data.
This script calculates accuracy and other metrics for emotion predictions.
"""

import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from qa_metrics.em import em_match
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def evaluate_predictions_binary(json_path: str) -> None:
    """
    Evaluates model predictions using binary accuracy (exact match).
    
    Args:
        json_path (str): Path to the JSON file containing predictions and ground truth
        
    The function prints:
        - Individual prediction results
        - Overall accuracy score
    """
    # Load the evaluation data
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    y_true_binary = []
    y_pred_binary = []

    # Process each prediction
    for item in data:
        ground_truth = item["emotion"]
        prediction = item["pred_emotion"]
        match = em_match([ground_truth], prediction)
        print("ground_truth:", ground_truth)
        print("prediction:", prediction)
        print("match:", match, "\n\n")

        # Convert to binary labels: 1 for correct, 0 for incorrect
        y_true_binary.append(1)
        y_pred_binary.append(1 if match else 0)

    # Calculate and print accuracy
    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    # Additional metrics (commented out)
    # precision, recall, f1, _ = precision_recall_fscore_support(y_true_binary, y_pred_binary, average="binary")

    print(f"Accuracy:  {accuracy:.4f}")
    # print(f"Precision: {precision:.4f}")
    # print(f"Recall:    {recall:.4f}")
    # print(f"F1:        {f1:.4f}")


if __name__ == "__main__":
    # Example evaluation on a specific output file
    evaluate_predictions_binary(
        "wo_country_pref_outputs/wo_country_pref_gemini1.5_flash/eng_wo-country-pref_gemini1.5_flash.json")

