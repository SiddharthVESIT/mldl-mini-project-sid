import os
import torch
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import PyPDF2
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report
import argparse

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text

def extract_head_tail(t, head_words=128, tail_words=384):
    words = t.split()
    if len(words) <= 512:
        return t
    return " ".join(words[:head_words] + ["... [TEXT TRUNCATED] ..."] + words[-tail_words:])

def main():
    parser = argparse.ArgumentParser()
    # Allowing user to pass ground truths as comma separated string
    parser.add_argument("--true_labels", type=str, default="", help="Comma separated true labels for the 4 PDFs (in alphabetical order of files)")
    args = parser.parse_args()

    pdf_dir = './legal_bert_v2'
    model_path = './legal_bert_v2'
    
    # 1. Find all PDFs and sort them so order is predictable
    pdf_files = sorted([f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')])
    
    if not pdf_files:
        print("No PDF files found in the directory.")
        return
        
    print(f"Found {len(pdf_files)} PDF files. Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    classes = joblib.load(f'{model_path}/label_classes.joblib')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    y_pred = []
    results = []
    
    print("\nPROCESSING PDFs...")
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        raw_text = extract_text_from_pdf(pdf_path)
        
        if not raw_text.strip():
            print(f"Could not extract text from {pdf_file}")
            y_pred.append("UNKNOWN")
            continue
            
        processed_text = extract_head_tail(raw_text)
        inputs = tokenizer(processed_text, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)
            predicted_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0][predicted_idx].item()
            prediction = classes[predicted_idx]
            
        y_pred.append(prediction)
        results.append({
            "File": pdf_file,
            "Prediction": prediction,
            "Confidence": confidence
        })
        
    print("\n" + "="*50)
    print("                PREDICTIONS")
    print("="*50)
    for res in results:
        print(f"File: {res['File']}")
        print(f" -> Predicted Outcome: {res['Prediction'].upper()} (Confidence: {res['Confidence']:.2%})\n")

    # Metrics Calculation
    if args.true_labels:
        y_true = [label.strip().lower() for label in args.true_labels.split(',')]
        if len(y_true) != len(y_pred):
            print(f"[ERROR] You provided {len(y_true)} true labels but there are {len(y_pred)} PDFs.")
            return
        
        print("="*50)
        print("                METRICS")
        print("="*50)
        # Using zero_division=0 to handle cases where classes might be missing in these 4 samples
        acc = accuracy_score(y_true, y_pred)
        # average='weighted' or 'macro' is good for multi-class
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        print(f"Accuracy:  {acc:.4f}")
        print(f"Recall:    {recall:.4f} (Macro Average)")
        print(f"Precision: {precision:.4f} (Macro Average)")
        print(f"F1 Score:  {f1:.4f} (Macro Average)")
        print("\nClassification Report:\n")
        print(classification_report(y_true, y_pred, zero_division=0))
    else:
        print("="*50)
        print("[NOTE]: Actual metrics (Accuracy, Recall) cannot be calculated without True Labels.")
        print("To see the metrics, run this script again and provide the true labels for these 4 cases:")
        print('Example: python evaluate_pdfs.py --true_labels "affirmed, dismissed, applied, distinguished"')

if __name__ == '__main__':
    main()
