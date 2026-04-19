import torch
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time
from tqdm import tqdm
import random
from sklearn.metrics import accuracy_score

# Generate dummy text dataset
def generate_synthetic_dataset(num_samples=1000):
    print(f"Generating large synthetic dataset with {num_samples} samples...")
    vocab = [
        "court", "appeal", "application", "dismissed", "judgment", "plaintiff", 
        "defendant", "contract", "breach", "damages", "law", "evidence", "trial",
        "jurisdiction", "statute", "claim", "liability", "negligence", "duty", "care",
        "case", "lawsuit", "verdict", "sentence", "prosecutor", "defense", "jury",
        "witness", "testimony", "objection", "sustained", "overruled", "guilty",
        "innocent", "acquitted", "convicted", "appeal", "supreme", "court"
    ]
    labels = ['affirmed', 'applied', 'approved', 'cited', 'considered', 
              'discussed', 'distinguished', 'followed', 'referred to', 'related']
    
    dataset = []
    for _ in range(num_samples):
        # Generate random text length between 50 and 600 words
        length = random.randint(50, 600)
        text = " ".join(random.choices(vocab, k=length))
        label = random.choice(labels)
        dataset.append((text, label))
    return dataset, labels

import argparse

def main():
    parser = argparse.ArgumentParser(description="Evaluate Legal-BERT Model")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to generate and evaluate")
    args = parser.parse_args()
    
    model_path = './legal_bert_v2'
    
    # Load model, tokenizer, and classes
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    classes = joblib.load(f'{model_path}/label_classes.joblib')
    
    # Put model on GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # 1. Generate Dataset
    num_samples = args.num_samples
    dataset, all_labels = generate_synthetic_dataset(num_samples=num_samples)
    
    y_true = []
    y_pred = []
    
    print(f"Evaluating on device: {device}...")
    start_time = time.time()
    
    batch_size = 16
    
    def extract_head_tail(t, head_words=128, tail_words=384):
        words = t.split()
        if len(words) <= 512:
            return t
        return " ".join(words[:head_words] + ["... [TEXT TRUNCATED] ..."] + words[-tail_words:])
    
    # 2. Inference Loop
    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
        batch = dataset[i:i+batch_size]
        texts = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        
        processed_texts = [extract_head_tail(t) for t in texts]
        
        inputs = tokenizer(processed_texts, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)
            predicted_indices = torch.argmax(probs, dim=1)
            
            for idx in predicted_indices:
                y_pred.append(classes[idx.item()])
        
        y_true.extend(labels)
        
    end_time = time.time()
    
    # 3. Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    print("\n" + "="*40)
    print("         EVALUATION RESULTS")
    print("="*40)
    print(f"Total Samples Evaluated: {len(dataset)}")
    print(f"Batch Size: {batch_size}")
    print(f"Overall Accuracy: {accuracy:.2%}")
    print(f"Total Time Taken: {end_time - start_time:.2f} seconds")
    print(f"Throughput: {len(dataset) / (end_time - start_time):.2f} samples/second")
    print(f"Device Used: {device}")
    print("="*40)

if __name__ == '__main__':
    main()
