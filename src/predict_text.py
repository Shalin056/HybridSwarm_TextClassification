import os
import sys
import numpy as np
import tensorflow as tf
import pickle
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Add src to path
module_path = r"C:\Users\shali\Documents\shalin\ASU_2nd_SEM\APM 523 Optimization\APM523_HybridSwarm_TextClassification\src"
if module_path not in sys.path:
    sys.path.append(module_path)

from models import BertClassifier  # Ensure models.py has the updated BertClassifier

# Paths
processed_dir = '../data/processed/'
models_dir = '../outputs/models/'
tfidf_vectorizer_path = os.path.join(processed_dir, 'tfidf_vectorizer.pkl')
gwo_lstm_path = os.path.join(models_dir, 'gwo_lstm.keras')
bert_path = os.path.join(models_dir, 'baseline_bert_default.keras')

# Load TF-IDF vectorizer
with open(tfidf_vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

# Load models
gwo_lstm_model = tf.keras.models.load_model(gwo_lstm_path)
bert_model = tf.keras.models.load_model(bert_path, custom_objects={'BertClassifier': BertClassifier})

# Load BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
max_length = 64

# Class labels (adjust based on your dataset)
class_labels = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

def preprocess_text(text, for_bert=False):
    if for_bert:
        encodings = tokenizer([text], max_length=max_length, padding='max_length', truncation=True, return_tensors='tf')
        return {'input_ids': encodings['input_ids'], 'attention_mask': encodings['attention_mask']}
    else:
        tfidf = vectorizer.transform([text]).toarray()
        return tfidf

def predict_text(model, text, is_bert=False):
    preprocessed = preprocess_text(text, for_bert=is_bert)
    pred = model.predict(preprocessed, verbose=0)
    class_idx = np.argmax(pred, axis=1)[0]
    return class_labels[class_idx], pred[0][class_idx]

# Interactive loop
print("Text Classification Demo (Enter 'quit' to exit)")
while True:
    user_input = input("Enter a news headline or text: ").strip()
    if user_input.lower() == 'quit':
        print("Exiting...")
        break
    if not user_input:
        print("Please enter some text!")
        continue

    # Predict with GWO-LSTM
    gwo_pred, gwo_prob = predict_text(gwo_lstm_model, user_input, is_bert=False)
    print(f"GWO-LSTM Prediction: {gwo_pred} (Confidence: {gwo_prob:.4f})")

    # Predict with BERT
    bert_pred, bert_prob = predict_text(bert_model, user_input, is_bert=True)
    print(f"BERT Prediction: {bert_pred} (Confidence: {bert_prob:.4f})")
    print("-" * 50)