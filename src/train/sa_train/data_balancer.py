import os
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import pipeline
from collections import Counter
from tqdm import tqdm

# Carga un dataset de noticias o tweets que tienda a tener más sentimiento
def create_balanced_dataset():
    # Cargar datasets
    conll = load_dataset("conll2003")
    news = load_dataset("ag_news", split="train").shuffle(seed=42).select(range(5000))
    
    # Configurar pipeline de sentiment analysis
    sentiment_pipeline = pipeline(
        "sentiment-analysis", 
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=0 if torch.cuda.is_available() else -1
    )
    
    # Mapear ag_news a formato conll
    def convert_news_to_conll(example):
        # Tokenizar el texto
        text = example["text"]
        tokens = text.split()[:128]  # Limitar longitud
        
        # Crear etiquetas O para todos los tokens
        ner_tags = [0] * len(tokens)  # 0 = 'O' en CoNLL
        
        return {
            "tokens": tokens,
            "ner_tags": ner_tags,
            "pos_tags": [0] * len(tokens),  # Etiquetas ficticias
            "chunk_tags": [0] * len(tokens),  # Etiquetas ficticias
            "id": example.get("id", "news-" + str(hash(text) % 10000))
        }
    
    # Procesar ag_news
    processed_news = news.map(convert_news_to_conll)
    
    # Etiquetar sentimiento para todos los datasets
    def add_sentiments(batch):
        with torch.no_grad():
            sentences = [" ".join(tokens) for tokens in batch["tokens"]]
            results = sentiment_pipeline(sentences, truncation=True, max_length=128)
            
        labels = []
        for res in results:
            if res["label"] == "LABEL_0":
                labels.append("NEG")
            elif res["label"] == "LABEL_1":
                labels.append("NEU")
            elif res["label"] == "LABEL_2":
                labels.append("POS")
        
        batch["sentiment"] = labels
        return batch
    
    # Etiquetar sentimientos
    conll_with_sentiment = conll.map(add_sentiments, batched=True, batch_size=32)
    news_with_sentiment = processed_news.map(add_sentiments, batched=True, batch_size=32)
    
    # Analizar distribución
    def print_distribution(dataset, name):
        sentiments = dataset["sentiment"]
        counter = Counter(sentiments)
        total = len(sentiments)
        print(f"\nDistribución de sentimientos en {name}:")
        for label, count in counter.items():
            percentage = (count / total) * 100
            print(f"  {label}: {count} ({percentage:.2f}%)")
    
    print_distribution(conll_with_sentiment["train"], "CoNLL (original)")
    print_distribution(news_with_sentiment, "News")
    
    # Balancear el dataset
    # (Implementar lógica para seleccionar ejemplos y balancear)
    
    # Combinar datasets
    combined = concatenate_datasets([conll_with_sentiment["train"], news_with_sentiment])
    
    print_distribution(combined, "Dataset combinado")
    
    # Guardar dataset balanceado
    combined.save_to_disk("balanced_dataset")
    print("Dataset balanceado guardado en 'balanced_dataset'")

if __name__ == "__main__":
    create_balanced_dataset()
