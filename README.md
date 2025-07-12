# Automatic Alert Generation with NER and Sentiment Analysis

## Project Overview
This project implements an end-to-end system for automatic alert generation from news articles and social media posts. The system leverages Named Entity Recognition (NER) and Sentiment Analysis (SA) techniques to produce contextual alerts relevant to reputation monitoring, economic updates, and geopolitical risks, following a three-stage pipeline architecture as described in our paper.

## How It Works
1. **Named Entity Recognition (NER):** Identifies key entities such as people, organizations, monetary values, and locations using a hybrid neural architecture that combines contextual embeddings from a transformer model with linguistic features and character-level representations in a BiLSTM-CRF framework.
2. **Sentiment Analysis (SA):** Classifies text sentiment as positive, neutral, or negative using a neural network model that leverages both textual content and identified named entities.
3. **Alert Generation (AG):** Combines NER and SA results to generate meaningful alerts using a sequence-to-sequence model based on a fine-tuned T5 architecture.

## Example
**Input (Text):**  
"Musk accused of making a Nazi salute during Trump's inauguration..."

**Output:**  
"Reputation Risk: Elon Musk"

## Project Levels
The project can be developed at different levels of complexity:
- **Basic (up to 7.0 points):** Separate NER and SA models, rule-based alert generation.
- **Intermediate (up to 9.0 points):** Joint architecture for NER and SA with a combined loss function and AI-based alert generation.
- **Advanced (up to 10.0 points):** Image processing, captioning model (CNN + RNN) to generate textual descriptions and enhance alert generation.

## Tools & Requirements
- **Datasets Used:**
  - [CoNLL-2003](https://paperswithcode.com/dataset/conll-2003) (NER)
  - [Twitter Sentiment](https://www.kaggle.com/datasets/kazanova/sentiment140) (SA)
  - Custom Alert Dataset (Alert Generation)
- **Frameworks & Tools:**
  - [PyTorch](https://pytorch.org/) for neural network implementation
  - [spaCy](https://spacy.io/) for NER features extraction
  - [Hugging Face Transformers](https://huggingface.co/) for T5 model implementation
  - [TensorBoard](https://www.tensorflow.org/tensorboard) for training visualization

## Installation & Execution
```bash
# Clone the repository
git clone https://github.com/JVISERASS/Automatic-alert-generation-with-NER-and-SA.git
cd Automatic-alert-generation-with-NER-and-SA

# Create a virtual environment & install dependencies
python -m venv venv
source venv/bin/activate  # (Windows: venv\Scripts\activate)
pip install -r requirements.txt

# Download necessary SpaCy model
python -m spacy download en_core_web_trf

# Prepare dataset directories
mkdir -p src/data/ner_cache
mkdir -p src/data/sa_vocabs
mkdir -p src/models/ner_model
mkdir -p src/models/sa_model
mkdir -p src/models/alert_generator_model
```

## Model Training
Our system consists of three main components, each with its own training process. The following sections explain how to train each model.

### 1. Named Entity Recognition (NER) Model

The NER model uses a BiLSTM-CRF architecture with contextual embeddings, linguistic features, and character-level representations:

```bash
# Train the NER model
cd src/train/ner_train
python train.py

# Monitoring training progress with TensorBoard
tensorboard --logdir=runs
```

Key parameters in `config.py` you might want to adjust:
- `LEARNING_RATE`: Default is 1e-4
- `BATCH_SIZE`: Default is 512 (or 32 for lower memory)
- `EPOCHS`: Default is 50 with early stopping
- `MODEL_SAVE_PATH`: Path to save the best model

### 2. Sentiment Analysis (SA) Model

The SA model uniquely incorporates both text tokens and NER tags to enable more contextualized sentiment detection:

```bash
# Train the SA model
cd src/train/sa_train
python train.py

# To evaluate the model after training
python evaluate.py

# Monitoring training progress with TensorBoard
tensorboard --logdir=runs/sentiment_analysis_experiment
```

Key parameters in the training script you might want to adjust:
- `LEARNING_RATE`: Default is 1e-4
- `BATCH_SIZE`: Default is 512 (or 32 for lower memory)
- `EPOCHS`: Default is 100 with early stopping
- `PATIENCE`: Default is 7 epochs for early stopping

### 3. Alert Generation Model

The alert generation component is a sequence-to-sequence model based on a fine-tuned T5 architecture:

```bash
# Train the Alert Generation model
cd src/train/alert_train
python -m alert_train.train

# Monitoring training with TensorBoard
tensorboard --logdir=models/alert_generator_model/logs
```

Key parameters in `config.py` you might want to adjust:
- `LEARNING_RATE`: Default is 5e-5
- `BATCH_SIZE`: Default is 8 (with gradient accumulation)
- `MAX_EPOCHS`: Default is 100 with early stopping
- `EVAL_STEPS`: Default is 200 steps per evaluation

## Running Inference on New Data

Once you have trained all three models, you can use the integrated pipeline to process new text data:

### Method 1: Using the inference script

```bash
# Run the complete pipeline on data in src/data/news_tweets.txt
cd src/inference
python main.py
```

### Method 2: Processing a custom input file

Create a text file with your input texts (one per line), then:

```bash
# Process a custom file
cd src/inference
python main.py --input /path/to/your/input_file.txt --output /path/to/save/results.json
```

### Method 3: Using the API for programmatic access

You can import the inference pipeline in your Python code:

```python
from src.inference.main import load_ner_model, load_sa_model, load_alert_generator_model
from src.inference.main import predict_sentiment, generate_alert

# Load models (do this once)
ner_predictor = load_ner_model()
sa_model, text_vocab, ner_vocab, sentiment_vocab, sentiment_vocab_inverse, sa_device = load_sa_model()
alert_model, alert_tokenizer, alert_device = load_alert_generator_model()

# Process a text
text = "Apple announces a new iPhone model with revolutionary AI features."

# 1. Named Entity Recognition
ner_prediction = ner_predictor.predict(text)
ner_result = ner_prediction["ner_result"]

# 2. Sentiment Analysis
sentiment, confidence = predict_sentiment(
    sa_model, text, ner_result, text_vocab, ner_vocab, 
    sentiment_vocab_inverse, sa_device
)

# 3. Alert Generation
alert = generate_alert(
    alert_model, alert_tokenizer, text, ner_result, 
    sentiment, alert_device
)

print(f"Alert: {alert}")
```

## Experiment Results

Our experimental results demonstrated the effectiveness of our approach:
- NER component: F1 score of 84.05% on the CoNLL-2003 test set (batch size 512)
- SA component: Validation accuracy of 93.84% on Twitter data (batch size 512)
- Alert generation: ROUGE-1 of 73.99, ROUGE-2 of 62.09, and ROUGE-L of 73.60 (epoch 88)

For detailed analysis and discussion of results, please refer to our paper.

## Contributors
- Miguel Angel Vallejo de Bergia ([GitHub](https://github.com/mangelv011)) 
- Bernardo Ordas Cernadas ([GitHub](https://github.com/berordas)) 
- Javier Viseras Comin ([GitHub](https://github.com/JVISERASS)) 


