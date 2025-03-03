# 🔍 Named Entity Recognition (NER) using DistilBERT

This project fine-tunes a **DistilBERT** model for Named Entity Recognition (NER) on the **MIT Restaurant Search Dataset**. The model is trained using **Hugging Face's Transformers library** to recognize entities such as food, cuisine types, and locations in restaurant-related queries.

---

## 📌 Features

- 📥 **Dataset Loading**: Downloads and processes the MIT Restaurant Search dataset.
- 🔤 **Tokenization**: Uses `distilbert-base-uncased` tokenizer for processing text.
- 🏷 **NER Tagging**: Assigns named entity labels to tokens.
- 🔄 **Fine-Tuning**: Trains the model with Hugging Face's `Trainer` API.
- 📈 **Evaluation**: Computes precision, recall, F1-score, and accuracy using `seqeval`.
- 🚀 **Inference Pipeline**: Deploys a trained model for entity recognition.



## 🚀 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/NER-DistilBERT.git
cd NER-DistilBERT
pip install -r requirements.txt
```

### 📦 Dependencies

- `transformers`
- `datasets`
- `torch`
- `numpy`
- `pandas`
- `evaluate`
- `seaborn`
- `matplotlib`
- `requests`

You can install them using:

```bash
pip install transformers datasets torch numpy pandas evaluate seaborn matplotlib requests
```


## 🔧 Training the Model

Run the script to fine-tune DistilBERT on the dataset:

```bash
python train.py
```

This will:

1. Download and preprocess the dataset.
2. Tokenize and align entity labels.
3. Train the model using Hugging Face's Trainer.
4. Save the fine-tuned model in the `finetuned-ner/` directory.


## 📊 Model Evaluation

After training, the model is evaluated using `seqeval`, which provides precision, recall, F1-score, and accuracy metrics.


## 🔎 Inference

To use the trained model for inference:

```python
from transformers import pipeline

ner_pipeline = pipeline('token-classification', model='finetuned-ner', aggregation_strategy='simple')

example_text = "which restaurant serves the best sushi in New York?"
print(ner_pipeline(example_text))
```

### 🔹 Sample Output

```json
[
  {"entity_group": "LOCATION", "word": "New York", "score": 0.99},
  {"entity_group": "FOOD", "word": "sushi", "score": 0.98}
]
```

## 📜 File Structure

```
NER-DistilBERT/
│── train.py  # Main script for training and inference
│── README.md  # Project documentation
│── requirements.txt  # Dependencies
│── finetuned-ner/  # Directory where the trained model is saved
```


## 🎯 Future Improvements

- Fine-tune on a larger NER dataset.
- Implement hyperparameter tuning.
- Deploy as an API using FastAPI.

## 📝 License

This project is open-source and available under the MIT License.
