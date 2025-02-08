# Twitter Sentiment Analysis with BERT

This repository contains the code for performing multi-class sentiment analysis on a Twitter dataset using a pre-trained BERT model. The model classifies tweets into different sentiment categories. The pipeline includes data preprocessing, tokenization, model training, evaluation, and prediction generation.

## Table of Contents
- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Files](#files)
- [Results](#results)
- [License](#license)

## Project Overview

The project utilizes Hugging Face Transformers and PyTorch to fine-tune a BERT model for sentiment classification. The dataset used for training is the "Twitter Multi-Class Sentiment" dataset, which contains labeled tweets with different sentiment labels. The steps include:
1. Loading and preprocessing the dataset.
2. Tokenizing the text data for BERT.
3. Training the BERT model.
4. Evaluating the model on a test set.
5. Visualizing the performance with confusion matrix and other plots.

## Requirements

The following Python packages are required:
- `torch`
- `transformers`
- `datasets`
- `evaluate`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `pandas`
- `numpy`

These can be installed using the following:

```bash
pip install torch transformers datasets evaluate scikit-learn matplotlib seaborn pandas numpy
```

## Installation

1. Clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/twitter-sentiment-analysis-bert.git
cd twitter-sentiment-analysis-bert
```

2. Install the necessary dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training the Model
1. Clone the repository and navigate to the project folder.
2. Run the `main.py` script to start training:

```bash
python main.py
```

This will:
- Load the Twitter sentiment dataset.
- Preprocess the data.
- Train a BERT model on the sentiment classification task.
- Evaluate the model's performance on the test set.
- Generate and display the confusion matrix.

### Generating Predictions
To generate predictions for new text input, you can call the `get_prediction` function with your input text:

```python
text_sample = "I am super happy today!"
prediction = get_prediction(text_sample, model, tokenizer, id2label)
print(f"Prediction: {prediction}")
```

## Files

- `main.py`: Main script that runs the entire pipeline from data loading to model training and evaluation.
- `requirements.txt`: List of Python dependencies for the project.
- `data/`: (Optional) Folder for storing any locally saved data.
- `models/`: (Optional) Folder for saving the trained BERT model.

## Results

After running the training script, the model will output:
1. A classification report showing metrics like accuracy and F1-score.
2. A confusion matrix for visualizing the model's performance across different sentiment classes.

Example of the output:
```bash
Prediction: joy
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
