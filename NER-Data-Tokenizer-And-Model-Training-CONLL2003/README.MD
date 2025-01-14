# Named Entity Recognition (NER) with Transformers: Data Tokenization and Model Training  

This project demonstrates how to train a **Named Entity Recognition (NER)** model using the **CONLL-2003 dataset** and **Hugging Face Transformers**. It covers data preprocessing, tokenization, and fine-tuning a transformer model for NER tasks.

---

## Project Highlights  

- **Data Preparation**:  
  Parse and preprocess the CONLL-2003 dataset for training, validation, and testing.  

- **Tokenization**:  
  Use Hugging Face Tokenizer to prepare text data for transformer-based models while handling special tokens and labels.  

- **Model Training**:  
  Fine-tune a pre-trained transformer model on the NER task using the processed dataset.  

- **Evaluation**:  
  Measure the model's performance using metrics such as accuracy, precision, recall, and F1-score.  

---

## Technologies Used  

- **Hugging Face Transformers** for model fine-tuning.  
- **Python** for scripting and data manipulation.  

---

## How to Use  

1. Clone the repository:  
   ```bash
   git clone https://github.com/mohammadreza-mohammadi94/Transformers-Hub
   ```  

2. Navigate to the directory:  
   ```bash
   cd Transformers-Hub/NER-Data-Tokenizer-And-Model-Training-CONLL2003
   ```  

3. Open and run the notebook:  
   ```bash
   NER_Data_Tokenizer_&_Model_Training.ipynb
   ```  

4. Ensure the required libraries are installed:  
   ```bash
   pip install transformers datasets torch
   ```  

5. Follow the notebook to preprocess the data, train the model, and evaluate its performance.  

This project aims to provide a clear understanding of training NER models with transformer architectures and serves as a foundation for further experimentation in NLP tasks.  
