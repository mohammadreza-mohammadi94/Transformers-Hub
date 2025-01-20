# Question Answering with Transformers on SQuAD Dataset

This project demonstrates a basic implementation of a Question Answering (QA) model using **Hugging Face Transformers**. The model is fine-tuned on the **SQuAD (Stanford Question Answering Dataset)** using the `distilbert-base-uncased` pre-trained model.  

**Note**: This implementation is for demonstration purposes only and involves training on a subsample of the dataset for just 3 epochs due to limited computational resources. The model's performance can be significantly improved by training on the entire dataset for more epochs.

---

## Project Overview

- **Objective**: Train a DistilBERT-based model to answer questions given a context.
- **Dataset**: SQuAD (Stanford Question Answering Dataset), loaded via Hugging Face's `datasets` library.
- **Training**: The model was trained for 3 epochs on a Google Colab instance, taking approximately 3 hours.

---

## Implementation Steps

1. **Dataset Loading and Preprocessing**:
   - Tokenized questions and context paragraphs from the SQuAD dataset.
   - Aligned tokenized data with answers, ensuring proper start and end positions.

2. **Model**:  
   - Pre-trained `distilbert-base-uncased` model was fine-tuned for the question-answering task.
   - The Hugging Face `Trainer` API was used for training and evaluation.

3. **Training Details**:  
   - Training arguments included a learning rate of 2e-5 and batch sizes of 16.
   - Model was saved after training for further inference.

4. **Evaluation**:  
   - The model was evaluated on the validation set to assess performance.

5. **Inference**:
   - The fine-tuned model was tested with sample questions to verify its ability to extract answers from the given context.

---

## How to Use

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install dependencies:
   ```bash
   pip install transformers datasets evaluate
   ```

3. Run the script/notebook:
   - Ensure you have access to the SQuAD dataset and necessary computational resources.
   - Modify paths and configurations as needed.

---

## Key Notes

- **Training Limitations**: The model was trained on a subsample for demonstration purposes. For better results:
  - Train on the full dataset.
  - Increase the number of training epochs.
  - Use more powerful hardware (e.g., GPUs or TPUs).

- **Performance**: The current setup may produce suboptimal results. Fine-tuning on a larger dataset with longer training can enhance accuracy.

---

## Sample Test Output

**Context**:  
_Hugging Face is a technology company based in New York and Paris._  

**Question**:  
_Where is Hugging Face based?_  

**Answer**:  
_New York and Paris._
