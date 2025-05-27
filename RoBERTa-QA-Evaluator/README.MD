# 🤖 Question Answering with Transformers

This project demonstrates how to use Hugging Face's `transformers` library to answer questions using a pre-trained RoBERTa model fine-tuned on SQuAD v2.

---

## 📦 Requirements

Install dependencies with:

```bash
!pip install -q --upgrade datasets huggingface_hub fsspec
```

---

## 🚀 How It Works

* Loads the `deepset/roberta-base-squad2` model for question answering
* Uses the SQuAD v2 validation set (first 5 samples)
* Answers custom questions using a defined context

---

## 📝 Example Output

```
Question: Where Iran is located?
Answer: Middle East (Asia)

Question: What is capital of Iran?
Answer: Tehran
```


