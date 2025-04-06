#!/bin/bash
echo "===== Installing Dependencies ====="
pip install --no-cache-dir -r requirements.txt

echo "===== Downloading Language Model ====="
python -c "
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-multilingual-cased', torch_dtype=torch.float16)
"

echo "===== Downloading spaCy Model ====="
python -m spacy download fr_core_news_sm