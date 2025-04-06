﻿#!/bin/bash
echo "===== Installing Dependencies ====="
pip install --no-cache-dir -r requirements.txt

echo "===== Downloading Language Model ====="
python -c "from transformers import DistilBertTokenizer, DistilBertForSequenceClassification; print('Model imports successful')"

echo "===== Downloading spaCy Model ====="
python -m spacy download fr_core_news_sm
