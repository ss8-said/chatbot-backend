#!/bin/sh
echo "===== INSTALLING ====="
pip install -r requirements.txt
python -m spacy download fr_core_news_sm
echo "===== BUILD COMPLETE ====="