#!/bin/bash
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m spacy download fr_core_news_sm
