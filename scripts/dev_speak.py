import os, sys
import torch

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import random
np.random.seed(0)
random.seed(0)

import nltk
from nltk.corpus import wordnet as wn
nltk.download('averaged_perceptron_tagger')
from nltk import word_tokenize

import ast
import itertools
from tqdm import tqdm
tqdm.pandas()
import itertools
import json



import re
from sklearn.feature_extraction import _stop_words
from sklearn.feature_extraction.text import CountVectorizer
vectorizer_model = CountVectorizer(stop_words="english")
import string
import jiwer



from transformers import BertTokenizer, AutoTokenizer, TFAutoModelForSequenceClassification
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
from sentence_transformers import SentenceTransformer, CrossEncoder, util, models



train_data = pd.read_csv('all_activities.csv')
init_size = train_data.shape[0]
device = "cuda:0" if torch.cuda.is_available() else "cpu"
from transformers import pipeline, AutoModel


pipe = pipeline("text-classification", model = "papluca/xlm-roberta-base-language-detection",device=device)
test_predictions = pipe(train_data['parsed_clauses'].tolist())
train_data['lang'] = [elem['label'] for elem in test_predictions]
train_data = train_data[train_data['lang'] == 'en']
train_data.to_csv('all_activities.csv')