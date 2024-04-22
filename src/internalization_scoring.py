import os, sys
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import random
import transformers
transformers.set_seed(0)
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
import string
import jiwer

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
from sentence_transformers import SentenceTransformer, CrossEncoder, util, models
cross_encoder = CrossEncoder('cross-encoder/stsb-distilroberta-base',device=device)



all_data = pd.read_csv('governed_activities.csv');all_data.shape[0]
asf_clauses = pd.read_csv('asf_rules.csv')

def bi_score(query,group_id):
  ##### Re-Ranking #####
  # Now, score all recurring activities with topic rules
  wer =[]
  rules = asf_clauses[asf_clauses['label']==group_id]['rules'].to_list()
  wer = cross_encoder.predict([(query,rule) for rule in rules])
  best_match  = rules[np.argmax(wer,axis=0)]
  return (best_match,max(wer))


all_data['cos_score'] = all_data.progress_apply(lambda x: bi_score(x['parsed_clauses'],x['group_id']),axis=1)
all_data[['best_match', 'cos_score']] = pd.DataFrame(all_data['cos_score'].tolist(), index=all_data.index)
all_data.to_csv('internalization_scores.csv',index=False)