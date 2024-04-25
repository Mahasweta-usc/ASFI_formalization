
# session restart may be necessary
import os, sys
os.system("python -m spacy download en_core_web_sm")
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
import stanza
stanza.download('en')
nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,ner,depparse',use_gpu=True)
from sklearn.feature_extraction import _stop_words
from sklearn.feature_extraction.text import CountVectorizer
vectorizer_model = CountVectorizer(stop_words="english")
import string

import gensim
import pprint
from gensim import corpora
from gensim.utils import simple_preprocess
from gensim.test.utils import common_corpus, common_dictionary
from gensim.models.coherencemodel import CoherenceModel


import math
from scipy.spatial import distance
from scipy.signal import find_peaks
from scipy.stats import chi2_contingency

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.patches as mpatches
plt.rcParams["figure.figsize"] = (11.7,8.27)
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})

main_arguments = ['ARG' + str(idx) for idx in range(6)]

from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging

##extract predicate arguments

predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz",cuda_device=torch.cuda.current_device())
arg_types = ['ARGM-GOL','ARGM-COM','ARGM-NEG','ARGM-MOD','ARGM-DIR','ARGM-LOC','ARGM-MNR','ARGM-TMP','ARGM-ADV','ARGM-PRP','ARGM-ADJ','ARGM-LVB','ARGM-CAU' ,
             'ARGM-PNC','ARGM-EXT','ARGM-REC','ARGM-PRD','ARGM-DIS','ARGM-DSP','ARGM-RLC','ARG0','ARG1','ARG2','ARG3','ARG4','ARG5','ARG6','V']
predictor._model = predictor._model.cuda()

def srl_arg(sentences):
  # string = string.lower()
  parsed = predictor.predict_batch_json(
      sentences
  )
  outcome = dict()
  for sentence,parse in zip(sentences,parsed):
    sentence = sentence['sentence']
    verb_list = []
    for elem in parse['verbs']:
      parsed_items = dict()
      for item in arg_types:
        arg_found = re.findall("\[{}: (.*?)]".format(item), elem['description'])
        if len(arg_found) : parsed_items[item] = arg_found
      if bool(parsed_items): verb_list.append(parsed_items)
    outcome[sentence] = verb_list
  return outcome


#read original sentence and SRL tree to generate clauses
def parse2rule(text_surr,parsed):
  phrases, order = [],[]

  try: parsed['V']
  except: return
  # check for coresets
  if any(elem in main_arguments for elem in list(parsed.keys())):
    # print(valid)
    for k in arg_types:

      try: args = parsed[k]
      except: continue
      for v in args:
        # search argument position in original sentence

        query = "".join([elem for elem in v.lower() if elem.isalpha()])
        # print(text_surr,query)
        found = [m.start() for m in re.finditer(query,text_surr)]
        if found:
          try: found_ = [elem for elem in found if elem in range(min(order),max(order))][0]
          except: found_ = found[0]
          order.append(found_);phrases.append(v)

    # rearrange in relative order
    constructed = [x for _,x in sorted(zip(order,phrases))]
    return " ".join(constructed)

# text stats

from datetime import datetime

def convert_time(x):
  try:
    datetime.strptime(x.split()[0], '%Y-%m-%d')
    return datetime.fromisoformat(x.split()[0])
  except: return np.nan

##normalize 'podling' and 'project'

def process_(text):
  text = text.replace("\r\n"," ").replace("\n"," ")
  text = re.sub('podling', 'project', text,flags=re.IGNORECASE)
  regex = r'\b(\w+)(?:\W+\1\b)+'
  text = re.sub(regex, r'\1', text, flags=re.IGNORECASE)
  return text.lower()

def create_clauses(x):
  clauses = []
  # all frames parsed in a sentence
  for key,item in x['srl_parsed'].items():

    # verb frames per sentence
    text = "".join([elem for elem in key.lower() if elem.isalpha()])

    # Only keep non-overlapping clauses: prevents double counting activities
    verb_toks = item.copy()

    for parsed in item:
      try: action = parsed['V'][0].lower().strip()
      except: continue

      flag = False
      for parsed_ in item:
        if parsed_ != parsed:
          for k,v in parsed_.items():
            if any(action in argument.lower().split() and k != 'V' for argument in v):
              try: verb_toks.remove(parsed);flag=True
              except: pass
        if flag : break

    if item and not verb_toks: verb_toks = item.copy()

    flag = False
    # parse every verb frame per sentence
    for parsed in verb_toks:
      parsed = {k:[v] if not isinstance(v,list) else v for k,v in parsed.items()}
      rule = parse2rule(text,parsed)
      if rule : clauses.append(rule);flag = True
    # if not flag : clauses.append(key)

  # remove overlapping clauses
  all_clauses = []
  for clause in clauses:
    if all(clause not in elem or clause == elem for elem in clauses) : all_clauses.append(clause)
  return all_clauses

# Generate rules from policies
data = pd.read_csv("policies.csv");print(data.columns)

data['sentences'] = data['policy.statement'].apply(lambda x : [sentence.text for sentence in nlp(x).sentences]) ##for email data, use 'reply' column
data['srl_ip'] = data['sentences'].apply(lambda x : [{'sentence' : elem} for elem in x])
data['srl_parsed'] = data['srl_ip'].apply(lambda x: srl_arg(x))
data['parsed_clauses'] = data.apply(lambda x : create_clauses(x),axis=1)

rules_data = data.copy()
rules_data['label'] = rules_data.index.tolist()
rules_data = rules_data.explode('parsed_clauses')
rules_data.dropna(inplace=True);print('Number of parsed rules/clauses: ', rules_data.shape[0])
rules_data['rules'] = rules_data['parsed_clauses'].apply(lambda x : process_(x))
rules_data.to_csv('asf_rules.csv',index=False) #save emails to all_activities.csv
