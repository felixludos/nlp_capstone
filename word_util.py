

import sys, os, time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
import csv
import json
import util
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

stop_words = set(stopwords.words('english'))

GLOBAL_BAG = None

fileroot = r'aristo-mini/questions'
filenames = ['AI2-Elementary-NDMC-Feb2016-Dev.jsonl', 'AI2-8thGr-NDMC-Feb2016-Dev.jsonl',
             'AI2-8thGr-NDMC-Feb2016-Train.jsonl', 'AI2-Elementary-NDMC-Feb2016-Train.jsonl']



def is_noun(tag):
	return tag in ['NN', 'NNS', 'NNP', 'NNPS']

def is_verb(tag):
	return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

def is_adverb(tag):
	return tag in ['RB', 'RBR', 'RBS']

def is_adjective(tag):
	return tag in ['JJ', 'JJR', 'JJS']

def penn_to_wn(tag):
	if is_adjective(tag):
		return wn.ADJ
	elif is_noun(tag):
		return wn.NOUN
	elif is_adverb(tag):
		return wn.ADV
	elif is_verb(tag):
		return wn.VERB
	return wn.NOUN

def normalize_text(text):

	tags = nltk.pos_tag(word_tokenize(text))
	tokens = []
	for tag in tags:
		wn_tag = penn_to_wn(tag[1])
		tokens.append(WordNetLemmatizer().lemmatize(tag[0],wn_tag))
	return tokens

def filter_tokens(tok):
	return len(tok) > 1 and tok != '___'

def bag_text(text):
	return Counter(tok for tok in normalize_text(text.lower()) if filter_tokens(tok))

def bag_question(q):
	bag = bag_text(q['question']['stem'])
	for ans in q['question']['choices']:
		bag += bag_text(ans['text'])
	return bag

def tfidf(bag, full):
	terms = []
	for x, tf in bag.items():
		df = full[x]
		terms.append((x, tf/df))
	return sorted(terms, reverse=True, key=lambda t: t[1])

def load_questions(*paths):
	if len(paths) == 0:
		paths = [os.path.join(fileroot, fn) for fn in filenames]
		
	questions = []
	
	for path in paths:
		with open(path, 'r') as f:
			lines = f.readlines()
			for line in lines:
				questions.append(json.loads(line))
			
	return questions

def load(root=None):
	
	if root is None:
		root = fileroot
	
	for fn in filenames:
		
		questions = load_questions(os.path.join(root, fn))
		
		bags = [bag_question(q) for q in questions]
		
		global GLOBAL_BAG
		
		if GLOBAL_BAG is None:
			GLOBAL_BAG = Counter()
			
		for bag in bags:
			GLOBAL_BAG += bag
			
def extract_bag(bag):
	vals = tfidf(bag, GLOBAL_BAG)
	top = [w for w, v in vals[:10]]
	critical = [w for w, v in vals if v > 0.3]
	return set(top if len(top) > len(critical) else critical)
	
def extract(q):
	
	if GLOBAL_BAG is None:
		load()
		
	tokens = {}
	
	bag = bag_text(q['question']['stem'])
	for choice in q['question']['choices']:
		ans = bag_text(choice['text']) + bag
		#     ans.subtract(bag)
		#     ans = Counter(w for w, n in ans.items() if n > 0)
		vals = tfidf(ans, GLOBAL_BAG)
		top = [w for w, v in vals[:10]]
		critical = [w for w, v in vals if v > 0.3]
		tokens[choice['label']] = set(top if len(top) > len(critical) else critical)
	
	return tokens
	
	