
import sys, os, time
import numpy as np
import matplotlib.pyplot as plt
# import gensim
import pytorch_pretrained_bert as ppb
import torch
from torch import nn
from torch.nn import functional as F
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
# import logging
# logging.basicConfig(level=logging.INFO)
# from glove import Glove
# from glove import Corpus

# import word_util as wtil
# import graph_util as gtil

# import torch
import spacy

nlp = spacy.load('en') # evecute line below first




tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = None
modelp = None

def load_model(device):
	global model, modelp
	model = BertModel.from_pretrained('bert-base-uncased')
	model.eval()
	model.to(device)
	
	modelp = BertForMaskedLM.from_pretrained('bert-base-uncased')
	modelp.eval()
	modelp.to(device)

def get_toks(s):
	return tokenizer.tokenize(s)

def eval_str(s, device='cpu'):
	
	toks = tokenizer.tokenize(s)
	
	if toks[0] != '[CLS]':
		toks.insert(0, '[CLS]')
	
	if toks[-1] != '[SEP]':
		toks.append('[SEP]')
	
	#     print(toks)
	
	segs = []
	idx = 0
	for t in toks:
		segs.append(idx)
		if t == '[SEG]':
			idx += 1
	
	tokens_tensor = torch.tensor([tokenizer.convert_tokens_to_ids(toks)]).to(device)
	segments_tensors = torch.tensor([segs]).to(device)
	
	with torch.no_grad():
		encoded_layers, _ = model(tokens_tensor, segments_tensors)
	# We have a hidden states for each of the 12 layers in model bert-base-uncased
	embs = encoded_layers[-1][0]
	
	return embs, toks



def format_question(question, device='cpu'):
	
	q, ans = question['question']['stem'], [c['text'] for c in question['question']['choices']]
	
	opts = []
	
	for a in ans:
		x = '[CLS] ' + q.replace('?', ' ? ').replace('.', ' . ') + ' [SEP] ' + a + ' [SEP]'
		
		toks = tokenizer.tokenize(x)
		#         print(toks)
		
		idx = toks.index('[SEP]')
		seg = [0] * (idx + 1) + [1] * (len(toks) - idx - 1)
		
		tokens_tensor = torch.tensor([tokenizer.convert_tokens_to_ids(toks)])
		segments_tensors = torch.tensor([seg])
		
		tokens_tensor = tokens_tensor.to(device)
		segments_tensors = segments_tensors.to(device)
		
		with torch.no_grad():
			encoded_layers, _ = model(tokens_tensor, segments_tensors)
		# We have a hidden states for each of the 12 layers in model bert-base-uncased
		embs = encoded_layers[-1][0][1:-1]
		
		# print(embs.shape)
		
		opts.append(embs[segments_tensors.byte().squeeze()[1:-1]])
	
	x = '[CLS] ' + q.replace('?', ' ? ').replace('.', ' . ') + ' [SEP]'
	toks = tokenizer.tokenize(x)
	
	seg = [0 ] *(len(toks))
	
	tokens_tensor = torch.tensor([tokenizer.convert_tokens_to_ids(toks)]).to(device)
	segments_tensors = torch.tensor([seg]).to(device)
	
	with torch.no_grad():
		encoded_layers, _ = model(tokens_tensor, segments_tensors)
	# We have a hidden states for each of the 12 layers in model bert-base-uncased
	embs = encoded_layers[-1][0][1:-1]
	
	return embs, opts

