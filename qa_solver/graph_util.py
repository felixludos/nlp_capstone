
import sys, os, time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
import util
# import nltk



PATH = '../../../Downloads/aristo-tuple-kb-v5-mar2017/aristo-tuple-kb-v5-mar2017/aristo.txt'
PATH2 = '../../../Downloads/aristo-tuple-kb-v5-mar2017/aristo.txt'

G = None
iG = None

def load(path=None):
	if path is None:
		path = PATH if os.path.isfile(PATH) else PATH2
		
	global G, iG

	G = util.adict()
	iG = util.adict()
	
	with open(path, 'r') as tsvfile:
		reader = csv.reader(tsvfile, delimiter='\t')
		for i, row in tqdm(enumerate(reader)):
			sub, rel, obj = row
			if sub not in G:
				G[sub] = util.adict()
			if rel not in G[sub]:
				G[sub][rel] = set()
			G[sub][rel].add(obj)
			if obj not in iG:
				iG[obj] = util.adict()
			if rel not in iG[obj]:
				iG[obj][rel] = set()
			iG[obj][rel].add(sub)


def _search(key, graph, tokens, threshold, fuel, root=[], collected=set()):
	if fuel == 0 or key not in graph:
		root.append(key)
		overlap = len(tokens.intersection(root))
		if overlap >= threshold:
			collected.add('\t'.join(root))
	else:
		
		fuel -= 1
		
		root.append(key)
		for rel in graph[key]:
			rroot = root.copy()
			rroot.append(rel)
			for obj in graph[key][rel]:
				if fuel > 0:
					_search(obj, graph, tokens, threshold, 0, rroot.copy(), collected)
				_search(obj, graph, tokens, threshold, fuel, rroot.copy(), collected)
	
	return collected


def query(tokens, level=2, threshold=2, limit=50):
	if G is None:
		load()
	
	out = []
	for sub in G:
		if sub in tokens:
			
			out.extend(_search(sub, G, tokens, threshold, level, root=[], collected=set()))
			if len(out) > limit:
				return out
	return out








