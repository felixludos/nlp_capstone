from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from qa_solver import word_util as wtil

torch.set_printoptions(linewidth=120)
np.set_printoptions(linewidth=120, suppress=True)

DATA_DIR = Path('../data')
QUESTIONS_DATA_DIR = DATA_DIR / 'questions'
BIG_DATA_DIR = Path('../data-nb')


def cosd(x, y):
    if x.ndimension() == 1:
        x = x.unsqueeze(0)
    if y.ndimension() == 1:
        y = y.unsqueeze(0)
    x = F.normalize(x, 2, -1)
    y = F.normalize(y, 2, -1)
    return -x @ y.transpose(-1, -2) / 2 + .5


def l2(x, y):
    if x.ndimension() == 1:
        x = x.unsqueeze(0)
    if y.ndimension() == 1:
        y = y.unsqueeze(0)
    x = x.unsqueeze(-2)
    y = y.unsqueeze(-3)
    return (x - y).pow(2).mean(-1)


def filter_tokens(s):
    s = s.lower()
    if s[-1] in {'.', '?'}:
        s = s[:-1]
    s = s.split(' ')
    return s


def top_k_question(query, full_bag, k=5):
    picks = wtil.tfidf(Counter(filter_tokens(query)), full_bag)[:k]
    return [w for w, s in picks]


def get_connections(picks, mentions):
    matches = set()
    for q in picks:
        matches.update(mentions[q])
    return matches


def get_closest(query, vecs, dist_func=l2, k=2):
    distances = dist_func(query, vecs)
    return torch.topk(distances, k, dim=-1, largest=False, sorted=False)


def convert(words, lang):
    # Old fasttext conversion:
    # return torch.from_numpy(np.stack([lang.get_word_vector(w) for w in words])).float()

    return torch.stack([lang[w] for w in words])


def solve(q, lang, vecs, elements, rows, table, full_bag, mentions, k=5):
    words = top_k_question(q['question']['stem'], full_bag, k=k)

    v = convert(words, lang)

    cls = get_closest(v, vecs)[1]

    conns = get_connections(elements[cls].reshape(-1), mentions)

    wopts = set()
    for i in conns:
        wopts.update(rows[i])
    wopts = list(wopts)

    opts = torch.from_numpy(np.stack([table[w] for w in wopts])).float()

    labels = []
    for a in q['question']['choices']:
        lbl = a['label']
        v = convert(top_k_question(a['text'], full_bag), lang).view(-1, 300)
        nb = get_closest(v, opts, k=10)[0]
        conf = 1 / nb.mean()
        labels.append((lbl, conf))

    sol = sorted(labels, key=lambda x: x[1])[-1][0]
    return sol


def load_data(ds_name: str) -> tuple:
    if ds_name == 'elem':
        root = QUESTIONS_DATA_DIR / 'AI2-Elementary-NDMC-Feb2016-Train.jsonl'
        lookup = BIG_DATA_DIR / 'train_elem_tokens_emb.pth.tar'
    elif ds_name == '8th':
        root = QUESTIONS_DATA_DIR / 'AI2-8thGr-NDMC-Feb2016-Train.jsonl'
        lookup = BIG_DATA_DIR / 'train_8thgr_tokens_emb.pth.tar'
    else:
        raise Exception('unknown dataset')

    return root, lookup


def load_kb(kb_dir: str):
    # Load in KB data
    table = torch.load(kb_dir)
    rows = table['rows']
    elements = np.array(table['elements'])
    vecs = torch.from_numpy(table['vecs']).float()
    table = dict(zip(elements, vecs))

    return vecs, rows, elements, table


def evaluate(ds_name: str, kb_dir: str):
    # Load data
    root, lookup_dir = load_data(ds_name)
    questions = wtil.load_questions(root)
    lookup = torch.load(lookup_dir)
    full_bag = lookup['bag']

    # Load language in fasttext-like format
    lang = dict(zip(lookup['words'], lookup['vecs']))

    vecs, rows, elements, table = load_kb(kb_dir)

    mentions = {}
    for i, row in enumerate(rows):
        for w in row:
            if w not in mentions:
                mentions[w] = []
            mentions[w].append(i)

    correct_answers = [q['answerKey'] for q in questions]
    predicted_answers = []
    num_correct = 0
    for i, q in tqdm(enumerate(questions), total=len(questions)):
        sol = solve(q, lang, vecs, elements, rows, table, full_bag, mentions)
        if sol == correct_answers[i]:
            num_correct += 1
        predicted_answers.append(sol)

        if i % 10 == 0:
            tqdm.write(f'{i + 1}/{len(questions)} {num_correct / (i + 1):.4f}')

    # Print statistics
    tqdm.write('Done {:.4f}'.format(num_correct / len(questions)))
    tqdm.write('True: ' + ''.join(correct_answers))
    tqdm.write('Solutions: ' + ''.join(predicted_answers))


# Example call:
# evaluate(ds_name='elem', kb_dir=BIG_DATA_DIR / 'fast_table.pth.tar')
