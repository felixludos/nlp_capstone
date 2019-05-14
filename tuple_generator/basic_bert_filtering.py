import open_ie_api
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM
from pathlib import Path
import torch
import math
from tqdm import tqdm

openbook_data_path = 'data/openbook.txt'

bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def get_score(sentence: str):
    tokenize_input = bert_tokenizer.tokenize(sentence)
    tensor_input = torch.tensor([bert_tokenizer.convert_tokens_to_ids(tokenize_input)])
    predictions = bert_model(tensor_input)
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(predictions.squeeze(), tensor_input.squeeze()).data
    return math.exp(loss)


def print_aristo_scores():
    scores = []
    with open('data/aristo.txt', 'r') as f:
        for line in tqdm(f, total=282594):
            sentence = " ".join(line.strip().split('\t'))
            scores.append((sentence, get_score(sentence)))
    scores.sort(key=lambda x: x[1])
    print(scores)


# TODO: use this api to get tuples from openbook.txt
# tuples = open_ie_api.call_api_single(openbook_data_path)
def print_openbookqa_scores():
    scores = []
    with open('outputs/openbook_tuples_old.txt', 'r') as f:
        for line in f:
            sentence = " ".join(line.strip().split('\t')[1:])
            scores.append((sentence, get_score(sentence)))
    scores.sort(key=lambda x: x[1], reverse=True)
    print(*scores, sep='\n')


print_aristo_scores()
