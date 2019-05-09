from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM
import torch
import math

bertMaskedLM = BertForMaskedLM.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def get_score(sentence):
    tokenize_input = tokenizer.tokenize(sentence)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    predictions = bertMaskedLM(tensor_input)
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(predictions.squeeze(), tensor_input.squeeze()).data
    return math.exp(loss)




# TODO: fine-tune BERT with aristo tuples
# TODO: Also, probably create multiple language models for sub, pred, obj
def print_aristo_scores():
    scores = []
    with open('data/aristo.txt', 'r') as f:
        for line in f:
            sentence = " ".join(line.strip().split('\t'))
            scores.append((sentence, get_score(sentence)))
    scores.sort(key=lambda x: x[1])
    print(scores)


def print_openbookqa_scores():
    scores = []
    with open('outputs/openbook_tuples_old.txt', 'r') as f:
        for line in f:
            sentence = " ".join(line.strip().split('\t')[1:])
            scores.append((sentence, get_score(sentence)))
    scores.sort(key=lambda x: x[1], reverse=True)
    print(*scores, sep='\n')


print_openbookqa_scores()
