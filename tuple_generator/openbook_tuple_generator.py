from tuple_generator import open_ie_api

input_path = 'data/openbook/openbook_corpus.txt'
output_path = 'data/openbook_tuples.tsv'

with open(input_path, 'r') as f:
    facts = [line.strip()[1:-1] for line in f.readlines()]

tuples = open_ie_api.call_api_many(facts, verbose=True)
tuples = ("\t".join(part.strip() for part in t) for t in tuples)
with open(output_path, 'w') as f:
    print(*tuples, sep='\n', file=f)
