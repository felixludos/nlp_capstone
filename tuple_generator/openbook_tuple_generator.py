import open_ie_api

openbook_path = 'data/openbook.txt'
with open(openbook_path, 'r') as f:
    facts = [line.strip()[1:-1] for line in f.readlines()]

tuples = open_ie_api.call_api_many(facts, verbose=True)
with open("data/openbook_tuples.tsv", 'w') as f:
    print(*["\t".join(t) for t in tuples], sep='\n', file=f)
