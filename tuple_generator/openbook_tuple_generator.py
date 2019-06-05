import argparse

from tuple_generator import open_ie_api

parser = argparse.ArgumentParser(description='Generate tuples using OpenIE.')
parser.add_argument('input_path', type=str)
parser.add_argument('output_path', type=str)
args = parser.parse_args()

with open(args.input_path, 'r') as f:
    facts = [line.strip()[1:-1] for line in f.readlines()]

tuples = open_ie_api.call_api_many(facts, verbose=True)
tuples = ("\t".join(part.strip() for part in t) for t in tuples)
with open(args.output_path, 'w') as f:
    print(*tuples, sep='\n', file=f)
