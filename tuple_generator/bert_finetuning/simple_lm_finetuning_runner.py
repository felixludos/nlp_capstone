import sys
from pathlib import Path

from simple_lm_finetuning import main

data_dir = Path('./data')
aristo_path = data_dir / 'aristo.txt'

# Assemble the command into sys.argv
sys.argv = [
    "simple_lm_finetuning.py",  # command name, not used by main
    "--train_corpus", str(aristo_path),
    "--bert_model", "bert-base-uncased",
    "--do_lower_case",
    "--output_dir", "output/finetuned_lm",
    "--do_train"
]

main()
