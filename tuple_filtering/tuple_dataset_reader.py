from typing import Dict, Iterable

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from overrides import overrides


@DatasetReader.register("tuple_reader")
class TupleDatasetReader(DatasetReader):
    def __init__(self,
                 lazy: bool,
                 tokenizer: Tokenizer,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer}

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, 'r') as data_file:
            for line in data_file:
                subject, predicate, obj = line.split('\t')
                yield self.text_to_instance(subject, predicate, obj)

    def text_to_instance(self, subject: str, predicate: str, obj: str) -> Instance:
        subject_tokens = self._tokenizer.tokenize(subject)
        predicate_tokens = self._tokenizer.tokenize(predicate)
        object_tokens = self._tokenizer.tokenize(obj)

        instance_dict = {
            'subject': TextField(subject_tokens, self._token_indexers),
            'predicate': TextField(predicate_tokens, self._token_indexers),
            'object': TextField(object_tokens, self._token_indexers)
        }

        return Instance(instance_dict)
