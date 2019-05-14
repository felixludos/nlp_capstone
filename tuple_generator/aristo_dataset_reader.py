from typing import Dict, List, Sequence, Iterable
import itertools
import logging

from overrides import overrides

from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers.dataset_utils import to_bioul
from allennlp.data.fields import TextField, SequenceLabelField, Field, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers import Tokenizer, WordTokenizer


@DatasetReader.register("aristo")
class AristoDatasetReader(DatasetReader):
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
        # FIXME: create different embedding spaces for subject, predicate and object
        subject_tokens = self._tokenizer.tokenize(subject)
        predicate_tokens = [Token(predicate)]
        object_tokens = self._tokenizer.tokenize(obj)

        instance_dict = {
            'subject': TextField(subject_tokens, self._token_indexers),
            'predicate': TextField(predicate_tokens, self._token_indexers),
            'object': TextField(object_tokens, self._token_indexers)
        }

        return Instance(instance_dict)
