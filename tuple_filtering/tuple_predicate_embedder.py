from typing import Dict, Optional

import torch
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, FeedForward
from allennlp.nn import RegularizerApplicator


@Model.register("tuple_predicate_embedder")
class TuplePredicateEmbedder(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 regularizer: Optional[RegularizerApplicator],
                 entity_embedder: TextFieldEmbedder,
                 predicate_embedder: TextFieldEmbedder,
                 output_layer: FeedForward):
        # Ensure that output dim of predicate embedder is same as that of FeedForward
        if not predicate_embedder.get_output_dim() == output_layer.get_output_dim():
            raise RuntimeError("Predicate embedder dim must be equal to that of output layer")

        super().__init__(vocab, regularizer)
        self._entity_embedder = entity_embedder
        self._predicate_embedder = predicate_embedder
        self._output_layer = output_layer

    def forward(self,
                subject_tokens: Dict[str, torch.LongTensor],
                object_tokens: Dict[str, torch.LongTensor],
                predicate_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        # Embed entities
        subject_embedding = self._entity_embedder(subject_tokens)
        object_embedding = self._entity_embedder(object_tokens)

        # Concatenate the entity embeddings and forward pass through FeedForward
        entities_cat = torch.cat([subject_embedding, object_embedding], dim=0)
        out_embedding = self._output_layer(entities_cat)

        # Calculate the loss and other metrics
        output_dict = {'embedding': out_embedding}
        if predicate_tokens:
            predicate_embedding = self._predicate_embedder(predicate_tokens)
            loss = self.loss(out_embedding, predicate_embedding)
            for metric in self.metrics.values():
                metric(out_embedding, predicate_embedding)
            output_dict["loss"] = loss

        return output_dict
