from typing import Dict, Optional

import torch
import torch.nn.functional as F
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, FeedForward, Seq2VecEncoder
from allennlp.nn import RegularizerApplicator
from allennlp.nn import util


@Model.register("tuple_predicate_embedder")
class TuplePredicateEmbedder(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 entity_embedder: TextFieldEmbedder,
                 entity_seq2vec: Seq2VecEncoder,
                 predicate_embedder: TextFieldEmbedder,
                 predicate_seq2vec: Seq2VecEncoder,
                 entity_output_layer: FeedForward,
                 predicate_output_layer: FeedForward,
                 regularizer: Optional[RegularizerApplicator] = None):
        super().__init__(vocab, regularizer)

        self._entity_embedder = entity_embedder
        self._entity_seq2vec = entity_seq2vec

        self._predicate_embedder = predicate_embedder
        self._predicate_seq2vec = predicate_seq2vec

        self._entity_output_layer = entity_output_layer
        self._predicate_output_layer = predicate_output_layer

    def forward(self,
                subject_tokens: Dict[str, torch.LongTensor],
                object_tokens: Dict[str, torch.LongTensor],
                predicate_tokens: Dict[str, torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        # Embed entities
        subject_embedding = self._entity_seq2vec(
            self._entity_embedder(subject_tokens),
            mask=util.get_text_field_mask(subject_tokens).float()
        )
        object_embedding = self._entity_seq2vec(
            self._entity_embedder(object_tokens),
            mask=util.get_text_field_mask(object_tokens).float()
        )

        # Concatenate the entity embeddings and forward pass through FeedForward
        entities_cat = torch.cat([subject_embedding, object_embedding], dim=1)
        out_embedding = self._entity_output_layer(entities_cat)

        # Calculate the loss and other metrics
        output_dict = {'embedding': out_embedding}
        if predicate_tokens:
            mask = util.get_text_field_mask(predicate_tokens).float()
            predicate_embedding = self._predicate_seq2vec(
                self._predicate_embedder(predicate_tokens),
                mask=mask
            )
            gold_embedding = self._predicate_output_layer(predicate_embedding)

            # Compute cosine loss between gold embedding and outputted embedding
            cosine_loss_label = torch.tensor([1], dtype=out_embedding.dtype, device=out_embedding.device)
            output_dict["loss"] = F.cosine_embedding_loss(out_embedding, gold_embedding, cosine_loss_label)

        return output_dict
