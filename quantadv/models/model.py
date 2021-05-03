
from typing import Dict

import torch

from allennlp.data import (
    Vocabulary,
    TextFieldTensors,
)
from allennlp.nn import util
from allennlp.models import Model
from allennlp.modules import Seq2VecEncoder, FeedForward, Embedding
from allennlp.modules.span_extractors import EndpointSpanExtractor
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from allennlp.modules.text_field_embedders import TextFieldEmbedder


@Model.register('simple_classifier')
class SimpleClassifierModel(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 containment_feature_dim: int,
                 encoder: Seq2VecEncoder):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        self.num_labels = vocab.get_vocab_size("labels")
        print(self.num_labels)
        self.classifier = torch.nn.Linear(
            encoder.get_output_dim(), self.num_labels)

        print("=====>", vocab.get_vocab_size("tokens"))
        num_containment_labels = vocab.get_vocab_size("containment_labels")
        self.containment_embedding = Embedding(
            num_containment_labels, num_containment_labels)

        self.span_extractor = EndpointSpanExtractor(input_dim=50, combination="x,y")

        self.label_accuracy = CategoricalAccuracy()
        self.label_f1_metrics = {}
        for i in range(self.num_labels):
            f1 = F1Measure(positive_label=i)
            self.label_f1_metrics[
                vocab.get_token_from_index(index=i, namespace="labels")] = f1

    def forward(self,
                tokens: TextFieldTensors,
                predicates: torch.LongTensor,
                containment: torch.LongTensor,
                label: torch.Tensor) -> Dict[str, torch.Tensor]:

        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(tokens)

        # Shape: (batch_size, num_tokens)
        # mask = util.get_text_field_mask(tokens)

        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_predicates = self.span_extractor(embedded_text, predicates)

        predicates_mask = (predicates[:, :, 0] >= 0).squeeze(-1)

        # Shape: (batch_size, encoding_dim)
        encoded_predicates = self.encoder(embedded_predicates, predicates_mask)
        # Shape: (batch_size, num_labels)
        logits = self.classifier(encoded_predicates)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits)
        # Shape: (1,)
        loss = torch.nn.functional.cross_entropy(logits, label)

        self.label_accuracy(logits, label)

        # compute F1 per label
        for i in range(self.num_labels):
            metric = self.label_f1_metrics[
                self.vocab.get_token_from_index(index=i, namespace="labels")]
            metric(probs, label)

        self.label_accuracy(logits, label)

        return {'loss': loss, 'probs': probs, 'logits': logits}

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metric_dict = {}

        sum_f1 = 0.0
        for name, metric in self.label_f1_metrics.items():
            metric_val = metric.get_metric(reset)
            metric_dict[name + '_P'] = metric_val["precision"]
            metric_dict[name + '_R'] = metric_val["recall"]
            metric_dict[name + '_F1'] = metric_val["f1"]
            sum_f1 += metric_val["f1"]

        names = list(self.label_f1_metrics.keys())
        total_len = len(names)
        average_f1 = sum_f1 / total_len
        metric_dict['average_F1'] = average_f1
        metric_dict['accuracy'] = self.label_accuracy.get_metric(reset)
        return metric_dict
