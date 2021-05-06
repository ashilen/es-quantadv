import json
from typing import Dict, Iterable

from allennlp.data import (
    DatasetReader,
    Instance,
)

from allennlp.data.tokenizers import Token
from allennlp.data.fields import (
    SpanField, TextField, LabelField, ListField,
    SequenceLabelField, MetadataField
)
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer


@DatasetReader.register("containment_to_quant_reader")
class UDSTimeReader(DatasetReader):
    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.token_indexers = {
            "tokens": SingleIdTokenIndexer(),
        }

    def _read(self, file_path: str) -> Iterable[Instance]:

        with open(file_path, "r") as file:
            for line in file:
                line = json.loads(line)

                quantifier = line["quantifier"]

                # hold out the quantifier from the tokens
                tokens = [
                    Token(token) if token is not quantifier else Token("")
                    for token in line["tokens"].split()]

                tokens_field = TextField(tokens, self.token_indexers)

                pred_1_field = SpanField(
                    line["pred_1_idx"], line["pred_1_idx"], tokens_field)
                pred_2_field = SpanField(
                    line["pred_2_idx"], line["pred_2_idx"], tokens_field)

                predicate_sequence_field = ListField(
                    [pred_1_field, pred_2_field])

                containment_field = SequenceLabelField(
                    [str(line["containment"]), str(line["containment"])],
                    predicate_sequence_field,
                    label_namespace="containment_labels")

                label_field = LabelField(quantifier)

                fields = {
                    "tokens": tokens_field,
                    "label": label_field,
                    "containment": containment_field,
                    "predicates": predicate_sequence_field,
                    "meta": MetadataField(line["meta"])
                }

                yield Instance(fields)


@DatasetReader.register("quant_to_containment_reader")
class QuantToContainmentReader(DatasetReader):
    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.token_indexers = {
            "tokens": SingleIdTokenIndexer(),
        }

    def _read(self, file_path: str) -> Iterable[Instance]:

        with open(file_path, "r") as file:
            for line in file:
                line = json.loads(line)

                quantifier = line["quantifier"]

                tokens = [Token(token) for token in line["tokens"].split()]

                tokens_field = TextField(tokens, self.token_indexers)

                pred_1_field = SpanField(
                    line["pred_1_idx"], line["pred_1_idx"], tokens_field)
                pred_2_field = SpanField(
                    line["pred_2_idx"], line["pred_2_idx"], tokens_field)

                predicate_sequence_field = ListField(
                    [pred_1_field, pred_2_field])

                quantifier_field = SequenceLabelField(
                    [quantifier, quantifier],
                    predicate_sequence_field,
                    label_namespace="quantifier_labels")

                label_field = LabelField(str(line["containment"]))

                fields = {
                    "tokens": tokens_field,
                    "label": label_field,
                    "quantifier": quantifier_field,
                    "predicates": predicate_sequence_field
                }

                yield Instance(fields)
