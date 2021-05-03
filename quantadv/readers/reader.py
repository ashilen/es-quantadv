import json
from typing import Dict, Iterable

from allennlp.data import (
    DatasetReader,
    Instance,
)

from allennlp.data.tokenizers import Token
from allennlp.data.fields import IndexField, TextField, LabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer


@DatasetReader.register('uds_time_reader')
class Reader(DatasetReader):
    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.token_indexers = {'tokens': SingleIdTokenIndexer()}

    def _read(self, file_path: str) -> Iterable[Instance]:

        with open(file_path, 'r') as file:
            for line in file:
                line = json.loads(line)

                quantifier = line["quantifier"]

                # hold out the quantifier from the tokens
                tokens = [
                    Token(token)  # if token is not quantifier else Token("")
                    for token in line["tokens"].split()]

                tokens_field = TextField(tokens, self.token_indexers)

                pred_1_field = IndexField(line["pred_1_idx"], tokens_field)
                pred_2_field = IndexField(line["pred_2_idx"], tokens_field)

                containment_field = LabelField(
                    str(line["containment"]),
                    # label_namespace="labels",
                    skip_indexing=True
                )

                label_field = LabelField(quantifier)

                fields = {
                    "tokens": tokens_field,
                    "label": label_field,
                    "containment": containment_field,
                    "pred_1": pred_1_field,
                    "pred_2": pred_2_field
                }

                yield Instance(fields)
