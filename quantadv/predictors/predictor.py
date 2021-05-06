from allennlp.predictors.predictor import Predictor
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from pprint import pprint


@Predictor.register("predictor")
class Predictor(Predictor):
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        label_vocab = self._model.vocab.get_index_to_token_vocabulary("labels")

        # outputs["predicates"] = instance.fields["predicates"]
        print(outputs["logits"][1:], outputs["logits"][1:].argmax())
        print(label_vocab)
        outputs["predicted"] = label_vocab[outputs["logits"][1:].argmax() + 1]
        outputs["label"] = instance.fields["label"].label
        outputs["containment"] = instance.fields["containment"][0]
        outputs["p1_id"] = instance.fields["meta"]["p1_id"]
        outputs["p2_id"] = instance.fields["meta"]["p2_id"]
        outputs["tokens"] = " ".join([str(token) for token in instance.fields["tokens"].tokens])

        del outputs["logits"]
        del outputs["loss"]
        del outputs["probs"]
        del outputs["meta"]

        if outputs["label"] != "0":
            return sanitize(outputs)
