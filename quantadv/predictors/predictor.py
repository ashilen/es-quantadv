from allennlp.predictors.predictor import Predictor
from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from pprint import pprint


@Predictor.register('predictor')
class Predictor(Predictor):
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        label_vocab = self._model.vocab.get_index_to_token_vocabulary('labels')

        pprint(dir(instance.fields['pred_1']))

        outputs['tokens'] = " ".join([str(token) for token in instance.fields['tokens'].tokens])
        outputs['containment'] = instance.fields['containment'].label
        outputs['predicted'] = label_vocab[outputs['logits'].argmax()]
        outputs['label'] = instance.fields['label'].label
        outputs['pred1'] = instance.fields['pred_1'].human_readable_repr()
        outputs['pred2'] = instance.fields['pred_2'].human_readable_repr()

        del outputs['logits']
        del outputs['loss']
        del outputs['probs']

        if outputs['label'] != "0" and outputs['predicted'] != "0":
            return sanitize(outputs)
