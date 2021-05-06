import argparse
from pprint import pprint
from quantadv.const import DATA_DIR

from allennlp.data import (
    DataLoader,
    Vocabulary,
)
from allennlp.models import Model
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.training.trainer import GradientDescentTrainer, Trainer
from allennlp.training.optimizers import AdamOptimizer
from allennlp.training.util import evaluate

from quantadv.readers.reader import UDSTimeReader
from quantadv.models.model import SimpleClassifierModel


def ModelFactory(vocab: Vocabulary) -> Model:
    print("Building the model")
    vocab_size = vocab.get_vocab_size("tokens")
    embedder = BasicTextFieldEmbedder(
        {"tokens": Embedding(embedding_dim=10, num_embeddings=vocab_size)}
    )
    encoder = BagOfEmbeddingsEncoder(embedding_dim=10)
    return SimpleClassifierModel(vocab, embedder, encoder)


def TrainerFactory(
    model: Model,
    serialization_dir: str,
    train_loader: DataLoader,
    dev_loader: DataLoader,
) -> Trainer:
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer = AdamOptimizer(parameters)  # type: ignore
    trainer = GradientDescentTrainer(
        model=model,
        serialization_dir=serialization_dir,
        data_loader=train_loader,
        validation_data_loader=dev_loader,
        num_epochs=5,
        optimizer=optimizer,
    )
    return trainer


def data_files():

    return map(lambda x: x % DATA_DIR, [
        "%s/train",
        "%s/dev",
        "%s/test",
        "%s/serialize",
    ])


def train_and_test_model():
    (
        train_file, dev_file,
        test_file, serialization_dir
    ) = data_files()

    reader = UDSTimeReader()

    train_data = list(reader.read(train_file))
    dev_data = list(reader.read(dev_file))

    vocab = Vocabulary.from_instances(train_data + dev_data)
    model = ModelFactory(vocab)

    train_loader = SimpleDataLoader(train_data, 8, shuffle=True)
    dev_loader = SimpleDataLoader(dev_data, 8, shuffle=False)

    train_loader.index_with(vocab)
    dev_loader.index_with(vocab)

    trainer = TrainerFactory(model, serialization_dir, train_loader, dev_loader)
    print("Starting training")
    trainer.train()
    print("Finished training")

    # Now we can evaluate the model on a new dataset.
    test_data = list(reader.read(test_file))
    data_loader = SimpleDataLoader(test_data, batch_size=8)
    data_loader.index_with(model.vocab)

    results = evaluate(model, data_loader)
    pprint(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Model the temporal structure of labels.")
    parser.add_argument("--temp2quant", action="store_true")
    parser.add_argument("--quant2temp", action="store_true")
    args = parser.parse_args()

    train_and_test_model()
