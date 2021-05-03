import os

DATA_DIR = "data"
DATA_READ_FILE = "time_eng_ud_v1.2_2015_10_30.tsv"
DATA_WRITE_FILE = "quant_time_eng_ud_v1.2_2015_10_30.tsv"
DATA_READ_PATH = os.path.join(DATA_DIR, DATA_READ_FILE)
DATA_WRITE_PATH = os.path.join(DATA_DIR, DATA_WRITE_FILE)

TOTAL_CONLLU_READ_PATH = os.path.join(DATA_DIR, "conllu", "total.conllu")

SENT_1_COL = "Sentence1.ID"
SENT_2_COL = "Sentence2.ID"

PRED_1_COL = "Event1.ID"
PRED_2_COL = "Event2.ID"

SENT_1_TEXT = "Sentence1.TEXT"
SENT_2_TEXT = "Sentence2.TEXT"

SENT_TEXT_QUANT = "Sentence{}.QUANT.{}"
SENT_TEXT_REST = "Sentence{}.REST.{}"

QUANTIFIERS = [
    "always",
    "invariably",
    "universally",
    "without exception",  # not gonna work!

    "never",

    "sometimes",
    "occasionally",

    "usually",
    "mostly",
    "generally",
    "almost always",  # not gonna work!
    "with few exceptions",  # not gonna work!

    "all the time",  # not gonna work!

    "often",
    "frequently",
    "commonly",

    "seldom",
    "infrequently",
    "rarely"
    "almost never",  # not gonna work!

    "whenever"]

RESTRICTORS = [
    "if",
    "when"
]

DEV = "dev"
TRAIN = "train"
TEST = "test"
SPLIT = "Split"

QUANTIFIER = "quantifier"
PRED_1_IDX = "pred_1_idx"
PRED_2_IDX = "pred_2_idx"
CONTAINMENT = "containment"
TOKENS = "tokens"
