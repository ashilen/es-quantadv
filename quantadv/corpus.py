import csv
import argparse
import conllu
import json
import os
from decomp import UDSCorpus
from collections import defaultdict
from functools import reduce

from quantadv.const import (
    DATA_READ_PATH, TOTAL_CONLLU_READ_PATH,
    DEV, TRAIN, TEST, SPLIT, DATA_DIR,
    PRED_1_IDX, PRED_2_IDX, TOKENS,
    CONTAINMENT, QUANTIFIER,
)

from quantadv.util import (
    time_sent_id_to_uds_gid,
    time_pred_id_to_uds_pred_id,
    uds_pred_id_to_syntax_id
)

SENT_1_COL = "Sentence1.ID"
SENT_2_COL = "Sentence2.ID"

PRED_1_COL = "Event1.ID"
PRED_2_COL = "Event2.ID"

PRED_1_DUR = "Pred1.Duration"
PRED_2_DUR = "Pred2.Duration"

PRED_1_BEG = "Pred1.Beg"
PRED_1_END = "Pred1.End"
PRED_2_BEG = "Pred2.Beg"
PRED_2_END = "Pred2.End"

QUANTIFIERS = [
    "always",
    "invariably",
    "universally",
    # "without exception",  # not gonna work!

    "never",

    "sometimes",
    "occasionally",

    "usually",
    "mostly",
    "generally",
    # "almost always",  # not gonna work!

    # "all the time",  # not gonna work!

    "often",
    "frequently",
    "commonly",

    "seldom",
    "infrequently",
    "rarely"
    # "almost never",  # not gonna work!

    "whenever"]

RESTRICTORS = [
    "if",
    "when"
]


class Relation:
    def __init__(self, syntax_id, features, edge):
        self.syntax_id = syntax_id
        self.features = features
        self.edge = edge

    @property
    def lemma(self):
        return self.features["lemma"]


class Predicate:
    def __init__(self, graph, id):
        self._graph = graph
        self._id = id
        self._quantifier_nodes = self.get_quantifier_nodes()
        self._restrictor_nodes = self.get_restrictor_nodes()
        self._beg = None
        self._end = None

    def get_x_nodes(self, x_list):
        x_nodes = [
            (nid, node) for nid, node in self._graph.syntax_nodes.items()
            if node["lemma"].lower() in x_list
        ]

        syntax_edges = self._graph.syntax_edges()

        nodes = []

        for nid, node in x_nodes:
            syntax_edge = (self.syntax_id, nid)  # pred -> x_node
            if syntax_edge in syntax_edges:
                nodes.append(
                    Relation(
                        nid, node,
                        syntax_edges[syntax_edge])
                )

        return nodes

    def get_quantifier_nodes(self):
        return self.get_x_nodes(QUANTIFIERS)

    def get_restrictor_nodes(self):
        return self.get_x_nodes(RESTRICTORS)

    @property
    def graph(self):
        return self._graph

    @property
    def syntax_id(self):
        return uds_pred_id_to_syntax_id(self._id)

    @property
    def quantifiers(self):
        return [q.lemma for q in self._quantifier_nodes]

    @property
    def restrictors(self):
        return self._restrictor_nodes

    @property
    def id(self):
        return self._id

    @property
    def idx(self):
        return int(self.id.split("-")[-1])

    @property
    def sentence(self):
        return self._graph.sentence

    @property
    def sentence_id(self):
        return self._id.split("-semantics")[0]

    @property
    def lemma(self):
        return self._graph.syntax_nodes[self.syntax_id]["lemma"]

    @property
    def beg(self):
        return self._beg

    @beg.setter
    def beg(self, value):
        self._beg = value

    @property
    def end(self):
        return self._end

    @end.setter
    def end(self, value):
        self._end = value

    @property
    def interval(self):
        return range(self.beg, self.end)


class PredicatePair:
    def __init__(self, p1, p2, split):
        self.p1 = p1
        self.p2 = p2
        self.split = split

    def __iter__(self):
        return iter((self.p1, self.p2))

    @property
    def id(self):
        return "%s-%s" % (self.p1.id, self.p2.id)

    @property
    def is_quantified(self):
        return bool(self.p1.quantifiers or self.p2.quantifiers)

    @property
    def is_restricted(self):
        return bool(self.p1.restrictors or self.p2.restrictors)

    @property
    def is_complete(self):
        return self.is_quantified and self.is_restricted

    @property
    def is_same_sent(self):
        return self.p1.sentence_id == self.p2.sentence_id

    @property
    def quantifiers(self):
        return self.p1.quantifiers + self.p2.quantifiers

    @property
    def percent_timeline_containment(self):
        containment = set(self.p1.interval) & set(self.p2.interval)
        return len(containment) / 100

    @property
    def containment(self):
        p1, p2 = set(self.p1.interval), set(self.p2.interval)

        containment = p1 & p2
        max_ = max(len(p1), len(p2))

        if max_ == 0:
            return 0

        scale = 100 / max_

        return (len(containment) * scale) / 100

    def format(self):
        sing_tmpl = """
{gid}
{sent}
    Predicate: {pred}
    Quantifiers: {quants}
    Restrictors: {restrictors}
    Interval: {beg} - {end}
"""

        pair_tmpl = """
{}
    +
{}
        containment: {containment}

"""

        pair = [sing_tmpl.format(
            gid=p.id, sent=p.sentence, quants=" ".join(p.quantifiers),
            restrictors=" ".join([x.lemma for x in p.restrictors]), beg=p.beg, end=p.end,
            pred=p.lemma
        ) for p in self]

        return pair_tmpl.format(*pair, containment=self.containment)


class PredicatePairAverage(PredicatePair):
    """Holding structure for sets of identical predicate pairs
    annotated by different annotators. Defers all properties to an
    arbitrary underlying PP, except for `containment`, which is computed
    as the average of the underlying PPs' containment values.
    """
    def __init__(self, *predicate_pairs):
        self.predicate_pairs = [*predicate_pairs]
        self.exemplar = predicate_pairs[-1]

    def __getattr__(self, name):
        return getattr(self.exemplar, name)

    def push(self, predicate_pair):
        self.predicate_pairs.append(predicate_pair)

    @property
    def containment(self):
        return (
            sum([pp.containment for pp in self.predicate_pairs])
            / len(self.predicate_pairs)
        )


class Corpus:
    def __init__(self, predicate_pairs, same_sent=False, diff_sent=False):
        self._predicate_pairs = predicate_pairs

        if same_sent:
            self._predicate_pairs = filter(
                lambda pp: pp.is_same_sent, self._predicate_pairs)
        elif diff_sent:
            self._predicate_pairs = filter(
                lambda pp: not pp.is_same_sent, self._predicate_pairs)
        elif diff_sent and same_sent:
            raise Exception("Tomfoolery!")

        self.by_quant = defaultdict(list)
        for pp in self.predicate_pairs:
            for quant in (pp.quantifiers or [None]):
                self.by_quant[quant].append(pp)

    def __iter__(self):
        return iter(self.predicate_pairs)

    @property
    def predicate_pairs(self):
        # collapse annotations of identical pairs
        def collapse(acc, pp):
            if pp.id in acc and pp.split == acc[pp.id].split:
                acc[pp.id].push(pp)
            else:
                acc[pp.id] = PredicatePairAverage(pp)
            return acc

        return reduce(collapse, self._predicate_pairs, {}).values()

    @property
    def avgs(self, measure=lambda pp: pp.containment):
        _avgs = []

        for quant, pp_list in self.by_quant.items():
            sum_containment = sum([measure(pp) for pp in pp_list])
            mean_containment = sum_containment / len(pp_list)

            _avgs.append((quant, mean_containment))

        _avgs.sort(key=lambda pair: pair[1])

        return _avgs

    def format_avgs(self):
        tmpl = "{quant}: {avg}"

        return "\n".join([tmpl.format(
            quant=quant, avg=avg
        ) for quant, avg in self.avgs])

    def format_by_quant(self, quant):
        return "\n".join([
            pp.format() for pp in self
            if quant in pp.quantifiers
        ])

    def to_conllu(self, conllu_corpus, quant=None):
        conllu_by_id = {doc.metadata["sent_id"]: doc for doc in conllu_corpus}

        filt = (lambda pp: quant in pp.quantifiers) if quant else lambda _: _

        for pp in filter(filt, self.predicate_pairs):
            if pp.is_quantified:
                p1_sent_id = pp.p1.graph.sentence_id
                p2_sent_id = pp.p2.graph.sentence_id
                yield conllu_by_id[p1_sent_id].serialize()
                if not pp.is_same_sent:
                    yield conllu_by_id[p2_sent_id].serialize()

    def split(self, split):
        return [pp for pp in self if pp.split == split]

    def format_training_data(self):
        for split in [DEV, TRAIN, TEST]:
            file = os.path.join(DATA_DIR, split)

            with open(file, "w+") as f:
                for pp in self.split(split):
                    for quant in pp.quantifiers or ["0"]:
                        instance = {
                            # collapse two sentences into a single token vector
                            TOKENS: pp.p1.sentence if pp.is_same_sent else pp.p1.sentence + " " + pp.p2.sentence,
                            QUANTIFIER: quant,
                            CONTAINMENT: int(pp.containment * 100),
                            PRED_1_IDX: pp.p1.idx,
                            PRED_2_IDX: pp.p2.idx if pp.is_same_sent else len(pp.p1.sentence.split()) + pp.p2.idx
                        }

                        f.write(json.dumps(instance) + "\n")


def serialize_predicate_pairs():
    def normalize_times(times):
        times = [int(t.strip()) for t in times]
        min_ = min(times)
        max_ = max(times)

        def _norm(d):
            return 0 if max_ == 0 else int(round((d - min_) / max_, 2) * 100)

        return [_norm(t) for t in times]

    def serialize_row(uds, row):
        predicates = []
        for sent_column, pred_column in [
            (SENT_1_COL, PRED_1_COL),
            (SENT_2_COL, PRED_2_COL)
        ]:
            sent_cell = row[sent_column]
            sent_gid = time_sent_id_to_uds_gid(sent_cell)
            sent_graph = uds[sent_gid]

            pred_cell = row[pred_column]
            pred_id = time_pred_id_to_uds_pred_id(pred_cell)

            predicate = Predicate(sent_graph, pred_id)
            predicates.append(predicate)

        p1, p2 = predicates

        p1.beg, p1.end, p2.beg, p2.end = normalize_times([
            row[PRED_1_BEG], row[PRED_2_END],
            row[PRED_2_BEG], row[PRED_2_END]
        ])

        return [p1, p2]

    uds = UDSCorpus(version="2.0")
    with open(DATA_READ_PATH) as fr:
        reader = csv.DictReader(fr, delimiter="\t", quotechar='"')
        rows = list(reader)

        return [
            PredicatePair(*serialize_row(uds, row), split=row[SPLIT])
            for row in rows
        ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the UDS.")
    parser.add_argument("-avgs", action="store_true",
                        help="Print average containment by quantifier.")
    parser.add_argument("-to-conllu", action="store_true")
    parser.add_argument("-fmt-training-data", action="store_true",
                        help="Format splits for training")
    parser.add_argument("--same-sent", action="store_true",
                        help="Restrict predicate pairs to those in same sentence.")
    parser.add_argument("--diff-sent", action="store_true",
                        help="Restrict predicate pairs to those in same sentence.")
    parser.add_argument("--sent-id", type=str,
                        help="Sentence to look at.")
    parser.add_argument("--pred-id", type=str,
                        help="Predicate to look at.")
    parser.add_argument("--quantifier", type=str,
                        choices=QUANTIFIERS,
                        help="Quantifier to look for.")
    parser.add_argument("--restrictor", type=str,
                        choices=RESTRICTORS,
                        help="Restrictor to look for.")

    args = parser.parse_args()

    predicate_pairs = serialize_predicate_pairs()

    corpus = Corpus(
        predicate_pairs,
        same_sent=args.same_sent, diff_sent=args.diff_sent)

    if args.avgs:
        print(corpus.format_avgs())

    elif args.fmt_training_data:
        corpus.format_training_data()

    elif args.to_conllu:
        f = open(TOTAL_CONLLU_READ_PATH)
        conllu_corpus = conllu.parse(f.read())

        for example in corpus.to_conllu(conllu_corpus, quant=args.quantifier):
            print(example)

    elif args.quantifier:
        print(corpus.format_by_quant(args.quantifier))

    else:
        count = 0
        for pp in corpus.predicate_pairs:
            if pp.is_quantified:
                count += 1
                print(pp.format())

        print("%d complete pairs" % count)
