import csv
from decomp import UDSCorpus
from pprint import pprint
from const import (
    SENT_1_COL, SENT_2_COL,
    SENT_1_TEXT, SENT_2_TEXT,
    DATA_READ_PATH, DATA_WRITE_PATH,
    RESTRICTORS, QUANTIFIERS,
    SENT_TEXT_QUANT, SENT_TEXT_REST
)

from util import to_gid

uds = UDSCorpus(split="train", version="2.0")

fieldnames = []


def annotate_uds_t():
    target_sents = []

    with open(DATA_READ_PATH) as fr, open(DATA_WRITE_PATH, "w+") as fw:
        reader = csv.DictReader(fr, delimiter="\t", quotechar='"')
        rows = list(reader)

        fieldnames = reader.fieldnames

        for row in rows:
            sent_1_gid = to_gid(row[SENT_1_COL])
            sent_2_gid = to_gid(row[SENT_2_COL])

            sent_1 = uds[sent_1_gid].sentence
            sent_2 = uds[sent_2_gid].sentence

            row[SENT_1_TEXT] = sent_1
            row[SENT_2_TEXT] = sent_2

            for sent_idx, sent in enumerate([sent_1, sent_2]):

                quant_count = 1
                quant_found = False
                for quant in QUANTIFIERS:
                    if quant in sent:
                        quant_count += 1
                        quant_found = True
                        header = SENT_TEXT_QUANT.format(sent_idx + 1, quant_count)
                        row[header] = quant
                        if header not in fieldnames:
                            fieldnames.append(header)

                rest_count = 1
                rest_found = False
                for rest in RESTRICTORS:
                    if rest in sent:
                        rest_count += 1
                        rest_found = True
                        header = SENT_TEXT_REST.format(sent_idx + 1, rest_count)
                        row[header] = rest
                        if header not in fieldnames:
                            fieldnames.append(header)

                if quant_found and rest_found:
                    target_sents.append(sent)

        fieldnames += [SENT_1_TEXT, SENT_2_TEXT]
        writer = csv.DictWriter(fw, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    pprint(target_sents)
    print("\n")
    print("Found %s sentences with both a quantifier and restrictive term." % len(target_sents))
