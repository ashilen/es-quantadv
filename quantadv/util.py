
def time_id_to_gid(time_id):
    return time_id \
        .replace('en-ud', 'ewt') \
        .replace('.conllu ', '-')


def time_sent_id_to_uds_gid(time_id):
    # time_id looks like "en-ud-test.conllu 1257"
    return time_id_to_gid(time_id)


def time_pred_id_to_uds_pred_id(time_id):
    # time id looks like "en-ud-train.conllu 3735_14"
    gid = time_id_to_gid(time_id)
    base, num = gid.split("_")

    # time dataset has 0-indexed predicate offsets,
    # uds has 1-indexed
    num = int(num) + 1

    return "%s-semantics-pred-%d" % (base, num)


def uds_pred_id_to_syntax_id(pred_id):
    # 'ewt-train-287-semantics-pred-28', 'ewt-train-287-syntax-27')
    return pred_id.replace("semantics-pred", "syntax")
