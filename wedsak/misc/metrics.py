from sklearn.metrics import precision_recall_fscore_support


def evaluate_predictions(
    gold,
    pred,
    task_name: str,
    pos_label="1",
    BASE_COLS=["note_id", "start", "end", "label"],
):

    COLS = BASE_COLS + [task_name]

    y_pred = pred[COLS]
    y_true = gold[COLS]

    y_pred_gold = y_true.merge(
        y_pred, on=BASE_COLS, how="outer", suffixes=("_gold", "_pred")
    )
    gold_col = task_name + "_gold"
    pred_col = task_name + "_pred"

    support = sum(y_pred_gold[gold_col] == pos_label)
    n_instances = sum(y_pred_gold[gold_col].notna())

    y_pred_gold.fillna("0", inplace=True)

    if support == 0:
        return {
            "0": {
                "precision": 0.0,
                "recall": 0.0,
                "fscore": 0.0,
                "support": support,
            },
            "1": {
                "precision": 0.0,
                "recall": 0.0,
                "fscore": 0.0,
                "support": support,
            },
            "n_instances": n_instances,
        }
    else:
        precision, recall, fscore, support = precision_recall_fscore_support(
            y_true=y_pred_gold[gold_col],
            y_pred=y_pred_gold[pred_col],
            # pos_label=pos_label, # UserWarning: Note that pos_label (set to '1') is ignored when average != 'binary' (got None). You may use labels=[pos_label] to specify a single positive class.
            average=None,
        )
        precision = precision.tolist()
        recall = recall.tolist()
        fscore = fscore.tolist()
        support = support.tolist()

        metrics = {
            "0": dict(
                precision=precision[0],
                recall=recall[0],
                fscore=fscore[0],
                support=support[0],
            ),
            "1": dict(
                precision=precision[1],
                recall=recall[1],
                fscore=fscore[1],
                support=support[1],
            ),
            "n_instances": n_instances,
        }

        return metrics
