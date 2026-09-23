"""All threshold selection is validation-only; metrics respect unknown labels."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from .hierarchy import label_state, close_topics


def targets(rows, schema):
    states = [label_state(r["targets"], schema) for r in rows]
    return np.asarray([s[0] for s in states], dtype=np.float32), np.asarray([s[1] for s in states], dtype=bool)


def counts(y, pred, mask):
    y, pred = y.astype(bool), pred.astype(bool)
    return ((y & pred & mask).sum(0), (~y & pred & mask).sum(0), (y & ~pred & mask).sum(0))


def scores(y, pred, mask):
    tp, fp, fn = counts(y, pred, mask)
    div = lambda a, b: np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)
    precision, recall = div(tp, tp+fp), div(tp, tp+fn)
    f1 = div(2*tp, 2*tp+fp+fn)
    supported = (y.astype(bool) & mask).sum(0) > 0
    micro_p = float(tp.sum() / max(1, (tp+fp).sum()))
    micro_r = float(tp.sum() / max(1, (tp+fn).sum()))
    return {"micro_precision": micro_p, "micro_recall": micro_r,
            "micro_f1": float(2*tp.sum()/max(1, (2*tp+fp+fn).sum())),
            "macro_f1_all_labels": float(f1.mean()) if len(f1) else 0.,
            "macro_f1_supported_labels": float(f1[supported].mean()) if supported.any() else 0.,
            "macro_precision": float(precision.mean()) if len(precision) else 0.,
            "macro_recall": float(recall.mean()) if len(recall) else 0.,
            "known_pairs": int(mask.sum()), "unknown_pairs": int((~mask).sum())}


def tune_threshold(rows, schema, probabilities, split):
    if split != "validation":
        raise ValueError("Threshold fitting is allowed only on validation")
    y, mask = targets(rows, schema)
    best = (-1., .5)
    # Global threshold: per-topic estimates would overfit rare validation labels.
    for t in np.linspace(.05, .95, 19):
        f = scores(y, probabilities >= t, mask)["micro_f1"]
        if f > best[0]:
            best = (f, float(t))
    return best[1]


def report(rows, schema, subjects_pred, probabilities, threshold, train_support):
    y, mask = targets(rows, schema)
    pred = probabilities >= threshold
    actual = [schema["subjects"].index(r["subject"]) for r in rows]
    result = {"n": len(rows), "threshold": threshold,
              "subject_accuracy": float(accuracy_score(actual, subjects_pred)),
              "subject_macro_f1": float(f1_score(actual, subjects_pred, labels=list(range(len(schema["subjects"]))), average="macro", zero_division=0)),
              "topics_masked": scores(y, pred, mask)}
    # Roots omitted from hierarchical metric to avoid inflating it by subject accuracy.
    hlabels = sorted(set(schema["nodes"])-set(schema["subjects"]), key=int)
    hstates = [label_state(r["targets"], schema, hlabels, hierarchical=True) for r in rows]
    hy = np.asarray([s[0] for s in hstates], dtype=bool)
    hm = np.asarray([s[1] for s in hstates], dtype=bool)
    hp = np.zeros_like(hy)
    hindex = {t:i for i,t in enumerate(hlabels)}
    for i, row in enumerate(pred):
        for t in close_topics([schema["labels"][j] for j in np.flatnonzero(row)], schema):
            if t in hindex:
                hp[i,hindex[t]] = True
    result["hierarchical_masked_without_roots"] = scores(hy, hp, hm)
    partial = np.asarray([not all(label_state(r["targets"], schema)[1]) for r in rows])
    # Partial here means unknown pairs, including ancestors. Explicit intermediate below.
    intermediate = np.asarray([any(t in schema["nodes"][n]["path"][:-1] for t in r["targets"] for n in schema["nodes"]) for r in rows])
    result["intermediate_label_rows"] = int(intermediate.sum())
    if intermediate.any():
        result["topics_intermediate_rows"] = scores(y[intermediate], pred[intermediate], mask[intermediate])
    tp, fp, fn = counts(y, pred, mask)
    result["per_topic"] = [{"id": t, "name": schema["nodes"][t]["name"],
        "train_support": int(train_support.get(t, 0)), "eval_support": int(y[:,j].sum()),
        "known_negative": int(((y[:,j] == 0) & mask[:,j]).sum()),
        "precision": float(tp[j]/max(1,tp[j]+fp[j])), "recall": float(tp[j]/max(1,tp[j]+fn[j])),
        "f1": float(2*tp[j]/max(1,2*tp[j]+fp[j]+fn[j]))} for j,t in enumerate(schema["labels"])]
    result["rare_topics"] = {}
    for name, lo, hi in (("unseen_train",0,1),("train_1_to_4",1,5),("train_5_to_19",5,20),("train_20_plus",20,10**9)):
        selected = np.asarray([lo <= train_support.get(t,0) < hi for t in schema["labels"]])
        result["rare_topics"][name] = {"label_count": int(selected.sum()), **scores(y[:,selected], pred[:,selected], mask[:,selected])}
    return result
