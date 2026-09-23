"""Serial one-vs-rest SGD: sparse text matrix; no parallel 593-way copies."""
from pathlib import Path
import joblib
import numpy as np
from scipy import sparse
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from .data import feature_text
from .metrics import targets


def train(rows, schema, config, run):
    texts = [feature_text(r) for r in rows]
    cfg = config["baseline"]
    word = TfidfVectorizer(ngram_range=(1,2), max_features=cfg["word_features"],
                          min_df=cfg["min_df"], sublinear_tf=True, dtype=np.float32)
    char = TfidfVectorizer(analyzer="char", ngram_range=(3,5), max_features=cfg["char_features"],
                          min_df=cfg["min_df"], sublinear_tf=True, dtype=np.float32)
    x = sparse.hstack([word.fit_transform(texts), char.fit_transform(texts)], format="csr", dtype=np.float32)
    def new_model():
        return SGDClassifier(loss="log_loss", alpha=cfg["alpha"], max_iter=cfg["epochs"],
                             tol=None, random_state=config["seed"], n_jobs=1)
    subject = new_model().fit(x, [schema["subjects"].index(r["subject"]) for r in rows])
    y, mask = targets(rows, schema)
    weights = np.zeros((len(schema["labels"]), x.shape[1]), dtype=np.float32)
    bias = np.zeros(len(schema["labels"]), dtype=np.float32)
    active = np.zeros(len(schema["labels"]), dtype=bool)
    for j in range(len(active)):
        known = mask[:,j]
        if np.unique(y[known,j]).size != 2:
            continue  # No positive OR no known negative: abstain.
        model = new_model().fit(x[known], y[known,j])
        weights[j] = model.coef_[0]
        bias[j] = model.intercept_[0]
        active[j] = True
        if (j+1) % 100 == 0:
            print(f"topics {j+1}/{len(active)}", flush=True)
    joblib.dump({"word":word,"char":char,"subject":subject,"weights":weights,"bias":bias,"active":active}, Path(run)/"baseline.joblib", compress=3)
    return {"active_topics":int(active.sum()), "features":int(x.shape[1]),
            "csr_bytes":int(x.data.nbytes+x.indices.nbytes+x.indptr.nbytes), "topic_weight_bytes":int(weights.nbytes)}


def predict(texts, run, batch_size=128):
    model = joblib.load(Path(run)/"baseline.joblib")
    subj, probs = [], []
    for start in range(0,len(texts),batch_size):
        chunk = texts[start:start+batch_size]
        x = sparse.hstack([model["word"].transform(chunk),model["char"].transform(chunk)],format="csr",dtype=np.float32)
        subj.append(model["subject"].predict(x))
        p = expit(x @ model["weights"].T + model["bias"])
        p[:,~model["active"]] = 0
        probs.append(p)
    return np.concatenate(subj), np.concatenate(probs)
