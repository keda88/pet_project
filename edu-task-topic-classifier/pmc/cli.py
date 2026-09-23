"""Reproducible CLI; importing prepare does not import ML libraries."""
import argparse
import importlib.metadata
import json
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from .data import (prepare, read_json, write_json, verify_artifacts, load_rows,
                   validate_config, feature_text, sha256)


def reserve_test(data, run):
    """One final evaluation per prepared dataset, across all model runs."""
    path = Path(data)/"TEST_CONSUMED.json"
    with path.open("x", encoding="utf-8") as f:
        json.dump({"run":str(Path(run).resolve()),"utc":datetime.now(timezone.utc).isoformat(),
                   "status":"reserved; failures also consume test access"}, f)
    return path


def provenance(data):
    return sha256(Path(data)/"manifest.json")


def predict_scores(texts, run, config, device=None):
    if config["kind"] == "baseline":
        from .baseline import predict
        return predict(texts,run)
    from .bert import predict
    return predict(texts,run,device)


def train_command(args):
    config = validate_config(read_json(args.config))
    manifest = verify_artifacts(args.data)
    run = Path(args.run)
    if run.exists():
        raise FileExistsError("Use a new --run directory; never overwrite an experiment")
    rows = load_rows(args.data,"train",config["train_limit"],config["seed"])
    validation = load_rows(args.data,"validation",config["validation_limit"],config["seed"])
    schema = read_json(Path(args.data)/"schema.json")
    if not rows or not validation or set(r["subject"] for r in rows) != set(schema["subjects"]):
        raise ValueError("Empty split or training sample missing a subject")
    run.mkdir(parents=True)
    write_json(run/"config.json",config)
    write_json(run/"schema.json",schema)
    support = dict(Counter(t for r in rows for t in r["targets"]))
    write_json(run/"train_support.json",support)
    environment = {"python":platform.python_version(),"platform":platform.platform(),
        "packages":{d.metadata["Name"]:d.version for d in importlib.metadata.distributions()},
        "code_sha256":{p.name:sha256(p) for p in Path(__file__).parent.glob("*.py")},
        "data_manifest_sha256":provenance(args.data),"manifest":manifest,
        "train_rows":len(rows),"validation_rows":len(validation)}
    write_json(run/"provenance.json",environment)
    if config["kind"] == "baseline":
        from .baseline import train
        training = train(rows,schema,config,run)
    else:
        from .bert import train
        training = train(rows,validation,schema,config,run)
    write_json(run/"training.json",training)
    from .metrics import tune_threshold, report
    subjects, probabilities = predict_scores([feature_text(r) for r in validation],run,config)
    threshold = tune_threshold(validation,schema,probabilities,"validation")
    write_json(run/"threshold.json",{"threshold":threshold,"fitted_on":"validation",
                                    "criterion":"masked topic micro F1","grid":"0.05..0.95 step 0.05"})
    result = report(validation,schema,subjects,probabilities,threshold,support)
    write_json(run/"validation_metrics.json",result)
    write_json(run/"COMPLETE.json",{"completed":True})
    print(json.dumps({"run":str(run),"training":training,"validation":{k:v for k,v in result.items() if k != "per_topic"}},ensure_ascii=False,indent=2))


def evaluate_command(args):
    run = Path(args.run)
    if not (run/"COMPLETE.json").exists():
        raise ValueError("Training is incomplete")
    verify_artifacts(args.data)
    if provenance(args.data) != read_json(run/"provenance.json")["data_manifest_sha256"]:
        raise ValueError("Evaluation dataset differs from training dataset")
    config = validate_config(read_json(run/"config.json"))
    if args.split == "test":
        if not args.final_test or config["mode"] != "full":
            raise ValueError("Test requires full config and explicit --final-test after model selection")
        reserve_test(args.data,run)
    rows = load_rows(args.data,args.split,config["validation_limit"] if args.split == "validation" else None,config["seed"])
    from .metrics import report
    subjects, probabilities = predict_scores([feature_text(r) for r in rows],run,config,args.device)
    threshold = read_json(run/"threshold.json")["threshold"]
    result = report(rows,read_json(run/"schema.json"),subjects,probabilities,threshold,read_json(run/"train_support.json"))
    path = run/f"{args.split}_metrics.json"
    write_json(path,result)
    print(path)


def predict_command(args):
    run = Path(args.run)
    if not (run/"COMPLETE.json").exists():
        raise ValueError("Training is incomplete")
    text = args.text if args.text is not None else Path(args.text_file).read_text(encoding="utf-8")
    text = feature_text({"task":text})
    config = validate_config(read_json(run/"config.json"))
    schema = read_json(run/"schema.json")
    subjects, probabilities = predict_scores([text],run,config,args.device)
    threshold = read_json(run/"threshold.json")["threshold"]
    selected = [t for t,p in zip(schema["labels"],probabilities[0]) if p >= threshold]
    from .hierarchy import most_specific, close_topics
    most = most_specific(selected,schema)
    subject = schema["subjects"][int(subjects[0])]
    result = {"subject":{"id":subject,"name":schema["nodes"][subject]["name"]},
              "target_topic_ids":most,"ancestor_topic_ids":sorted(close_topics(most,schema),key=int),
              "threshold":threshold,"topics":[{"id":t,"name":schema["nodes"][t]["name"],"score":float(p)}
                    for t,p in zip(schema["labels"],probabilities[0]) if t in most],
              "abstained_topics":not most}
    print(json.dumps(result,ensure_ascii=False,indent=2))


def main():
    parser = argparse.ArgumentParser(description="Классификатор учебных задач: только текст условия")
    sub = parser.add_subparsers(dest="command",required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--source",default="../physics_math_dataset")
    p.add_argument("--output",default="artifacts/data")
    p.add_argument("--include-review",action="store_true")
    p = sub.add_parser("train")
    p.add_argument("--data",default="artifacts/data")
    p.add_argument("--config",required=True)
    p.add_argument("--run",required=True)
    p = sub.add_parser("evaluate")
    p.add_argument("--data",default="artifacts/data")
    p.add_argument("--run",required=True)
    p.add_argument("--split",choices=["validation","test"],default="validation")
    p.add_argument("--final-test",action="store_true")
    p.add_argument("--device",choices=["cpu","auto","mps"])
    p = sub.add_parser("predict")
    p.add_argument("--run",required=True)
    text = p.add_mutually_exclusive_group(required=True)
    text.add_argument("--text")
    text.add_argument("--text-file")
    p.add_argument("--device",choices=["cpu","auto","mps"])
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(args.source,args.output,args.include_review),ensure_ascii=False,indent=2))
    elif args.command == "train":
        train_command(args)
    elif args.command == "evaluate":
        evaluate_command(args)
    else:
        predict_command(args)


if __name__ == "__main__":
    main()
