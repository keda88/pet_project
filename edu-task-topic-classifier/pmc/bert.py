"""FP32 ruBERT with two heads, masked BCE, MPS and CPU execution."""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # Before importing torch.
import time
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
from .data import feature_text, read_json, write_json
from .metrics import targets


def device_for(requested):
    if requested == "cpu":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if requested == "mps":
        raise RuntimeError("MPS requested but unavailable")
    print("MPS unavailable; using CPU", flush=True)
    return torch.device("cpu")


class DualHead(nn.Module):
    def __init__(self, encoder, subjects, topics):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(.1)
        self.subject = nn.Linear(encoder.config.hidden_size, subjects)
        self.topics = nn.Linear(encoder.config.hidden_size, topics)

    def forward(self, tokens):
        pooled = self.dropout(self.encoder(**tokens).last_hidden_state[:,0])
        return self.subject(pooled), self.topics(pooled)


def masked_loss(subject_logits, topic_logits, subjects, y, mask, pos_weight, topic_weight):
    subject = nn.functional.cross_entropy(subject_logits, subjects, reduction="none")
    raw = nn.functional.binary_cross_entropy_with_logits(topic_logits, y, reduction="none", pos_weight=pos_weight)
    topics = (raw * mask).sum(1) / mask.sum(1).clamp_min(1)
    return (subject + topic_weight * topics).mean()


def batch_tokens(tokenizer, texts, device, length, fixed_padding=False):
    return {k:v.to(device) for k,v in tokenizer(texts, padding="max_length" if fixed_padding else True, truncation=True,
                    max_length=length, return_tensors="pt").items()}


def maintain_mps(device, cfg, step):
    interval = cfg.get("mps_empty_cache_interval", 16)
    if device.type == "mps" and step % interval == 0:
        torch.mps.synchronize()
        torch.mps.empty_cache()


def configure_device(cfg):
    device = device_for(cfg["device"])
    if device.type == "mps":
        torch.mps.set_per_process_memory_fraction(cfg.get("mps_memory_fraction", 1.0))
    print(f"BERT device: {device}", flush=True)
    return device


def memory_status(device):
    if device.type != "mps":
        return ""
    return (f", MPS tensors={torch.mps.current_allocated_memory()/2**30:.2f} GiB"
            f", driver={torch.mps.driver_allocated_memory()/2**30:.2f} GiB")


def save_checkpoint(model, path):
    temporary = path.with_suffix(".tmp")
    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()}, temporary)
    temporary.replace(path)


def train(rows, validation, schema, config, run):
    run = Path(run)
    cfg = config["bert"]
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(cfg.get("cpu_threads", 6))
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = configure_device(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"], trust_remote_code=False)
    encoder = AutoModel.from_pretrained(cfg["model"], revision=cfg["revision"], trust_remote_code=False,
                                        attn_implementation="eager")
    encoder.config.save_pretrained(run / "encoder_config")
    tokenizer.save_pretrained(run / "tokenizer")
    if cfg.get("gradient_checkpointing", True):
        encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    model = DualHead(encoder,len(schema["subjects"]),len(schema["labels"])).to(device)
    y, mask = targets(rows, schema)
    vy, vm = targets(validation, schema)
    active = (y.sum(0) > 0) & (((1-y)*mask).sum(0) > 0)
    mask &= active[None,:]
    vm &= active[None,:]
    pos = y.sum(0)
    neg = ((1-y)*mask).sum(0)
    pw = torch.tensor(np.clip(np.sqrt(neg/np.maximum(pos,1)),1,cfg["pos_weight_cap"]),dtype=torch.float32,device=device)
    write_json(run / "active.json", active.tolist())
    sy = np.asarray([schema["subjects"].index(r["subject"]) for r in rows])
    vsy = np.asarray([schema["subjects"].index(r["subject"]) for r in validation])
    optimizer = torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=.01,
                                  foreach=cfg.get("optimizer_foreach", False))
    batch_size, accum = cfg["batch_size"], cfg["gradient_accumulation"]
    best, stale, history = float("inf"), 0, []
    rng = np.random.default_rng(config["seed"])
    def loss_for(indices, data, yy, mm, ss):
        token = batch_tokens(tokenizer,[feature_text(data[i]) for i in indices],device,cfg["max_length"],cfg.get("fixed_padding",False))
        sl, tl = model(token)
        return masked_loss(sl,tl,torch.tensor(ss[indices],dtype=torch.long,device=device),
            torch.tensor(yy[indices],dtype=torch.float32,device=device),
            torch.tensor(mm[indices],dtype=torch.float32,device=device),pw,cfg["topic_loss_weight"])
    for epoch in range(cfg["epochs"]):
        model.train()
        order = rng.permutation(len(rows))
        train_total = 0.
        started = time.monotonic()
        step = 0
        # Accumulate per-window sample means, including a short final window.
        for start in range(0,len(order),batch_size*accum):
            window = order[start:start+batch_size*accum]
            optimizer.zero_grad(set_to_none=True)
            for offset in range(0,len(window),batch_size):
                idx = window[offset:offset+batch_size]
                loss = loss_for(idx,rows,y,mask,sy)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite loss; try device=cpu")
                (loss * (len(idx)/len(window))).backward()
                train_total += float(loss.detach().cpu())*len(idx)
                del loss
                step += 1
                maintain_mps(device,cfg,step)
            nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            maintain_mps(device,cfg,step)
            if step % cfg.get("log_interval", 128) == 0 or start+len(window) == len(order):
                print(f"Epoch {epoch+1} batch {step}/{(len(rows)+batch_size-1)//batch_size}: "
                      f"{(time.monotonic()-started)/step:.3f} s/batch" + memory_status(device),flush=True)
        model.eval()
        validation_total = 0.
        with torch.inference_mode():
            for start in range(0,len(validation),batch_size):
                idx = np.arange(start,min(start+batch_size,len(validation)))
                validation_total += float(loss_for(idx,validation,vy,vm,vsy).cpu())*len(idx)
                maintain_mps(device,cfg,start//batch_size+1)
        value = validation_total/len(validation)
        if not np.isfinite(value):
            raise RuntimeError("Non-finite validation loss")
        item = {"epoch":epoch+1,"train_loss":train_total/len(rows),"validation_loss":value}
        history.append(item)
        print(item,flush=True)
        save_checkpoint(model,run/"last.pt")
        if value < best-cfg["min_delta"]:
            best, stale = value, 0
            save_checkpoint(model,run/"best.pt")
        else:
            stale += 1
        write_json(run/"history.json",history)
        if stale >= cfg["patience"]:
            break
    return {"device":str(device),"precision":"float32","active_topics":int(active.sum()),
            "best_validation_loss":best,"epochs_completed":len(history)}


def predict(texts, run, device_override=None):
    run = Path(run)
    config = read_json(run/"config.json")["bert"]
    schema = read_json(run/"schema.json")
    device = configure_device(dict(config,device=device_override or config["device"]))
    tokenizer = AutoTokenizer.from_pretrained(run/"tokenizer",local_files_only=True)
    encoder_config = AutoConfig.from_pretrained(run/"encoder_config",local_files_only=True)
    encoder = AutoModel.from_config(encoder_config,attn_implementation="eager")
    model = DualHead(encoder,len(schema["subjects"]),len(schema["labels"]))
    model.load_state_dict(torch.load(run/"best.pt",map_location="cpu",weights_only=True))
    model.to(device).eval()
    active = np.asarray(read_json(run/"active.json"),dtype=bool)
    subjects, probabilities = [], []
    with torch.inference_mode():
        for start in range(0,len(texts),config["batch_size"]):
            sl, tl = model(batch_tokens(tokenizer,texts[start:start+config["batch_size"]],device,config["max_length"],config.get("fixed_padding",False)))
            subjects.append(sl.argmax(1).cpu().numpy())
            p = tl.sigmoid().cpu().numpy()
            p[:,~active] = 0
            probabilities.append(p)
            del sl, tl
            maintain_mps(device,config,start//config["batch_size"]+1)
    return np.concatenate(subjects),np.concatenate(probabilities)
