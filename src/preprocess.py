"""Pre-processing pipeline for Continual-Task-Stream-25.
This module builds a task stream generator that yields
(name, train_loader, val_loader) tuples ready for continual training.
All HuggingFace assets are cached under `.cache/`.
"""
from __future__ import annotations

import random
from functools import partial
from typing import Any, Dict, Iterator, List, Tuple

import torch
from datasets import IterableDataset, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase

__all__ = ["get_task_stream"]

# -----------------------------------------------------------------------------
# Task registry ----------------------------------------------------------------
# -----------------------------------------------------------------------------
# We intentionally include >25 widely available HF datasets.  The loader will
# skip tasks that cannot be downloaded and stop once `cfg.dataset.num_tasks` is
# reached; this guarantees robustness across execution environments.
TASKS: List[Dict[str, Any]] = [
    # Translation (src, tgt language codes supplied)
    {"name": "opus_books_en_fr", "hf": ("opus_books", "en-fr"), "type": "translation", "src": "en", "tgt": "fr"},
    {"name": "wmt14_en_de", "hf": ("wmt14", "de-en"), "type": "translation", "src": "en", "tgt": "de"},
    {"name": "iwslt2017_en_de", "hf": ("iwslt2017", "iwslt2017-en-de"), "type": "translation", "src": "en", "tgt": "de"},
    {"name": "ted_hrlr_pt_en", "hf": ("ted_hrlr_translate", "pt_to_en"), "type": "translation", "src": "pt", "tgt": "en"},
    # Summarisation
    {"name": "cnn_dailymail", "hf": ("cnn_dailymail", "3.0.0"), "type": "summarisation", "doc": "article", "sum": "highlights"},
    {"name": "xsum", "hf": ("xsum", None), "type": "summarisation", "doc": "document", "sum": "summary"},
    {"name": "samsum", "hf": ("samsum", None), "type": "summarisation", "doc": "dialogue", "sum": "summary"},
    # Classification (single sentence)
    {"name": "ag_news", "hf": ("ag_news", None), "type": "classification", "text": "text", "label": "label"},
    {"name": "amazon_polarity", "hf": ("amazon_polarity", None), "type": "classification", "text": "content", "label": "label"},
    {"name": "yelp_polarity", "hf": ("yelp_polarity", None), "type": "classification", "text": "text", "label": "label"},
    {"name": "sst2", "hf": ("glue", "sst2"), "type": "classification", "text": "sentence", "label": "label"},
    {"name": "mnli", "hf": ("glue", "mnli"), "type": "classification", "text": "premise", "label": "label"},
    {"name": "rte", "hf": ("glue", "rte"), "type": "classification", "text": "sentence1", "label": "label"},
    {"name": "cola", "hf": ("glue", "cola"), "type": "classification", "text": "sentence", "label": "label"},
    {"name": "mrpc", "hf": ("glue", "mrpc"), "type": "classification", "text": "sentence1", "label": "label"},
    {"name": "boolq", "hf": ("super_glue", "boolq"), "type": "classification", "text": "passage", "label": "label"},
    {"name": "trec", "hf": ("trec", None), "type": "classification", "text": "text", "label": "label"},
    {"name": "dbpedia_14", "hf": ("dbpedia_14", None), "type": "classification", "text": "content", "label": "label"},
    # QA treated generatively (question→answer)
    {"name": "squad_v2", "hf": ("squad_v2", None), "type": "qa", "question": "question", "answers": "answers"},
    {"name": "tydiqa_goldp", "hf": ("tydiqa", "goldp"), "type": "qa", "question": "question", "answers": "answers"},
    # Code generation (treated generatively)
    {"name": "code_alpaca", "hf": ("code_alpaca", "default"), "type": "code", "prompt": "prompt", "completion": "completion"},
]


# -----------------------------------------------------------------------------
# Prompt builders -------------------------------------------------------------
# -----------------------------------------------------------------------------

def build_translation(example, src: str, tgt: str):
    # HuggingFace translation datasets typically have a 'translation' field
    # containing a dictionary with language codes as keys
    translation = example.get("translation", example)
    src_text = translation.get(src, example.get(src, ""))
    tgt_text = translation.get(tgt, example.get(tgt, ""))
    return f"Translate from {src} to {tgt}: {src_text}\nTARGET:", tgt_text


def build_summarisation(example, doc: str, sum: str):
    doc_text = example.get(doc, "")
    sum_text = example.get(sum, "")
    return f"Summarise: {doc_text}\nSUMMARY:", sum_text


def build_classification(example, text: str, label: str):
    text_content = example.get(text, "")
    label_value = example.get(label, 0)
    return f"Classify: {text_content}\nLABEL:", str(int(label_value))  # keep numeric label as string token


def build_qa(example, question: str, answers: str):
    question_text = example.get(question, "")
    answers_data = example.get(answers, {})
    if isinstance(answers_data, dict) and answers_data.get("text"):
        tgt = answers_data["text"][0]
    else:
        tgt = "unanswerable"
    return f"Answer the question: {question_text}\nANSWER:", tgt


def build_code(example, prompt: str, completion: str):
    prompt_text = example.get(prompt, "")
    completion_text = example.get(completion, "")
    return f"### Instruction:\n{prompt_text}\n### Response:\n", completion_text


# -----------------------------------------------------------------------------
# Collate function -----------------------------------------------------------
# -----------------------------------------------------------------------------

def collate_fn_builder(tokenizer: PreTrainedTokenizerBase):
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def _fn(batch):
        ids = [torch.tensor(ex["input_ids"], dtype=torch.long) for ex in batch]
        labs = [torch.tensor(ex["labels"], dtype=torch.long) for ex in batch]
        attn = [torch.tensor(ex["attention_mask"], dtype=torch.long) for ex in batch]
        ids = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=pad_id)
        labs = torch.nn.utils.rnn.pad_sequence(labs, batch_first=True, padding_value=-100)
        attn = torch.nn.utils.rnn.pad_sequence(attn, batch_first=True, padding_value=0)
        return {"input_ids": ids, "attention_mask": attn, "labels": labs}

    return _fn


# -----------------------------------------------------------------------------
# Main API --------------------------------------------------------------------
# -----------------------------------------------------------------------------

def get_task_stream(cfg, tokenizer: PreTrainedTokenizerBase):
    """Yield a deterministic but diverse task stream of length cfg.dataset.num_tasks."""
    random.seed(0)
    task_cnt = 0
    max_len = cfg.dataset.max_seq_length
    batch_size = cfg.training.batch_size
    desired = cfg.dataset.num_tasks

    for task in TASKS:
        if task_cnt >= desired:
            break
        try:
            dataset_name, dataset_conf = task["hf"]
            ds = load_dataset(dataset_name, dataset_conf, cache_dir=".cache/", streaming=cfg.dataset.streaming)
        except Exception as e:
            print(f"[WARN] Could not load {task['name']}: {e}")
            continue

        train_split = "train" if "train" in ds else list(ds.keys())[0]
        val_split = (
            "validation"
            if "validation" in ds
            else "test"
            if "test" in ds
            else train_split
        )
        ds_train_raw, ds_val_raw = ds[train_split], ds[val_split]

        # Limit streaming datasets for reproducibility -----------------------
        if isinstance(ds_train_raw, IterableDataset):
            ds_train_raw = ds_train_raw.take(cfg.dataset.max_samples_per_task)
        if isinstance(ds_val_raw, IterableDataset):
            ds_val_raw = ds_val_raw.take(int(cfg.dataset.max_samples_per_task * 0.1))

        # Select prompt builder ---------------------------------------------
        if task["type"] == "translation":
            builder = partial(build_translation, src=task["src"], tgt=task["tgt"])
        elif task["type"] == "summarisation":
            builder = partial(build_summarisation, doc=task["doc"], sum=task["sum"])
        elif task["type"] == "classification":
            builder = partial(build_classification, text=task["text"], label=task["label"])
        elif task["type"] == "qa":
            builder = partial(build_qa, question=task["question"], answers=task["answers"])
        elif task["type"] == "code":
            builder = partial(build_code, prompt=task["prompt"], completion=task["completion"])
        else:
            raise ValueError(task["type"])

        # Tokenisation -------------------------------------------------------
        def tok_fn(example):
            prompt, tgt = builder(example)
            enc_p = tokenizer(prompt, truncation=True, max_length=max_len)
            enc_t = tokenizer(tgt, truncation=True, max_length=max_len)
            input_ids = enc_p["input_ids"] + enc_t["input_ids"]
            labels = [-100] * len(enc_p["input_ids"]) + enc_t["input_ids"]
            return {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
            }

        ds_train = ds_train_raw.map(tok_fn, remove_columns=ds_train_raw.column_names)
        ds_val = ds_val_raw.map(tok_fn, remove_columns=ds_val_raw.column_names)

        collate = collate_fn_builder(tokenizer)
        train_loader = DataLoader(
            ds_train,
            batch_size=batch_size,
            shuffle=not cfg.dataset.streaming,
            collate_fn=collate,
        )
        val_loader = DataLoader(
            ds_val,
            batch_size=cfg.evaluation.eval_batch_size,
            shuffle=False,
            collate_fn=collate,
        )
        yield task["name"], train_loader, val_loader
        task_cnt += 1