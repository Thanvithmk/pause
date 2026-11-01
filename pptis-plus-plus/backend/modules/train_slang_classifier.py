"""Train the slang-versus-literal head on frozen xlm-roberta-base encodings.

The pretrained encoder is not treated as a classifier. This script fits a linear
head on the mean of the target-token hidden states, chooses the slang threshold
on the validation split, and scores the held-out test split once.

Run from backend/:
    PYTHONPATH=. python -m modules.train_slang_classifier
"""

import csv
import json
import os
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel, AutoTokenizer

from modules.slang_layer import (
    SlangLiteralHead,
    contextual_vector,
    default_classifier_path,
    default_labels_path,
    default_metrics_path,
)

SEED = 42
ENCODER_NAME = "xlm-roberta-base"


def load_rows(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sentence = (row.get("sentence") or "").strip()
            target = (row.get("target") or "").strip()
            label = (row.get("label") or "").strip().lower()
            split = (row.get("split") or "").strip().lower()
            if not sentence or not target or label not in ("slang", "literal") or split not in ("train", "val", "test"):
                raise ValueError(f"Invalid labelled row: {row}")
            rows.append(
                {
                    "split": split,
                    "sentence": sentence,
                    "target": target,
                    "label": label,
                    "y": 1 if label == "slang" else 0,
                }
            )
    return rows


def assert_held_out(rows):
    seen = {}
    for row in rows:
        key = (row["sentence"], row["target"])
        if key in seen and seen[key] != row["split"]:
            raise ValueError(f"Sentence appears in more than one split: {key}")
        seen[key] = row["split"]
    test_sentences = {row["sentence"] for row in rows if row["split"] == "test"}
    leaked = [
        row["sentence"]
        for row in rows
        if row["split"] in ("train", "val") and row["sentence"] in test_sentences
    ]
    if leaked:
        raise ValueError(f"Test sentences leaked into train or val: {leaked}")


def encode(tokenizer, encoder, rows, device):
    encoder.eval()
    vectors = [
        contextual_vector(tokenizer, encoder, row["sentence"], row["target"], device)
        for row in rows
    ]
    return torch.stack(vectors, dim=0)


def class_metrics(y_true, y_pred):
    matrix = [[0, 0], [0, 0]]
    for truth, pred in zip(y_true, y_pred):
        matrix[int(truth)][int(pred)] += 1

    def prf(label):
        tp = matrix[label][label]
        fp = matrix[1 - label][label]
        fn = matrix[label][1 - label]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        return {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}

    literal = prf(0)
    slang = prf(1)
    youden = slang["recall"] + literal["recall"] - 1
    macro_f1 = (literal["f1"] + slang["f1"]) / 2
    return {
        "confusion_matrix": {
            "labels": ["literal", "slang"],
            "rows_are": "actual",
            "columns_are": "predicted",
            "values": matrix,
        },
        "literal": literal,
        "slang": slang,
        "macro_f1": macro_f1,
        "youden_j": youden,
    }


def choose_threshold(probabilities, labels):
    labels = [int(label) for label in labels]
    slang_probs = [p for p, label in zip(probabilities, labels) if label == 1]
    literal_probs = [p for p, label in zip(probabilities, labels) if label == 0]
    separated = bool(slang_probs and literal_probs) and min(slang_probs) > max(literal_probs)
    if separated:
        threshold = (max(literal_probs) + min(slang_probs)) / 2
        return float(threshold), "midpoint between highest validation literal P(Slang) and lowest validation slang P(Slang)"

    best = None
    steps = [step / 100 for step in range(5, 96)]
    for threshold in steps:
        predicted = [1 if prob >= threshold else 0 for prob in probabilities]
        metrics = class_metrics(labels, predicted)
        key = (metrics["macro_f1"], metrics["youden_j"], metrics["literal"]["recall"], -threshold)
        if best is None or key > best[0]:
            best = (key, float(threshold), metrics)
    return best[1], "validation sweep maximizing macro F1, then Youden J, then literal recall"


def probabilities_from_linear(linear, features):
    with torch.no_grad():
        probs = torch.softmax(linear(features.float()), dim=-1)[:, 1]
    return [float(value) for value in probs]


def fit_linear_head(features, labels, penalty):
    classifier = LogisticRegression(
        C=penalty,
        class_weight="balanced",
        solver="liblinear",
        max_iter=1000,
        random_state=SEED,
    )
    classifier.fit(features.numpy(), np.asarray(labels, dtype=np.int64))
    if list(classifier.classes_) != [0, 1]:
        raise RuntimeError(f"Unexpected class order: {classifier.classes_}")
    linear = nn.Linear(features.shape[1], 2)
    with torch.no_grad():
        linear.weight.zero_()
        linear.bias.zero_()
        linear.weight[1].copy_(torch.tensor(classifier.coef_.reshape(-1), dtype=torch.float32))
        linear.bias[1] = float(classifier.intercept_[0])
    head = SlangLiteralHead(features.shape[1])
    head.linear.load_state_dict(linear.state_dict())
    torch_probs = probabilities_from_linear(head, features)
    sklearn_probs = classifier.predict_proba(features.numpy())[:, 1]
    drift = max(abs(left - right) for left, right in zip(torch_probs, sklearn_probs))
    if drift > 1e-4:
        raise RuntimeError(f"Torch head does not match logistic probabilities (max abs diff {drift})")
    return head



def pack_predictions(rows, probabilities, threshold):
    packed = []
    for row, prob in zip(rows, probabilities):
        p_slang = float(prob)
        p_literal = 1.0 - p_slang
        if p_slang >= threshold:
            predicted = "slang"
        elif p_literal >= threshold:
            predicted = "literal"
        else:
            predicted = "uncertain"
        packed.append(
            {
                "sentence": row["sentence"],
                "target": row["target"],
                "gold": row["label"],
                "predicted": predicted,
                "p_slang": p_slang,
                "p_literal": p_literal,
            }
        )
    return packed


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    labels_path = default_labels_path()
    rows = load_rows(labels_path)
    assert_held_out(rows)
    by_split = {name: [row for row in rows if row["split"] == name] for name in ("train", "val", "test")}
    if not by_split["train"] or not by_split["val"] or not by_split["test"]:
        raise RuntimeError("Train, validation, and test splits must all be non-empty")

    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME)
    encoder = AutoModel.from_pretrained(ENCODER_NAME).to(device)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False

    encoded = {name: encode(tokenizer, encoder, by_split[name], device) for name in ("train", "val", "test")}
    best_head = None
    for penalty in (0.1, 0.3, 1.0, 3.0):
        candidate = fit_linear_head(encoded["train"], [row["y"] for row in by_split["train"]], penalty)
        candidate_val = probabilities_from_linear(candidate, encoded["val"])
        candidate_threshold, candidate_rule = choose_threshold(candidate_val, [row["y"] for row in by_split["val"]])
        candidate_pred = [1 if prob >= candidate_threshold else 0 for prob in candidate_val]
        candidate_metrics = class_metrics([row["y"] for row in by_split["val"]], candidate_pred)
        key = (candidate_metrics["macro_f1"], candidate_metrics["youden_j"], -penalty)
        if best_head is None or key > best_head[0]:
            best_head = (key, candidate, candidate_val, candidate_threshold, candidate_rule, penalty, candidate_metrics)
    _key, linear, val_probs, threshold, threshold_rule, penalty, val_metrics = best_head
    test_probs = probabilities_from_linear(linear, encoded["test"])
    test_pred = [1 if prob >= threshold else 0 for prob in test_probs]
    test_metrics = class_metrics([row["y"] for row in by_split["test"]], test_pred)
    test_rows = pack_predictions(by_split["test"], test_probs, threshold)

    classifier_path = default_classifier_path()
    metrics_path = default_metrics_path()
    os.makedirs(os.path.dirname(classifier_path), exist_ok=True)
    torch.save(
        {
            "encoder_name": ENCODER_NAME,
            "state_dict": linear.state_dict(),
            "threshold": threshold,
            "threshold_rule": threshold_rule,
            "label_order": ["literal", "slang"],
            "representation": "mean target-token hidden state",
            "penalty_C": penalty,
            "seed": SEED,
        },
        classifier_path,
    )

    contrast = {
        "The chicken is cooked.",
        "Bro you're cooked 💀",
        "I cooked dinner.",
        "That exam cooked me.",
        "This fit is fire.",
        "The building caught fire.",
    }
    report = {
        "encoder_name": ENCODER_NAME,
        "encoder_trained_on_slang": False,
        "head": "L2 logistic regression on the frozen mean of target-token hidden states",
        "penalty_C": penalty,
        "seed": SEED,
        "device": "cpu",
        "dataset": os.path.abspath(labels_path),
        "counts": {name: len(by_split[name]) for name in ("train", "val", "test")},
        "label_counts": {
            name: dict(Counter(row["label"] for row in by_split[name])) for name in ("train", "val", "test")
        },
        "checkpoint": os.path.abspath(classifier_path),
        "threshold": threshold,
        "threshold_rule": threshold_rule,
        "validation": val_metrics,
        "test": test_metrics,
        "test_predictions": test_rows,
        "contrast_examples": [row for row in test_rows if row["sentence"] in contrast],
        "note": (
            "Probabilities are the trained head's outputs on this hand-labelled file. "
            "They are not the illustrative 0.04/0.96 figures printed in the master plan."
        ),
    }
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({
        "checkpoint": report["checkpoint"],
        "threshold": threshold,
        "validation_macro_f1": val_metrics["macro_f1"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_confusion_matrix": test_metrics["confusion_matrix"]["values"],
        "contrast_examples": report["contrast_examples"],
    }, indent=2))


if __name__ == "__main__":
    main()
