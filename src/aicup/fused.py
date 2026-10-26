"""The fused (Human SL + L1b) question model of the submission (issue #9).

A question's fused features are qfeatures.question_features of its games' 18 Human SL sums (24) followed by
the same features of the games' 10 L1b-head sums (16) -- l1b_cv.feature_sets' "fused", column for column.
The fused MLP (rankmodel.MLPModel(hidden=32), fitted by scripts/fit_fused_model.py exactly as
scripts/l1b_fold0_check.py did) is stored as a rank_model.json-like document that also names the L1b
models whose scores produced its training features (heads file, state and trunk sha256, training folds,
the folds they gave features for). At prediction time every L1b model of the document scores the
question's games, the MLP runs once per model on that model's fused features, and the probabilities are
averaged (the rule the fold-0 check validated); pairing_order makes sure the shards' L1b models are
exactly the document's.
"""
import json
from pathlib import Path

import numpy as np

from aicup.humansl import PROFILES
from aicup.ladder import environment, to_json
from aicup.metric import LEVELS
from aicup.qfeatures import feature_names, question_features
from aicup.rankmodel import MLPModel, from_state
from aicup.setmodel_io import file_sha256  # noqa: F401  (re-exported for callers and tests)

NAME = "mlp_h32_fused"
FORMAT = 1
FUSED_FEATURE_NAMES = feature_names(PROFILES) + [f"l1b_{n}" for n in feature_names(LEVELS)]
MODEL_KEYS = ("name", "heads_file", "state_sha256", "trunk_file", "trunk_sha256", "train_folds", "feature_folds")


def fused_features(hsl_sums, l1b_sums, n_moves):
    """[40]: question_features of the games' Human SL sums [m, 18], then of their L1b sums [m, 10], both with n_moves [m]."""
    return np.concatenate([question_features(hsl_sums, n_moves), question_features(l1b_sums, n_moves)])


def l1b_model_record(name, heads_file, sidecar, feature_folds):
    """The document's record of one L1b model: its heads file name, the sidecar's sha256s and training folds
    (aicup.l1b.save_heads), and the folds whose features it supplied."""
    return {"name": name, "heads_file": Path(heads_file).name, "state_sha256": sidecar["state_sha256"],
            "trunk_file": sidecar.get("trunk_file"), "trunk_sha256": sidecar.get("trunk_sha256"),
            "train_folds": sidecar.get("train_folds"), "feature_folds": [int(f) for f in feature_folds]}


def fused_document(mlp, l1b_models, train_folds, per_player, n_train, questions_sha256, extra=None):
    """The rank_model_fused.json document: name, features, profiles, levels, training provenance, l1b_models, state."""
    for m in l1b_models:
        missing = [k for k in MODEL_KEYS if k not in m]
        if missing:
            raise ValueError(f"l1b model record {m.get('name')!r} lacks {missing}")
    return {"name": NAME, "format": FORMAT, "features": list(FUSED_FEATURE_NAMES), "profiles": list(PROFILES), "levels": list(LEVELS),
            "train_folds": [int(f) for f in train_folds], "per_player": int(per_player), "train_questions": int(n_train),
            "questions_sha256": questions_sha256, "l1b_models": [dict(m) for m in l1b_models],
            "role": "the submission's decision model: MLP on fused Human SL + L1b question features, averaged over the L1b models",
            **(extra or {}), **environment("cpu"), "state": mlp.state_dict()}


def save_fused_model(doc, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_json(doc, indent=1) + "\n")
    return path


def load_fused_model(path):
    """(document with file_sha256 added, MLPModel); ValueError when its features / profiles / levels are not this code's
    or it names no L1b model."""
    path = Path(path)
    doc = json.loads(path.read_text())
    if doc.get("features") != FUSED_FEATURE_NAMES:
        raise ValueError(f"{path}: features differ from this code's fused features ({len(doc.get('features') or [])} vs {len(FUSED_FEATURE_NAMES)})")
    if doc.get("profiles") != list(PROFILES) or doc.get("levels") != list(LEVELS):
        raise ValueError(f"{path}: profiles or levels differ from this code's")
    if not doc.get("l1b_models"):
        raise ValueError(f"{path}: l1b_models is empty; the fused model must name the L1b models its features come from")
    model = from_state(doc["state"])
    if not isinstance(model, MLPModel):
        raise ValueError(f"{path}: state is not an MLP")
    return {**doc, "file_sha256": file_sha256(path)}, model


def pairing_order(doc, shard_models):
    """For each L1b model of the fused document, the index of the shards' model with the same state and trunk sha256.
    ValueError when a document model is missing from the shards, a shard model is not in the document, or the shards
    carry no L1b scores at all."""
    if not shard_models:
        raise ValueError("the shards carry no L1b scores (the shard step needs --l1b-heads for a fused decision)")
    key = lambda m: (m.get("state_sha256"), m.get("trunk_sha256"))  # noqa: E731
    where = {}
    for i, m in enumerate(shard_models):
        where.setdefault(key(m), []).append(i)
    order = []
    for m in doc["l1b_models"]:
        hits = where.get(key(m), [])
        if not hits:
            same_heads = [s for s in shard_models if s.get("state_sha256") == m.get("state_sha256")]
            why = "its trunk sha256 differs" if same_heads else "it is not among the shards' L1b models"
            raise ValueError(f"fused model's L1b model {m['name']} ({m.get('heads_file')}): {why}")
        if len(hits) > 1:
            raise ValueError(f"fused model's L1b model {m['name']} appears {len(hits)} times among the shards' models")
        order.append(hits[0])
    extra = [s.get("name") for i, s in enumerate(shard_models) if i not in order]
    if extra:
        raise ValueError(f"shard L1b models {extra} are not in the fused model; the shards must be scored with exactly its models")
    return order


def fused_predict(mlp, hsl_sums, l1b_sums, n_moves, order):
    """[10]: the mean over the document's models (shard columns `order`) of the MLP's probabilities on fused features."""
    probs = [mlp.predict_proba(fused_features(hsl_sums, l1b_sums[:, k, :], n_moves)[None, :])[0] for k in order]
    return np.mean(probs, axis=0)
