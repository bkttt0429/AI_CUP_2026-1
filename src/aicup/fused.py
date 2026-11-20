"""The fused (Human SL + L1b) question model of the submission (issue #9) and its opponent extension (issue #12).

A question's fused features are qfeatures.question_features of its games' 18 Human SL sums (24) followed by
the same features of the games' 10 L1b-head sums (16) -- l1b_cv.feature_sets' "fused", column for column.
The fused_opp features append the same 16 features of the L1b sums over the *opponent's* moves of the same
games, with the opponent's move counts (l1b_score.py --opponent; l1b_opp_check.features' "fused_opp"): online
games pair players of similar ranks, so the opponent's moves are a second look at the target's rank. The
fused_opp_diff features (issue #25) append three more: over the question's games the mean of (opponent pmi -
own pmi), the mean and the std of the opponent pmi, pmi being the posterior-mean rank index of a game's ten
L1b sums -- the server pairs players by their labels, so the opponents' strength relative to the target's own
play carries label-side information the 56 features left on the table (scripts/error_analysis.py, +0.018).
The fused MLP (rankmodel.MLPModel(hidden=32), fitted by scripts/fit_fused_model.py exactly as
scripts/l1b_fold0_check.py / l1b_opp_fold0_check.py did) is stored as a rank_model.json-like document that
also names the L1b models whose scores produced its training features (heads file, state and trunk sha256,
training folds, the folds they gave features for) and says whether it takes the opponent features. At
prediction time every L1b model of the document scores the question's games (and their opponents), the MLP
runs once per model on that model's features, and the probabilities are averaged (the rule the fold-0
checks validated); pairing_order makes sure the shards' L1b models are exactly the document's.
"""
import json
from pathlib import Path

import numpy as np

from aicup.handicap import HD_FEATURE_NAMES
from aicup.evidence import posterior_mean_index
from aicup.humansl import PROFILES
from aicup.ladder import environment, to_json
from aicup.metric import LEVELS
from aicup.qfeatures import feature_names, question_features, sum_cols
from aicup.rankmodel import MLPModel, from_state
from aicup.setmodel_io import file_sha256  # noqa: F401  (re-exported for callers and tests)

NAME = "mlp_h32_fused"
NAME_OPP = "mlp_h32_fused_opp"
NAME_OPP_DIFF = "mlp_h32_fused_opp_diff"
NAME_OPP_DIFF_HD = "mlp_h32_fused_opp_diff_hd"
FORMAT = 1
FUSED_FEATURE_NAMES = feature_names(PROFILES) + [f"l1b_{n}" for n in feature_names(LEVELS)]
OPP_FEATURE_NAMES = [f"opp_{n}" for n in feature_names(LEVELS)]
FUSED_OPP_FEATURE_NAMES = FUSED_FEATURE_NAMES + OPP_FEATURE_NAMES
DIFF_FEATURE_NAMES = ["opp_minus_own_pmi_mean", "opp_pmi_mean", "opp_pmi_std"]
FUSED_OPP_DIFF_FEATURE_NAMES = FUSED_OPP_FEATURE_NAMES + DIFF_FEATURE_NAMES
FUSED_OPP_DIFF_HD_FEATURE_NAMES = FUSED_OPP_DIFF_FEATURE_NAMES + HD_FEATURE_NAMES  # + the handicap question features (issue #47)
FEATURE_SETS = {"fused": FUSED_FEATURE_NAMES, "fused_opp": FUSED_OPP_FEATURE_NAMES, "fused_opp_diff": FUSED_OPP_DIFF_FEATURE_NAMES,
                "fused_opp_diff_hd": FUSED_OPP_DIFF_HD_FEATURE_NAMES}
NAMES = {"fused": NAME, "fused_opp": NAME_OPP, "fused_opp_diff": NAME_OPP_DIFF, "fused_opp_diff_hd": NAME_OPP_DIFF_HD}
DIFF_SETS = ("fused_opp_diff", "fused_opp_diff_hd")
MODEL_KEYS = ("name", "heads_file", "state_sha256", "trunk_file", "trunk_sha256", "train_folds", "feature_folds")


def fused_features(hsl_sums, l1b_sums, n_moves):
    """[40]: question_features of the games' Human SL sums [m, 18], then of their L1b sums [m, 10], both with n_moves [m]."""
    return np.concatenate([question_features(hsl_sums, n_moves), question_features(l1b_sums, n_moves)])


def fused_opp_features(hsl_sums, l1b_sums, n_moves, opp_sums, opp_n_moves):
    """[56]: fused_features, then question_features of the L1b sums over the opponent's moves [m, 10] with the
    opponent's move counts [m] (l1b_opp_check.features' "fused_opp", column for column)."""
    return np.concatenate([fused_features(hsl_sums, l1b_sums, n_moves), question_features(opp_sums, opp_n_moves)])


def diff_features(l1b_sums, opp_sums):
    """[3]: over a question's games, mean(opponent pmi - own pmi), mean(opponent pmi), std(opponent pmi); pmi is
    posterior_mean_index of a game's ten L1b sums [m, 10] (scripts/error_analysis.py's mlp_diff features, column for column)."""
    own = posterior_mean_index(np.ascontiguousarray(l1b_sums, dtype=float))
    opp = posterior_mean_index(np.ascontiguousarray(opp_sums, dtype=float))
    return np.array([(opp - own).mean(), opp.mean(), opp.std()])


def fused_opp_diff_features(hsl_sums, l1b_sums, n_moves, opp_sums, opp_n_moves):
    """[59]: fused_opp_features, then diff_features of the target's and the opponent's L1b sums."""
    return np.concatenate([fused_opp_features(hsl_sums, l1b_sums, n_moves, opp_sums, opp_n_moves), diff_features(l1b_sums, opp_sums)])


def question_diff_features(l1b, opp, questions):
    """[Q, 3]: diff_features of every question from l1b_score.py tables of the target's (l1b) and the opponent's (opp) sums."""
    cols = sum_cols(LEVELS)
    li, oi = l1b.set_index("game_key"), opp.set_index("game_key")
    L, O = li[cols].to_numpy(dtype=float), oi[cols].to_numpy(dtype=float)
    row_l, row_o = {k: i for i, k in enumerate(li.index)}, {k: i for i, k in enumerate(oi.index)}
    rows = [diff_features(L[[row_l[k] for k in ks]], O[[row_o[k] for k in ks]]) for ks in questions["game_keys"]]
    return np.stack(rows) if rows else np.zeros((0, len(DIFF_FEATURE_NAMES)))


def feature_set_of(opponent=False, feature_set=None):
    """The feature set name: `feature_set` when given, else fused_opp / fused from the older opponent flag."""
    if feature_set is None:
        feature_set = "fused_opp" if opponent else "fused"
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {feature_set!r}; one of {sorted(FEATURE_SETS)}")
    return feature_set


def feature_names_for(feature_set):
    """The feature names of a set given by name, or by the older opponent flag (True: fused_opp, False: fused)."""
    if isinstance(feature_set, (bool, np.bool_)):
        feature_set = "fused_opp" if feature_set else "fused"
    return list(FEATURE_SETS[feature_set_of(feature_set=feature_set)])


def l1b_model_record(name, heads_file, sidecar, feature_folds):
    """The document's record of one L1b model: its heads file name, the sidecar's sha256s and training folds
    (aicup.l1b.save_heads), and the folds whose features it supplied."""
    return {"name": name, "heads_file": Path(heads_file).name, "state_sha256": sidecar["state_sha256"],
            "trunk_file": sidecar.get("trunk_file"), "trunk_sha256": sidecar.get("trunk_sha256"),
            "train_folds": sidecar.get("train_folds"), "feature_folds": [int(f) for f in feature_folds]}


def fused_document(mlp, l1b_models, train_folds, per_player, n_train, questions_sha256, extra=None, opponent=False, feature_set=None):
    """The rank_model_fused[_opp[_diff]].json document: name, features (40 / 56 / 59), feature_set, the opponent flag
    (True for every set but "fused"), profiles, levels, training provenance, l1b_models, state."""
    for m in l1b_models:
        missing = [k for k in MODEL_KEYS if k not in m]
        if missing:
            raise ValueError(f"l1b model record {m.get('name')!r} lacks {missing}")
    feature_set = feature_set_of(opponent, feature_set)
    opponent = feature_set != "fused"
    what = ("fused Human SL + L1b question features" + (" + the L1b features of the opponent's moves" if opponent else "")
            + (" + the opponent-minus-own pmi differences" if feature_set in DIFF_SETS else "")
            + (" + the handicap (HD) question features" if feature_set == "fused_opp_diff_hd" else ""))
    return {"name": NAMES[feature_set], "format": FORMAT, "features": feature_names_for(feature_set), "opponent": bool(opponent), "feature_set": feature_set,
            "profiles": list(PROFILES), "levels": list(LEVELS),
            "train_folds": [int(f) for f in train_folds], "per_player": int(per_player), "train_questions": int(n_train),
            "questions_sha256": questions_sha256, "l1b_models": [dict(m) for m in l1b_models],
            "role": f"the submission's decision model: MLP on {what}, averaged over the L1b models",
            **(extra or {}), **environment("cpu"), "state": mlp.state_dict()}


def save_fused_model(doc, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_json(doc, indent=1) + "\n")
    return path


def load_fused_model(path):
    """(document with file_sha256, feature_set and opponent added, MLPModel). The features decide the kind: the 40 fused
    names (opponent False), the 56 fused_opp names or the 59 fused_opp_diff names (opponent True); ValueError when they
    are neither, when the document's own opponent flag or feature_set disagrees, when profiles / levels are not this
    code's, or when it names no L1b model."""
    path = Path(path)
    doc = json.loads(path.read_text())
    feats = doc.get("features")
    feature_set = next((k for k, names in FEATURE_SETS.items() if feats == names), None)
    if feature_set is None:
        sets = [f"{k} ({len(v)})" for k, v in FEATURE_SETS.items()]
        raise ValueError(f"{path}: features are neither this code's {', '.join(sets[:-1])} nor {sets[-1]} features ({len(feats or [])} given)")
    opponent = feature_set != "fused"
    if doc.get("opponent", opponent) != opponent:
        raise ValueError(f"{path}: opponent flag {doc.get('opponent')} disagrees with its {len(feats)} features")
    if doc.get("feature_set", feature_set) != feature_set:
        raise ValueError(f"{path}: feature_set {doc.get('feature_set')!r} disagrees with its {len(feats)} features ({feature_set})")
    if doc.get("profiles") != list(PROFILES) or doc.get("levels") != list(LEVELS):
        raise ValueError(f"{path}: profiles or levels differ from this code's")
    if not doc.get("l1b_models"):
        raise ValueError(f"{path}: l1b_models is empty; the fused model must name the L1b models its features come from")
    model = from_state(doc["state"])
    if not isinstance(model, MLPModel):
        raise ValueError(f"{path}: state is not an MLP")
    return {**doc, "opponent": opponent, "feature_set": feature_set, "file_sha256": file_sha256(path)}, model


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


def fused_predict(mlp, hsl_sums, l1b_sums, n_moves, order, opp_sums=None, opp_n_moves=None, feature_set=None, hd_feats=None):
    """[10]: the mean over the document's models (shard columns `order`) of the MLP's probabilities on the features of
    `feature_set` (default: fused_opp when the opponent's L1b sums [m, K, 10] and move counts [m] are given, else
    fused); fused_opp_diff adds the opponent-minus-own differences, fused_opp_diff_hd also the question's handicap
    features hd_feats [5] (aicup.handicap.question_hd_features_from_games), the same for every model. ValueError when
    the set needs opponent sums or handicap features and none are given."""
    feature_set = feature_set_of(opp_sums is not None, feature_set)
    if feature_set == "fused":
        feats = [fused_features(hsl_sums, l1b_sums[:, k, :], n_moves) for k in order]
    else:
        if opp_sums is None:
            raise ValueError(f"the {feature_set} features need the opponent's L1b sums")
        fn = fused_opp_diff_features if feature_set in DIFF_SETS else fused_opp_features
        feats = [fn(hsl_sums, l1b_sums[:, k, :], n_moves, opp_sums[:, k, :], opp_n_moves) for k in order]
        if feature_set == "fused_opp_diff_hd":
            if hd_feats is None:
                raise ValueError("the fused_opp_diff_hd features need the question's handicap features (hd_feats)")
            hd = np.asarray(hd_feats, dtype=float).reshape(-1)
            if hd.shape != (len(HD_FEATURE_NAMES),):
                raise ValueError(f"hd_feats must hold the {len(HD_FEATURE_NAMES)} handicap features, got shape {hd.shape}")
            feats = [np.concatenate([x, hd]) for x in feats]
    return np.mean([mlp.predict_proba(x[None, :])[0] for x in feats], axis=0)
