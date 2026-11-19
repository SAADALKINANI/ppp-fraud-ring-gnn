import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pppgraph.explain import evidence_trail, relation_attribution  # noqa: E402
from pppgraph.graph import (INTERNAL_RELATIONS, PUBLIC_RELATIONS, build_hetero_graph,  # noqa: E402
                            graph_features, loan_features, node_split)
from pppgraph.models import HGT, HeteroSAGE, train_gnn  # noqa: E402
from pppgraph.rings import detect_rings, evaluate_cases  # noqa: E402
from pppgraph.simulate import SimConfig, simulate  # noqa: E402

ALL = {**PUBLIC_RELATIONS, **INTERNAL_RELATIONS}


@pytest.fixture(scope="module")
def df():
    return simulate(SimConfig(n_legit=1500, n_rings=12, ring_size=(4, 12), n_lone_fraud=30, n_franchise_groups=10,
                              n_registered_agent_addrs=5, seed=1))


def test_simulator_schema_matches_sba_foia(df):
    for col in ["LoanNumber", "DateApproved", "BorrowerAddress", "BorrowerState", "InitialApprovalAmount",
                "JobsReported", "NAICSCode", "BusinessType", "OriginatingLender", "LoanStatus"]:
        assert col in df
    assert df.LoanNumber.is_unique
    assert 0 < df.is_fraud.mean() < 0.3
    assert df.known_fraud.sum() < df.is_fraud.sum()                 # weak labels: only some fraud is known
    assert (df.known_fraud <= df.is_fraud).all()


def test_rings_share_identifiers_more_than_legit(df):
    ring = df[df.typology == "ring"]
    legit = df[df.typology == "legit"]
    share = lambda d: d.groupby("Phone").LoanNumber.transform("count").gt(1).mean()  # noqa: E731
    assert share(ring) > share(legit)


def test_graph_features_have_no_self_label_leakage(df):
    known = df.known_fraud.to_numpy().astype(float)
    X, names = graph_features(df, known, ALL)
    assert np.isfinite(X).all()
    i = int(np.where(known == 1)[0][0])
    solo = df.index[(df.Phone == df.Phone[i])].tolist() == [i]
    if solo:
        assert X[i, names.index("phone_known_fraud_nbrs")] == 0


@pytest.mark.parametrize("ctor", [HeteroSAGE, HGT])
def test_gnn_trains_and_scores_all_loans(df, ctor):
    X, names = loan_features(df)
    gb = build_hetero_graph(df, X, names, ALL)
    split = node_split(df, 0)
    torch.manual_seed(0)
    model, scores, ap = train_gnn(ctor(gb.data.metadata()), gb.data, df.known_fraud.to_numpy(), split, epochs=15)
    assert scores.shape == (len(df),) and np.isfinite(scores).all() and 0 <= scores.min() <= scores.max() <= 1
    attr = relation_attribution(model, gb.data, gb.relations)
    assert set(gb.relations) <= set(attr.columns)


def test_ring_detection_and_evaluation(df):
    scores = df.is_fraud.to_numpy() * 0.9 + 0.05                    # oracle scores -> cases should be pure
    cases = detect_rings(df, scores)
    ev = evaluate_cases(cases, df, top_k=10)
    assert ev["mean_case_purity"] > 0.7 and ev["distinct_true_rings_surfaced"] >= 5


def test_evidence_trail_lists_shared_identifiers(df):
    i = int(df.index[df.typology == "ring"][0])
    ev = evidence_trail(df, i)
    assert all({"relation", "identifier", "other_loans"} <= set(e) for e in ev)
