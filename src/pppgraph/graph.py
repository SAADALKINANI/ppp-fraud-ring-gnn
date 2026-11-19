"""Feature engineering and heterogeneous graph construction.

Graph schema (node types / relations):

    loan --at_address--> address        (PUBLIC: FOIA file)
    loan --via_lender--> lender         (PUBLIC)
    loan --has_phone--> phone           (INTERNAL: application data)
    loan --has_email--> email           (INTERNAL)
    loan --pays_to--> account           (INTERNAL: disbursement bank account)
    loan --from_ip--> ip                (INTERNAL: submission device/IP)
    loan --filed_by--> preparer         (INTERNAL: agent / preparer)

Every relation also gets a reverse edge so messages flow loan -> entity -> loan (2 hops = "loans
sharing an identifier"; 4 hops = the ring around them).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from .simulate import FINTECH, NAICS, SOLE_PROP_MAX

PUBLIC_RELATIONS = {"address": "BorrowerAddress", "lender": "OriginatingLender"}
INTERNAL_RELATIONS = {"phone": "Phone", "email": "Email", "account": "BankAccount",
                      "ip": "IPAddress", "preparer": "Preparer"}
SOLO_TYPES = {"Sole Proprietorship", "Self-Employed Individuals", "Independent Contractors"}


def loan_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Application-level features available in the public FOIA file."""
    amt = df["InitialApprovalAmount"].to_numpy(float)
    jobs = df["JobsReported"].clip(lower=1).to_numpy(float)
    date = pd.to_datetime(df["DateApproved"])
    same_day_lender = df.groupby(["OriginatingLender", "DateApproved"])["LoanNumber"].transform("count")
    lender_vol = df.groupby("OriginatingLender")["LoanNumber"].transform("count")
    feats = {
        "log_amount": np.log1p(amt),
        "log_jobs": np.log1p(jobs),
        "log_amount_per_job": np.log1p(amt / jobs),
        "is_solo_type": df["BusinessType"].isin(SOLO_TYPES).astype(float),
        "at_solo_max": (np.abs(amt - SOLE_PROP_MAX) < 1).astype(float),
        "round_thousand": ((amt % 1000) == 0).astype(float),
        "round_jobs": ((jobs >= 10) & (jobs % 5 == 0)).astype(float),
        "new_business": (df["BusinessAgeDescription"] != "Existing or more than 2 years old").astype(float),
        "charged_off": (df["LoanStatus"] == "Charged Off").astype(float),
        "fintech_lender": df["OriginatingLender"].isin(FINTECH).astype(float),
        "day_index": ((date - date.min()).dt.days / 420.0).to_numpy(),
        "log_lender_same_day": np.log1p(same_day_lender) - np.log1p(lender_vol / 420),   # burstiness
    }
    for code in NAICS:
        feats[f"naics_{code}"] = (df["NAICSCode"] == code).astype(float)
    for bt in sorted(df["BusinessType"].unique()):
        feats[f"btype_{bt[:12]}"] = (df["BusinessType"] == bt).astype(float)
    names = list(feats)
    X = np.stack([np.asarray(feats[k], dtype=np.float32) for k in names], 1)
    return X, names


def graph_features(df: pd.DataFrame, train_known: np.ndarray, relations: dict[str, str]) -> tuple[np.ndarray, list[str]]:
    """Hand-engineered network features for the strong non-GNN baseline.

    For each relation: how many other loans share this identifier, and how many of those are
    *known* fraud in the training labels (guilt-by-association). Known-fraud counts exclude
    the loan itself, so there is no label leakage.
    """
    out, names = [], []
    for rel, col in relations.items():
        key = df[col].where(df[col] != "SELF")
        cnt = np.clip(key.map(key.value_counts()).fillna(1).to_numpy() - 1, 0, None)
        kf = np.where(key.notna(), pd.Series(train_known, index=df.index).groupby(key).transform("sum").fillna(0).to_numpy() - train_known, 0.0)
        out += [np.log1p(cnt), np.log1p(kf), (kf > 0).astype(float)]
        names += [f"{rel}_shared", f"{rel}_known_fraud_nbrs", f"{rel}_any_known_fraud"]
    return np.stack(out, 1).astype(np.float32), names


@dataclass
class GraphBundle:
    data: HeteroData
    feature_names: list[str]
    relations: list[str]
    entity_index: dict[str, pd.Index]


def build_hetero_graph(df: pd.DataFrame, X: np.ndarray, feature_names: list[str], relations: dict[str, str],
                       max_entity_degree: int | None = None) -> GraphBundle:
    data = HeteroData()
    data["loan"].x = torch.from_numpy(X)
    data["loan"].num_nodes = len(df)
    ent_index = {}
    for rel, col in relations.items():
        vals = df[col].where(df[col] != "SELF")
        mask = vals.notna().to_numpy()
        codes, uniques = pd.factorize(vals[mask])
        ent_index[rel] = pd.Index(uniques)
        deg = np.bincount(codes, minlength=len(uniques)).astype(np.float32)
        feats = [np.log1p(deg), (deg > 1).astype(np.float32)]
        if rel == "lender":
            feats.append(np.array([u in FINTECH for u in uniques], np.float32))
        else:
            feats.append(np.zeros(len(uniques), np.float32))
        data[rel].x = torch.from_numpy(np.stack(feats, 1))
        src = torch.from_numpy(np.where(mask)[0])
        dst = torch.from_numpy(codes)
        if max_entity_degree:     # optional pruning of mega-hubs
            keep = torch.from_numpy(deg[codes] <= max_entity_degree)
            src, dst = src[keep], dst[keep]
        data["loan", f"to_{rel}", rel].edge_index = torch.stack([src, dst])
        data[rel, f"rev_{rel}", "loan"].edge_index = torch.stack([dst, src])
    return GraphBundle(data, feature_names, list(relations), ent_index)


def node_split(df: pd.DataFrame, seed: int = 0, frac=(0.6, 0.2, 0.2)) -> dict[str, np.ndarray]:
    """Random transductive split of loan nodes, stratified on the (hidden) fraud flag."""
    rng = np.random.default_rng(seed)
    idx = {"train": [], "val": [], "test": []}
    for _, g in df.groupby("is_fraud"):
        p = rng.permutation(g.index.to_numpy())
        a, b = int(frac[0] * len(p)), int((frac[0] + frac[1]) * len(p))
        idx["train"] += list(p[:a]); idx["val"] += list(p[a:b]); idx["test"] += list(p[b:])
    return {k: np.sort(np.array(v)) for k, v in idx.items()}
