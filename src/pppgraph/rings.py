"""From loan scores to *cases*: fraud-ring discovery and investigator-centric metrics.

Investigators do not work loan-by-loan; they open cases on networks. We project the
heterogeneous graph to a weighted loan-loan graph (two loans are linked by every identifier
they share, ignoring mega-hubs such as national lenders and registered-agent addresses),
run Louvain community detection, and rank communities by the GNN's fraud scores.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

RING_COLS = ["BorrowerAddress", "Phone", "Email", "BankAccount", "IPAddress", "Preparer"]
WEIGHTS = {"BorrowerAddress": 1.0, "Phone": 1.5, "Email": 1.5, "BankAccount": 3.0, "IPAddress": 1.0, "Preparer": 0.5}


def loan_projection(df: pd.DataFrame, cols=RING_COLS, max_hub: int = 60) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(df.index)
    for col in cols:
        if col not in df:
            continue
        groups = df[df[col] != "SELF"].groupby(col).groups
        for _, members in groups.items():
            m = list(members)
            if len(m) < 2 or len(m) > max_hub:
                continue
            w = WEIGHTS.get(col, 1.0)
            for i in range(len(m)):            # star + chain keeps edges O(n) for big groups
                for j in ([i + 1] if len(m) > 12 else range(i + 1, len(m))):
                    if j < len(m):
                        a, b = m[i], m[j]
                        G.add_edge(a, b, weight=G[a][b]["weight"] + w if G.has_edge(a, b) else w)
    return G


def detect_rings(df: pd.DataFrame, scores: np.ndarray, min_size: int = 3, seed: int = 0, cols=RING_COLS) -> pd.DataFrame:
    G = loan_projection(df, cols)
    G.remove_nodes_from([n for n, d in dict(G.degree()).items() if d == 0])
    comms = nx.community.louvain_communities(G, weight="weight", resolution=1.0, seed=seed)
    rows = []
    for cid, c in enumerate(comms):
        if len(c) < min_size:
            continue
        idx = np.fromiter(c, int)
        sub = df.loc[idx]
        rows.append({"case_id": f"CASE-{cid:04d}", "members": idx.tolist(), "size": len(idx),
                     "mean_score": float(scores[idx].mean()), "max_score": float(scores[idx].max()),
                     "total_amount": float(sub["InitialApprovalAmount"].sum()),
                     "lenders": sub["OriginatingLender"].nunique(), "states": sub["BorrowerState"].nunique(),
                     "known_fraud_members": int(sub["known_fraud"].sum())})
    cases = pd.DataFrame(rows)
    if cases.empty:
        return cases
    cases["case_score"] = cases["mean_score"] * np.log1p(cases["size"])
    return cases.sort_values("case_score", ascending=False).reset_index(drop=True)


def evaluate_cases(cases: pd.DataFrame, df: pd.DataFrame, top_k: int = 50) -> dict:
    """Ground-truth evaluation (simulation only): are the top cases real rings, and how many
    of the true rings does the case list surface?"""
    top = cases.head(top_k)
    purity = [df.loc[m, "is_fraud"].mean() for m in top["members"]]
    found_rings = set()
    for m in top["members"]:
        r = df.loc[m, "ring_id"]
        r = r[r >= 0]
        if len(r):
            dom, cnt = r.value_counts().index[0], r.value_counts().iloc[0]
            if cnt / len(m) >= 0.5:
                found_rings.add(int(dom))
    n_rings = int(df.loc[df.ring_id >= 0, "ring_id"].nunique())
    fraud_amt = df.loc[df.is_fraud == 1, "InitialApprovalAmount"].sum()
    flagged = np.unique(np.concatenate(top["members"].to_list())) if len(top) else np.array([], int)
    return {"top_k_cases": int(len(top)), "mean_case_purity": float(np.mean(purity)) if purity else 0.0,
            "cases_majority_fraud": int(sum(p >= 0.5 for p in purity)),
            "distinct_true_rings_surfaced": len(found_rings), "total_true_rings": n_rings,
            "fraud_dollars_in_top_cases": float(df.loc[flagged].query("is_fraud == 1")["InitialApprovalAmount"].sum()),
            "share_of_all_fraud_dollars": float(df.loc[flagged].query("is_fraud == 1")["InitialApprovalAmount"].sum() / fraud_amt)}
