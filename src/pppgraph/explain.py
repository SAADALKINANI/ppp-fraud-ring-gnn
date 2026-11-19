"""Explanations an investigator (and a court) can follow.

1. **Relation attribution** - remove one relation type (e.g. all shared phones) from the
   graph and measure how much each loan's score drops. One forward pass per relation scores
   every loan at once.
2. **Evidence trail** - the concrete shared identifiers linking a loan to other loans,
   with which of those are already-confirmed fraud.
3. **Feature attribution** - gradient x input on the loan's own application features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .graph import INTERNAL_RELATIONS, PUBLIC_RELATIONS


@torch.no_grad()
def relation_attribution(model, data, relations: list[str]) -> pd.DataFrame:
    x = model.inference_inputs(data)
    base = torch.sigmoid(model(x, data.edge_index_dict)).numpy()
    out = {}
    for rel in relations:
        ei = {k: v for k, v in data.edge_index_dict.items() if rel not in (k[0], k[2])}
        ei.update({k: torch.zeros(2, 0, dtype=torch.long) for k in data.edge_index_dict if rel in (k[0], k[2])})
        out[rel] = base - torch.sigmoid(model(x, ei)).numpy()
    return pd.DataFrame(out).assign(score=base)


def feature_attribution(model, data, loan_idx: int, feature_names: list[str], top_k: int = 6) -> list[tuple[str, float]]:
    x = {k: v.clone() for k, v in model.inference_inputs(data).items()}
    x["loan"].requires_grad_(True)
    s = model(x, data.edge_index_dict)[loan_idx]
    g, = torch.autograd.grad(s, x["loan"])
    contrib = (g * x["loan"])[loan_idx].detach().numpy()[: len(feature_names)]
    order = np.argsort(-np.abs(contrib))[:top_k]
    return [(feature_names[i], round(float(contrib[i]), 4)) for i in order]


def evidence_trail(df: pd.DataFrame, loan_idx: int, max_items: int = 8) -> list[dict]:
    row = df.loc[loan_idx]
    items = []
    for rel, col in {**PUBLIC_RELATIONS, **INTERNAL_RELATIONS}.items():
        if rel == "lender" or col not in df or row[col] == "SELF":
            continue
        others = df[(df[col] == row[col]) & (df.index != loan_idx)]
        if 0 < len(others) <= 200:
            items.append({"relation": rel, "identifier": row[col], "other_loans": int(len(others)),
                          "confirmed_fraud_among_them": int(others["known_fraud"].sum()),
                          "total_amount_linked": float(others["InitialApprovalAmount"].sum())})
    return sorted(items, key=lambda d: (-d["confirmed_fraud_among_them"], -d["other_loans"]))[:max_items]
