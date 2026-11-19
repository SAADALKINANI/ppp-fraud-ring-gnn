"""Heterogeneous GNNs for loan-level fraud scoring.

* ``HeteroSAGE`` - relation-specific GraphSAGE convolutions combined with HeteroConv
  (one weight matrix per relation, so 'shares a phone' and 'uses the same lender' are
  learned as different kinds of evidence).
* ``HGT``        - Heterogeneous Graph Transformer (Hu et al., WWW 2020): type-specific
  attention lets the model learn *which* shared identifiers matter for *which* loans.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch_geometric.nn import HeteroConv, HGTConv, Linear, SAGEConv


class HeteroSAGE(nn.Module):
    def __init__(self, metadata, hidden: int = 64, layers: int = 3, dropout: float = 0.2, target: str = "loan"):
        super().__init__()
        self.target = target
        self.inp = nn.ModuleDict({t: Linear(-1, hidden) for t in metadata[0]})
        self.convs = nn.ModuleList(
            HeteroConv({et: SAGEConv((hidden, hidden), hidden) for et in metadata[1]}, aggr="sum")
            for _ in range(layers))
        self.norms = nn.ModuleList(nn.ModuleDict({t: nn.LayerNorm(hidden) for t in metadata[0]}) for _ in range(layers))
        self.drop = dropout
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x_dict, edge_index_dict):
        h = {t: torch.relu(self.inp[t](x)) for t, x in x_dict.items()}
        h0 = h[self.target]
        for conv, norm in zip(self.convs, self.norms):
            out = conv(h, edge_index_dict)
            h = {t: norm[t](torch.relu(out.get(t, torch.zeros_like(h[t]))) + h[t]) for t in h}  # residual
            h = {t: nn.functional.dropout(v, self.drop, self.training) for t, v in h.items()}
        return self.head(torch.cat([h[self.target], h0], 1)).squeeze(-1)


class HGT(nn.Module):
    def __init__(self, metadata, hidden: int = 64, heads: int = 4, layers: int = 2, dropout: float = 0.2,
                 target: str = "loan"):
        super().__init__()
        self.target = target
        self.inp = nn.ModuleDict({t: Linear(-1, hidden) for t in metadata[0]})
        self.convs = nn.ModuleList(HGTConv(hidden, hidden, metadata, heads) for _ in range(layers))
        self.drop = dropout
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x_dict, edge_index_dict):
        h = {t: torch.relu(self.inp[t](x)) for t, x in x_dict.items()}
        h0 = h[self.target]
        for conv in self.convs:
            out = conv(h, edge_index_dict)
            h = {t: nn.functional.dropout(out[t] + h[t], self.drop, self.training) for t in h}
        return self.head(torch.cat([h[self.target], h0], 1)).squeeze(-1)


def with_label_input(data, labels: torch.Tensor, visible: torch.Tensor, target: str = "loan"):
    """Append a 'known fraud' input channel to loan features (the *label trick*,
    Wang et al. 2021). Only nodes in ``visible`` reveal their label; everyone else gets 0."""
    lab = torch.zeros(data[target].num_nodes, 2)
    lab[visible, 0] = labels[visible]
    lab[visible, 1] = 1.0                       # "label is observed" indicator
    x = dict(data.x_dict)
    x[target] = torch.cat([data[target].x, lab], 1)
    return x


def train_gnn(model, data, known: np.ndarray, split: dict, epochs: int = 300,
              lr: float = 3e-3, patience: int = 40, label_trick: bool = True, seed: int = 0, log=None,
              weight_decay: float = 1e-4):
    """Full-batch training with weak (positive-unlabelled) labels.

    Positives = *known* fraud in the train split; all other train loans are treated as
    negatives (they are mostly legitimate, but some are undiscovered fraud - the PU setting
    investigators actually face). With ``label_trick`` each epoch a random half of the train
    nodes reveals its label as an input feature and the loss is computed on the other half,
    so the GNN learns guilt-by-association without seeing its own label.
    Model selection: average precision on the validation split's *known* labels only.
    """
    g = torch.Generator().manual_seed(seed)
    y = torch.from_numpy(known.astype(np.float32))
    tr = torch.from_numpy(split["train"])
    pos = y[tr].sum().clamp(min=1)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=(len(tr) - pos) / pos)
    none = torch.zeros(0, dtype=torch.long)
    full_visible = tr if label_trick else none
    tgt_type = getattr(model, "target", "loan")
    xin = lambda vis: with_label_input(data, y, vis, tgt_type)  # noqa: E731
    with torch.no_grad():
        model(xin(none), data.edge_index_dict)            # initialise lazy layers
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        if label_trick:
            perm = tr[torch.randperm(len(tr), generator=g)]
            vis, tgt = perm[: len(tr) // 2], perm[len(tr) // 2:]
        else:
            vis, tgt = none, tr
        logits = model(xin(vis), data.edge_index_dict)
        loss = loss_fn(logits[tgt], y[tgt]); loss.backward(); opt.step()
        if ep % 5 == 0:
            model.eval()
            with torch.no_grad():
                s = model(xin(full_visible), data.edge_index_dict).numpy()
            ap = average_precision_score(known[split["val"]], s[split["val"]])
            if log:
                log(ep, float(loss), ap)
            if ap > best + 1e-4:
                best, best_state, bad = ap, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 5
                if bad >= patience:
                    break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(xin(full_visible), data.edge_index_dict)).numpy()
    model.inference_inputs = lambda d=data: with_label_input(d, y, full_visible, tgt_type)
    return model, scores, best
