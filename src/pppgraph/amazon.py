"""Real-data validation on the Amazon fraudulent-reviewer graph (McAuley & Leskovec, 2013),
as packaged by CARE-GNN (Dou et al., CIKM 2020): 11,944 users, 25 behavioural features,
three relations (U-P-U: reviewed the same product; U-S-U: same star rating within a week;
U-V-U: top-5% mutual review-text similarity). This is a standard public benchmark for
relation-aware graph fraud detection - the same modelling problem as PPP fraud rings
(fraudsters camouflage individually but co-occur structurally).
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from torch_geometric.data import HeteroData

RELATIONS = {"net_upu": "same_product", "net_usu": "same_star_week", "net_uvu": "similar_text"}


def load_amazon(data_dir: str | Path) -> tuple[HeteroData, np.ndarray, np.ndarray]:
    data_dir = Path(data_dir)
    mat_path = data_dir / "Amazon.mat"
    if not mat_path.exists():
        zipfile.ZipFile(data_dir / "Amazon.zip").extractall(data_dir)
    m = sio.loadmat(mat_path)
    X = np.asarray(m["features"].todense(), dtype=np.float32)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    y = m["label"].ravel().astype(int)
    data = HeteroData()
    data["user"].x = torch.from_numpy(X)
    for key, name in RELATIONS.items():
        a = m[key].tocoo()
        data["user", name, "user"].edge_index = torch.from_numpy(np.vstack([a.row, a.col]).astype(np.int64))
    labelled = np.arange(3305, len(y))   # first 3,305 users are unlabelled in this release (CARE-GNN convention)
    return data, y, labelled
