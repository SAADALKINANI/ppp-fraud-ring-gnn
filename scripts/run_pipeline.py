"""PPP fraud-ring detection: simulate -> features -> baselines vs. heterogeneous GNNs ->
ring discovery -> explanations -> real-data validation on the Amazon fraud graph.

    python scripts/run_pipeline.py             # full run (~30 min, 2-core CPU; add --with-hgt on a GPU)
    python scripts/run_pipeline.py --quick     # smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from pppgraph.amazon import load_amazon  # noqa: E402
from pppgraph.explain import evidence_trail, feature_attribution, relation_attribution  # noqa: E402
from pppgraph.graph import (INTERNAL_RELATIONS, PUBLIC_RELATIONS, build_hetero_graph,  # noqa: E402
                            graph_features, loan_features, node_split)
from pppgraph.models import HGT, HeteroSAGE, train_gnn  # noqa: E402
from pppgraph.rings import RING_COLS, detect_rings, evaluate_cases  # noqa: E402
from pppgraph.simulate import SimConfig, simulate  # noqa: E402

FIG = ROOT / "results" / "figures"
log = lambda *a: print(*a, flush=True)  # noqa: E731


def investigator_metrics(y_true: np.ndarray, score: np.ndarray, amount: np.ndarray, typology: np.ndarray) -> dict:
    """Metrics in the units an OIG cares about: precision of a fixed-size work queue and dollars."""
    order = np.argsort(-score)
    k = max(1, int(0.05 * len(score)))          # capacity: investigate the top 5% of loans
    top = order[:k]
    ring, lone = typology == "ring", typology == "lone"
    return {"pr_auc": float(average_precision_score(y_true, score)), "roc_auc": float(roc_auc_score(y_true, score)),
            "precision_at_top5pct": float(y_true[top].mean()),
            "recall_at_top5pct": float(y_true[top].sum() / max(1, y_true.sum())),
            "ring_fraud_recall_at_top5pct": float((y_true[top] & ring[top]).sum() / max(1, ring.sum())),
            "lone_fraud_recall_at_top5pct": float((y_true[top] & lone[top]).sum() / max(1, lone.sum())),
            "fraud_dollars_in_queue": float(amount[top][y_true[top] == 1].sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--with-hgt", action="store_true", help="also train HGT (slow on CPU; use a GPU)")
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True); (ROOT / "artifacts").mkdir(exist_ok=True)
    cfg = SimConfig(n_legit=8000, n_rings=40, n_lone_fraud=120) if args.quick else SimConfig()
    seeds = args.seeds[:1] if args.quick else args.seeds
    ep = 60 if args.quick else 160

    df = simulate(cfg)
    df.to_csv(ROOT / "data" / "simulated_ppp_loans.csv.gz", index=False)
    R: dict = {"data": {"loans": len(df), "fraud_loans": int(df.is_fraud.sum()), "fraud_rate": float(df.is_fraud.mean()),
                        "rings": int(df.ring_id.nunique() - 1), "lone_fraud": int((df.typology == "lone").sum()),
                        "known_fraud_labels": int(df.known_fraud.sum()), "label_rate": cfg.label_rate}}
    log(R["data"])
    Xl, fnames = loan_features(df)
    y_true, known = df.is_fraud.to_numpy(), df.known_fraud.to_numpy()
    amount, typ = df.InitialApprovalAmount.to_numpy(), df.typology.to_numpy()
    ALL = {**PUBLIC_RELATIONS, **INTERNAL_RELATIONS}

    results = {}
    cache = ROOT / "artifacts" / "cache"; cache.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        split = node_split(df, seed)
        ck = cache / f"seed{seed}.json"
        if ck.exists() and seed != seeds[0]:                 # restart-safe: reuse finished seeds
            results[seed] = json.loads(ck.read_text()); log(f"seed {seed}: loaded from cache"); continue
        te = split["test"]
        train_known = np.zeros(len(df)); train_known[split["train"]] = known[split["train"]]
        res = {}

        def gbm(X, name):
            tr, va = split["train"], split["val"]
            m = xgb.XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8,
                                  colsample_bytree=0.8, tree_method="hist", n_jobs=2, early_stopping_rounds=40,
                                  scale_pos_weight=(len(tr) - known[tr].sum()) / max(1, known[tr].sum()), eval_metric="aucpr")
            m.fit(X[tr], known[tr], eval_set=[(X[va], known[va])], verbose=False)
            s = m.predict_proba(X)[:, 1]
            res[name] = investigator_metrics(y_true[te], s[te], amount[te], typ[te])
            log(f"  seed {seed} {name:28s} PR-AUC {res[name]['pr_auc']:.3f}  P@5% {res[name]['precision_at_top5pct']:.3f}")
            return s

        gbm(Xl, "xgb_application_only")
        Xg_pub, _ = graph_features(df, train_known, PUBLIC_RELATIONS)
        gbm(np.hstack([Xl, Xg_pub]), "xgb_+graph_feats_public")
        Xg_all, gnames = graph_features(df, train_known, ALL)
        s_xgb_all = gbm(np.hstack([Xl, Xg_all]), "xgb_+graph_feats_all")

        for rels, tag in [(PUBLIC_RELATIONS, "public"), (ALL, "all")]:
            gb = build_hetero_graph(df, Xl, fnames, rels)
            gnns = [("hetero_sage", HeteroSAGE)] + ([("hgt", HGT)] if args.with_hgt else [])
            for name, ctor in gnns:
                if (tag == "public" and name == "hgt") or (name == "hgt" and seed != seeds[0]):
                    continue                          # HGT: optional (slow on CPU); SAGE is the deployed model
                torch.manual_seed(seed)
                t0 = time.time()
                model = ctor(gb.data.metadata())
                model, s, val_ap = train_gnn(model, gb.data, known, split, epochs=ep, seed=seed)
                key = f"{name}_{tag}"
                res[key] = investigator_metrics(y_true[te], s[te], amount[te], typ[te]) | {"train_seconds": time.time() - t0}
                log(f"  seed {seed} {key:28s} PR-AUC {res[key]['pr_auc']:.3f}  P@5% {res[key]['precision_at_top5pct']:.3f}  ({time.time() - t0:.0f}s)")
                if seed == seeds[0] and key == "hetero_sage_all":
                    keep = (model, gb, s, split)
        results[seed] = res
        ck.write_text(json.dumps(res, default=float))

    names = list(results[seeds[0]])
    have = lambda n: [s for s in seeds if n in results[s]]  # noqa: E731
    R["models"] = {n: {k: float(np.mean([results[s][n][k] for s in have(n)])) for k in results[seeds[0]][n]} |
                   {f"{k}_std": float(np.std([results[s][n][k] for s in have(n)])) for k in ("pr_auc", "precision_at_top5pct")} |
                   {"n_seeds": len(have(n))} for n in names}
    R["seeds"] = seeds

    # -------- Ring discovery (uses the deployed GNN's scores) --------
    model, gb, s, split = keep
    t0 = time.time()
    cases = detect_rings(df, s)
    R["ring_discovery"] = {"gnn_ranked": evaluate_cases(cases, df, 50),
                           "size_ranked_baseline": evaluate_cases(cases.sort_values("size", ascending=False), df, 50),
                           "n_candidate_cases": int(len(cases)), "seconds": time.time() - t0}
    log("rings", R["ring_discovery"])
    cases.head(100).assign(members=lambda d: d.members.map(lambda m: [df.LoanNumber[i] for i in m])).to_json(
        ROOT / "results" / "top_cases.json", orient="records", indent=1)

    # -------- Explanations --------
    rel_attr = relation_attribution(model, gb.data, gb.relations)
    te = split["test"]
    top_test = te[np.argsort(-s[te])[:200]]
    R["relation_importance_top200"] = rel_attr.loc[top_test, gb.relations].clip(lower=0).mean().round(4).to_dict()
    ex_idx = int(top_test[0])
    R["example_explanation"] = {
        "loan_number": df.LoanNumber[ex_idx], "score": float(s[ex_idx]), "truly_fraud (hidden)": int(y_true[ex_idx]),
        "relation_attribution": rel_attr.loc[ex_idx, gb.relations].round(4).to_dict(),
        "top_application_features": feature_attribution(model, gb.data, ex_idx, gb.feature_names),
        "evidence_trail": evidence_trail(df, ex_idx)}
    log("explanation", json.dumps(R["example_explanation"], default=float)[:800])
    torch.save({"state_dict": model.state_dict(), "metadata": gb.data.metadata()}, ROOT / "artifacts" / "hetero_sage_all.pt")
    pd.DataFrame({"LoanNumber": df.LoanNumber, "score": s, "split": np.select(
        [np.isin(np.arange(len(df)), split["test"]), np.isin(np.arange(len(df)), split["val"])], ["test", "val"], "train")}
    ).to_csv(ROOT / "results" / "loan_scores.csv.gz", index=False)

    # -------- Real-data validation: Amazon fraud graph --------
    R["amazon_real_data"] = amazon_benchmark(seeds, quick=args.quick)
    plots(R)
    (ROOT / "results" / "metrics.json").write_text(json.dumps(R, indent=2, default=float))


def amazon_benchmark(seeds, quick=False) -> dict:
    data, y, lab = load_amazon(ROOT / "data" / "raw")
    X = data["user"].x.numpy()
    out = {"nodes": int(len(y)), "labelled": int(len(lab)), "fraud_rate": float(y[lab].mean()), "runs": {}}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        pos, neg = lab[y[lab] == 1], lab[y[lab] == 0]
        sp = {}
        for arr in (rng.permutation(pos), rng.permutation(neg)):          # stratified 40/20/40 (CARE-GNN protocol)
            a, b = int(.4 * len(arr)), int(.6 * len(arr))
            for k, part in (("train", arr[:a]), ("val", arr[a:b]), ("test", arr[b:])):
                sp[k] = np.concatenate([sp.get(k, np.array([], int)), part])
        res = {}
        m = xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=.8, colsample_bytree=.8,
                              n_jobs=2, early_stopping_rounds=40, eval_metric="aucpr")
        m.fit(X[sp["train"]], y[sp["train"]], eval_set=[(X[sp["val"]], y[sp["val"]])], verbose=False)
        p = m.predict_proba(X[sp["test"]])[:, 1]
        res["xgboost_features_only"] = {"roc_auc": roc_auc_score(y[sp["test"]], p), "pr_auc": average_precision_score(y[sp["test"]], p)}
        for name, ctor in [("hetero_sage_relation_aware", HeteroSAGE)]:
            torch.manual_seed(seed)
            model = ctor(data.metadata(), target="user", **({"layers": 2} if name.startswith("hetero") else {}))
            yy = np.zeros(len(y)); yy[lab] = y[lab]
            _, s, _ = train_gnn(model, data, yy, sp, epochs=60 if quick else 200, label_trick=False, seed=seed, lr=5e-3)
            res[name] = {"roc_auc": roc_auc_score(y[sp["test"]], s[sp["test"]]), "pr_auc": average_precision_score(y[sp["test"]], s[sp["test"]])}
        out["runs"][seed] = res
        log("amazon", seed, {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in res.items()})
    names = list(out["runs"][seeds[0]])
    out["summary"] = {n: {k: float(np.mean([out["runs"][s][n][k] for s in seeds])) for k in ("roc_auc", "pr_auc")} |
                      {f"{k}_std": float(np.std([out["runs"][s][n][k] for s in seeds])) for k in ("roc_auc", "pr_auc")}
                      for n in names}
    out["published_reference"] = "CARE-GNN (Dou et al., CIKM 2020) reports AUC 89.73 on Amazon with 40% training data"
    return out


def plots(R):
    m = R["models"]
    order = ["xgb_application_only", "xgb_+graph_feats_public", "hetero_sage_public",
             "xgb_+graph_feats_all", "hetero_sage_all", "hgt_all"]
    order = [o for o in order if o in m]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    cols = ["#aaa" if o.startswith("xgb") else "#1f4e9c" for o in order]
    ax[0].barh(order[::-1], [m[o]["pr_auc"] for o in order][::-1], color=cols[::-1],
               xerr=[m[o]["pr_auc_std"] for o in order][::-1])
    ax[0].set_xlabel("PR-AUC (test loans, all true fraud)"); ax[0].set_title("Fraud ranking quality")
    ax[1].barh(order[::-1], [m[o]["fraud_dollars_in_queue"] / 1e6 for o in order][::-1], color=cols[::-1])
    ax[1].set_xlabel("$M of fraudulent loans in the top-5% work queue"); ax[1].set_title("Dollars an investigator would reach")
    fig.tight_layout(); fig.savefig(FIG / "model_comparison.png", dpi=140); plt.close(fig)

    ri = R["relation_importance_top200"]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    k = sorted(ri, key=ri.get)
    ax.barh(k, [ri[x] for x in k], color="#1f4e9c")
    ax.set_xlabel("Mean score drop when relation removed"); ax.set_title("Which shared identifiers drive the top-200 alerts")
    fig.tight_layout(); fig.savefig(FIG / "relation_importance.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
