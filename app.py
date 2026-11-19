"""Fraud-ring case explorer.  Run:  streamlit run app.py   (after scripts/run_pipeline.py)"""
import json
import sys
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from pppgraph.explain import evidence_trail  # noqa: E402
from pppgraph.rings import RING_COLS  # noqa: E402

st.set_page_config(page_title="PPP Fraud Ring Explorer", layout="wide")


@st.cache_data
def load():
    df = pd.read_csv(ROOT / "data" / "simulated_ppp_loans.csv.gz", dtype={"LoanNumber": str, "NAICSCode": str, "BorrowerZip": str})
    sc = pd.read_csv(ROOT / "results" / "loan_scores.csv.gz", dtype={"LoanNumber": str})
    df = df.merge(sc, on="LoanNumber")
    cases = json.loads((ROOT / "results" / "top_cases.json").read_text())
    return df, cases


df, cases = load()
M = json.loads((ROOT / "results" / "metrics.json").read_text())
st.title("PPP Fraud Ring Explorer - heterogeneous GNN + community detection")
st.caption("SIMULATED loans in the SBA PPP FOIA schema with planted fraud rings (see README). Scores from the deployed HeteroSAGE model.")
m = M["models"]["hetero_sage_all"]; b = M["models"]["xgb_+graph_feats_all"]; rd = M["ring_discovery"]["gnn_ranked"]
c = st.columns(4)
c[0].metric("PR-AUC (GNN vs XGBoost+graph feats)", f"{m['pr_auc']:.3f}", f"{m['pr_auc'] - b['pr_auc']:+.3f}")
c[1].metric("Precision of top-5% work queue", f"{m['precision_at_top5pct']:.0%}")
c[2].metric("Top-50 cases that are real rings", f"{rd['cases_majority_fraud']}/{rd['top_k_cases']}")
c[3].metric("Share of fraud $ in top-50 cases", f"{rd['share_of_all_fraud_dollars']:.0%}")

st.subheader("Case queue (communities ranked by GNN risk)")
cq = pd.DataFrame(cases)[["case_id", "size", "mean_score", "total_amount", "lenders", "states", "known_fraud_members"]]
st.dataframe(cq.head(30), width="stretch", height=280)
cid = st.selectbox("Open case", cq.case_id.head(30))
case = next(x for x in cases if x["case_id"] == cid)
sub = df[df.LoanNumber.isin(case["members"])]

G = nx.Graph()
for _, r in sub.iterrows():
    G.add_node(r.LoanNumber, kind="loan", score=r.score, fraud=r.is_fraud)
    for col in RING_COLS:
        if r[col] != "SELF" and (sub[col] == r[col]).sum() > 1:
            G.add_node(r[col], kind=col); G.add_edge(r.LoanNumber, r[col])
pos = nx.spring_layout(G, seed=1)
ex, ey = [], []
for u, v in G.edges():
    ex += [pos[u][0], pos[v][0], None]; ey += [pos[u][1], pos[v][1], None]
fig = go.Figure(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="#bbb", width=1), hoverinfo="skip"))
loans = [n for n, d in G.nodes(data=True) if d["kind"] == "loan"]
ents = [n for n, d in G.nodes(data=True) if d["kind"] != "loan"]
fig.add_trace(go.Scatter(x=[pos[n][0] for n in loans], y=[pos[n][1] for n in loans], mode="markers", name="loan",
                         marker=dict(size=12, color=[G.nodes[n]["score"] for n in loans], colorscale="Reds", cmin=0, cmax=1,
                                     colorbar=dict(title="risk")), text=loans))
fig.add_trace(go.Scatter(x=[pos[n][0] for n in ents], y=[pos[n][1] for n in ents], mode="markers", name="shared identifier",
                         marker=dict(size=8, symbol="diamond", color="#1f4e9c"), text=[f"{G.nodes[n]['kind']}: {n}" for n in ents]))
fig.update_layout(height=520, showlegend=True, xaxis=dict(visible=False), yaxis=dict(visible=False))
left, right = st.columns([3, 2])
left.plotly_chart(fig, width="stretch")
with right:
    st.write(f"**{len(sub)} loans · ${sub.InitialApprovalAmount.sum():,.0f}** · ground truth (hidden from model): "
             f"{int(sub.is_fraud.sum())} fraudulent")
    top = sub.sort_values("score", ascending=False).iloc[0]
    st.write(f"Highest-risk loan **{top.LoanNumber}** (score {top.score:.2f}) - evidence trail:")
    st.dataframe(pd.DataFrame(evidence_trail(df, int(top.name))), width="stretch")
st.image(str(ROOT / "results" / "figures" / "model_comparison.png"))
