"""Realistic PPP loan simulator with planted fraud rings.

Why simulate? The real SBA PPP FOIA file (data.sba.gov) has no fraud labels, and the
"internal" signals investigators actually use (phone, e-mail, bank account, device/IP) are
not public. A simulator lets us (1) know the ground truth for every loan, (2) measure what
public-only data can and cannot catch, and (3) generate realistic *hard negatives*.

The output uses the **exact column names of the SBA PPP FOIA file**, so
``pppgraph.data.load_sba_ppp`` can swap in the real file without code changes.

Fraud typologies are modelled on patterns described in DOJ PPP fraud prosecutions and
SBA OIG reports (e.g. OIG Report 23-09):
  * Recruiter / "loan mill" rings - one organiser submits many applications for recruits,
    re-using a handful of addresses, phones, e-mails and bank accounts, often through the
    same lender in short bursts.
  * Identity / address farms - many shell "businesses" at one address or PO box.
  * Payroll inflation - jobs and amounts at program maxima, round numbers, sole proprietors
    claiming the $20,833 maximum, recently formed entities.
  * Lone actors - individual inflated applications with no network signal (the hard case).

Hard negatives (legitimate patterns that look suspicious):
  * Registered-agent / coworking addresses shared by hundreds of real firms.
  * Franchise groups with one owner, one accountant and many locations.
  * Legitimate tax preparers who file for many unrelated clients.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

STATES = ["CA", "TX", "FL", "NY", "IL", "GA", "PA", "OH", "NC", "MI", "NJ", "VA", "WA", "AZ", "MD", "LA"]
STATE_W = np.array([12, 9, 7, 6, 4, 4, 4, 3.5, 3.2, 3, 2.8, 2.6, 2.3, 2.2, 1.9, 1.5])
NAICS = {  # code: (description, typical jobs, avg annual wage)
    "722511": ("Full-Service Restaurants", 18, 24000), "238220": ("Plumbing & HVAC Contractors", 9, 52000),
    "541110": ("Offices of Lawyers", 6, 85000), "621111": ("Offices of Physicians", 11, 70000),
    "812112": ("Beauty Salons", 4, 26000), "485310": ("Taxi Service", 1, 30000),
    "484121": ("General Freight Trucking", 6, 48000), "531210": ("Offices of Real Estate Agents", 2, 60000),
    "812990": ("All Other Personal Services", 1, 28000), "561720": ("Janitorial Services", 7, 27000),
    "541990": ("Other Professional Services", 2, 55000), "448140": ("Family Clothing Stores", 5, 25000),
    "236118": ("Residential Remodelers", 5, 50000), "611620": ("Sports & Recreation Instruction", 2, 30000),
}
FRAUD_NAICS_BIAS = ["812990", "485310", "484121", "531210", "541990", "611620"]  # over-represented in OIG cases
BUSINESS_TYPES = ["Corporation", "Limited Liability Company(LLC)", "Sole Proprietorship",
                  "Subchapter S Corporation", "Self-Employed Individuals", "Independent Contractors"]
LENDERS = ([f"Community Bank {i}" for i in range(60)] + [f"Regional Bank {i}" for i in range(15)]
           + ["National Bank A", "National Bank B", "National Bank C"]
           + [f"FinTech Lender {c}" for c in "ABCDEF"])
FINTECH = [name for name in LENDERS if name.startswith("FinTech")]
SOLE_PROP_MAX = 20833


@dataclass
class SimConfig:
    n_legit: int = 45_000
    n_rings: int = 160
    ring_size: tuple[int, int] = (4, 40)
    n_lone_fraud: int = 600
    n_franchise_groups: int = 120
    n_registered_agent_addrs: int = 40
    label_rate: float = 0.25        # share of fraud loans "known" (prosecuted/confirmed) at training time
    seed: int = 42


def simulate(cfg: SimConfig = SimConfig()) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    rows: list[dict] = []
    loan_no = iter(range(1_000_000_000, 9_999_999_999))
    counters = {"addr": 0, "phone": 0, "email": 0, "acct": 0, "prep": 0, "ip": 0}

    def new(kind):
        counters[kind] += 1
        return f"{kind.upper()}-{counters[kind]:07d}"

    def base_loan(state=None, naics=None, lender=None, date=None):
        state = state or rng.choice(STATES, p=STATE_W / STATE_W.sum())
        naics = naics or rng.choice(list(NAICS))
        _, jobs_mu, wage = NAICS[naics]
        btype = rng.choice(BUSINESS_TYPES, p=[.2, .3, .17, .15, .1, .08])
        if btype in ("Sole Proprietorship", "Self-Employed Individuals", "Independent Contractors"):
            jobs = 1
            amount = min(SOLE_PROP_MAX, wage * rng.uniform(0.3, 1.1) * 2.5 / 12)
        else:
            jobs = max(1, int(rng.lognormal(np.log(jobs_mu), 0.8)))
            amount = jobs * wage * rng.uniform(0.6, 1.2) * 2.5 / 12
        lender = lender or rng.choice(LENDERS, p=_lender_weights())
        date = date or pd.Timestamp("2020-04-03") + pd.Timedelta(days=int(rng.choice(_date_days(), p=_date_p())))
        return {
            "LoanNumber": str(next(loan_no)), "DateApproved": date.date().isoformat(),
            "BorrowerName": f"BIZ {rng.integers(1e8):08d}", "BorrowerAddress": new("addr"),
            "BorrowerState": state, "BorrowerZip": f"{rng.integers(10000, 99999)}",
            "InitialApprovalAmount": float(np.round(amount, 2)), "JobsReported": int(jobs),
            "NAICSCode": naics, "BusinessType": btype, "OriginatingLender": lender,
            "BusinessAgeDescription": rng.choice(["Existing or more than 2 years old", "New Business or 2 years or less",
                                                  "Startup, Loan Funds will Open Business"], p=[.82, .15, .03]),
            "LoanStatus": rng.choice(["Paid in Full", "Exemption 4", "Charged Off"], p=[.91, .07, .02]),
            # internal-only linkage signals (not in the public FOIA file)
            "Phone": new("phone"), "Email": new("email"), "BankAccount": new("acct"),
            "Preparer": "SELF", "IPAddress": new("ip"),
            "is_fraud": 0, "ring_id": -1, "typology": "legit",
        }

    # ---------------- legitimate borrowers ----------------
    for _ in range(cfg.n_legit):
        rows.append(base_loan())

    # legitimate tax preparers serving many unrelated clients (hard negative for 'Preparer')
    legit_preparers = [new("prep") for _ in range(300)]
    for r in rows:
        if rng.random() < 0.35:
            r["Preparer"] = rng.choice(legit_preparers)

    # registered-agent / coworking addresses: hundreds of legit firms share them (hard negative)
    ra_addrs = [new("addr") for _ in range(cfg.n_registered_agent_addrs)]
    for r in rows:
        if rng.random() < 0.04:
            r["BorrowerAddress"] = rng.choice(ra_addrs)

    # franchise groups: one owner -> shared phone/email/preparer across several locations (hard negative)
    legit_idx = np.arange(len(rows))
    for _ in range(cfg.n_franchise_groups):
        members = rng.choice(legit_idx, size=rng.integers(3, 12), replace=False)
        phone, email, prep, naics = new("phone"), new("email"), new("prep"), rng.choice(["722511", "812112", "448140"])
        for m in members:
            rows[m].update(Phone=phone, Email=email, Preparer=prep, NAICSCode=naics)

    # ---------------- fraud rings ----------------
    for ring in range(cfg.n_rings):
        size = int(rng.integers(*cfg.ring_size))
        lender = rng.choice(FINTECH) if rng.random() < 0.75 else rng.choice(LENDERS)
        state = rng.choice(STATES, p=STATE_W / STATE_W.sum())
        start = pd.Timestamp("2020-04-03") + pd.Timedelta(days=int(rng.integers(5, 420)))
        # an organiser re-uses a *small pool* of identifiers; each recruit gets a random subset
        pools = {k: [new(k[:-1] if k != "addrs" else "addr") for _ in range(max(1, size // rng.integers(3, 8)))]
                 for k in ("addrs", "phones", "emails", "accts")}
        preparer = new("prep")
        ips = [new("ip") for _ in range(rng.integers(1, 4))]
        sophistication = rng.random()  # sophisticated rings re-use fewer identifiers -> harder
        for _ in range(size):
            r = base_loan(state=state, naics=rng.choice(FRAUD_NAICS_BIAS) if rng.random() < .7 else None,
                          lender=lender, date=start + pd.Timedelta(days=int(rng.integers(0, 21))))
            r.update(is_fraud=1, ring_id=ring, typology="ring", Preparer=preparer if rng.random() < .85 else "SELF")
            if rng.random() > sophistication * 0.6:
                r["BorrowerAddress"] = rng.choice(pools["addrs"])
            if rng.random() > sophistication * 0.5:
                r["Phone"] = rng.choice(pools["phones"])
            if rng.random() > sophistication * 0.5:
                r["Email"] = rng.choice(pools["emails"])
            if rng.random() < .6:
                r["BankAccount"] = rng.choice(pools["accts"])
            if rng.random() < .7:
                r["IPAddress"] = rng.choice(ips)
            _inflate(r, rng, strength=rng.uniform(0.3, 1.0))
            rows.append(r)

    # ---------------- lone-actor fraud (no network signal) ----------------
    for _ in range(cfg.n_lone_fraud):
        r = base_loan(naics=rng.choice(FRAUD_NAICS_BIAS) if rng.random() < .5 else None)
        r.update(is_fraud=1, typology="lone")
        _inflate(r, rng, strength=rng.uniform(0.2, 0.9))
        rows.append(r)

    df = pd.DataFrame(rows).sample(frac=1, random_state=cfg.seed).reset_index(drop=True)
    df["CurrentApprovalAmount"] = df["InitialApprovalAmount"]
    df["BorrowerCity"] = df["BorrowerZip"].str[:3]
    # weak labels: only some fraud is *known* (prosecuted / confirmed); everything else unlabelled
    known = (df["is_fraud"] == 1) & (rng.random(len(df)) < cfg.label_rate)
    df["known_fraud"] = known.astype(int)
    return df


def _inflate(r: dict, rng, strength: float) -> None:
    """Payroll-inflation signals, applied stochastically so no single rule catches all fraud."""
    if rng.random() < 0.55 * strength:
        r["BusinessType"] = rng.choice(["Sole Proprietorship", "Self-Employed Individuals", "Independent Contractors"])
        r["JobsReported"] = 1
        r["InitialApprovalAmount"] = float(SOLE_PROP_MAX if rng.random() < .7 else rng.integers(18000, SOLE_PROP_MAX))
    elif rng.random() < 0.5 * strength:
        r["JobsReported"] = int(rng.choice([10, 15, 20, 25, 50]))
        r["InitialApprovalAmount"] = float(round(r["JobsReported"] * rng.uniform(9000, 20833) / 1000) * 1000)
    if rng.random() < 0.6 * strength:
        r["BusinessAgeDescription"] = rng.choice(["New Business or 2 years or less", "Startup, Loan Funds will Open Business"])
    if rng.random() < 0.3 * strength:
        r["LoanStatus"] = "Charged Off"


def _lender_weights():
    w = np.array([1.0] * 60 + [3.0] * 15 + [12.0] * 3 + [6.0] * 6)
    return w / w.sum()


def _date_days():
    return np.arange(0, 420)


def _date_p():
    d = np.arange(0, 420)
    p = np.exp(-d / 25) + 0.6 * np.exp(-((d - 290) ** 2) / (2 * 30 ** 2))   # 2020 wave + 2021 wave
    return p / p.sum()
