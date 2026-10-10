"""Download, clean, reconcile and split the freMTPL2 frequency and severity tables.

Every cleaning step is counted in a data-quality log so the validation report can cite
exactly how many rows each decision touched.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

USER_AGENT = "AwpDemon portfolio aliaskari.georgia@gmail.com"
RATING_FEATURES = ["Area", "VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Density", "Region"]


# --------------------------------------------------------------------------- download
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(cfg: dict, root: Path, force: bool = False) -> dict:
    """Fetch both OpenML parquet files (CC0 on OpenML) and write a manifest with hashes."""
    raw_dir = root / cfg["data"]["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"retrieved_utc": datetime.now(UTC).isoformat(timespec="seconds"), "files": {}}
    for key, src in cfg["data"]["sources"].items():
        dest = raw_dir / src["file"]
        if force or not dest.exists():
            req = urllib.request.Request(src["url"], headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as out:
                out.write(resp.read())
        digest = sha256_of(dest)
        if src.get("sha256") and src["sha256"] != digest:
            raise ValueError(f"sha256 mismatch for {dest.name}: {digest} != {src['sha256']}")
        manifest["files"][key] = {
            "openml_id": src["openml_id"],
            "url": src["url"],
            "file": src["file"],
            "bytes": dest.stat().st_size,
            "sha256": digest,
            "license": "CC0 (OpenML listing); upstream R package CASdatasets is GPL (>= 2)",
        }
    (raw_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load_raw(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    freq = pd.read_parquet(directory / "freMTPL2freq.pq")
    sev = pd.read_parquet(directory / "freMTPL2sev.pq")
    freq = freq.copy()
    freq["IDpol"] = freq["IDpol"].astype("int64")
    sev = sev.copy()
    sev["IDpol"] = sev["IDpol"].astype("int64")
    for c in ["Area", "VehBrand", "VehGas", "Region"]:
        freq[c] = freq[c].astype(str)
    for c in ["VehPower", "VehAge", "DrivAge", "BonusMalus", "ClaimNb"]:
        freq[c] = freq[c].astype("int64")
    return freq, sev


# --------------------------------------------------------------------------- cleaning
@dataclass
class CleanResult:
    policies: pd.DataFrame          # one row per policy, ClaimNb reconciled and capped
    claims: pd.DataFrame            # one row per claim with an amount (IDpol, ClaimAmount)
    log: dict = field(default_factory=dict)


def claims_only_block_boundary(freq: pd.DataFrame) -> int:
    """Smallest IDpol carrying a zero-claim policy.

    In freMTPL2freq every row below this ID has ClaimNb >= 1 and ~97% of those claims have no
    amount in freMTPL2sev: it is an extract of claim records, not a sample of the portfolio.
    """
    return int(freq.loc[freq["ClaimNb"] == 0, "IDpol"].min())


def clean(freq: pd.DataFrame, sev: pd.DataFrame, cfg: dict) -> CleanResult:
    c = cfg["cleaning"]
    log: dict = {"raw_policies": len(freq), "raw_claim_records": len(sev),
                 "raw_exposure_years": float(freq["Exposure"].sum()), "raw_claims_reported": int(freq["ClaimNb"].sum())}
    df = freq.copy()

    # DQ-1: claims-only block
    boundary = claims_only_block_boundary(df)
    block = df["IDpol"] < boundary
    log["dq1_block_boundary_idpol"] = boundary
    log["dq1_block_policies"] = int(block.sum())
    log["dq1_block_share_with_claims"] = float((df.loc[block, "ClaimNb"] > 0).mean()) if block.any() else 0.0
    log["dq1_block_claims_missing_amount_share"] = (
        float((~df.loc[block & (df["ClaimNb"] > 0), "IDpol"].isin(sev["IDpol"])).mean()) if block.any() else 0.0
    )
    if c["exclude_claims_only_block"]:
        df = df.loc[~block].copy()

    # Claim records must belong to a retained policy
    claims = sev[sev["IDpol"].isin(df["IDpol"])][["IDpol", "ClaimAmount"]].reset_index(drop=True)
    log["sev_records_orphaned_or_excluded"] = int(len(sev) - len(claims))

    # DQ-2: reconcile counts with records that carry an amount
    n_rec = claims.groupby("IDpol").size()
    rec = df["IDpol"].map(n_rec).fillna(0).astype("int64")
    mismatch = rec != df["ClaimNb"]
    log["dq2_policies_count_mismatch"] = int(mismatch.sum())
    log["dq2_claims_without_amount"] = int((df["ClaimNb"] - rec).clip(lower=0).sum())
    if c["reconcile_claim_counts"]:
        df["ClaimNbReported"] = df["ClaimNb"]
        df["ClaimNb"] = rec
    else:
        df["ClaimNbReported"] = df["ClaimNb"]

    # DQ-3: cap claim counts (frequency target only; amounts are kept in full)
    cap_n = c["cap_claim_nb"]
    log["dq3_policies_claimnb_capped"] = int((df["ClaimNb"] > cap_n).sum())
    df["ClaimNb"] = df["ClaimNb"].clip(upper=cap_n)

    # DQ-4: cap exposure
    cap_e = c["cap_exposure"]
    log["dq4_policies_exposure_capped"] = int((df["Exposure"] > cap_e).sum())
    df["Exposure"] = df["Exposure"].clip(upper=cap_e)

    amount = claims.groupby("IDpol")["ClaimAmount"].sum()
    df["ClaimAmount"] = df["IDpol"].map(amount).fillna(0.0)
    df["LogDensity"] = np.log(df["Density"].clip(lower=1.0))

    log["clean_policies"] = len(df)
    log["clean_exposure_years"] = float(df["Exposure"].sum())
    log["clean_claims"] = int(df["ClaimNb"].sum())
    log["clean_claim_records"] = len(claims)
    log["clean_total_claim_amount"] = float(claims["ClaimAmount"].sum())
    return CleanResult(df.reset_index(drop=True), claims, log)


# --------------------------------------------------------------------------- split
def rating_group_key(df: pd.DataFrame) -> pd.Series:
    """Stable hash of the rating-feature combination.

    About 38% of rows share their full rating profile with another row; some are plausibly the
    same risk split across contract periods. Keeping a profile inside one split prevents the
    test set from containing near-copies of training rows.
    """
    key = pd.util.hash_pandas_object(df[RATING_FEATURES].astype(str), index=False)
    return key.astype("uint64")


def _uniform_from_key(key: np.ndarray, seed: int) -> np.ndarray:
    """SplitMix64 finaliser: a key -> a uniform in [0, 1), depending only on (key, seed).

    Assignment of a profile therefore never changes when other rows are added or removed, so
    sensitivity variants (e.g. keeping the claims-only block) are compared on the same split.
    """
    with np.errstate(over="ignore"):
        z = key.astype(np.uint64) + np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def assign_split(df: pd.DataFrame, cfg: dict, seed: int | None = None) -> pd.Series:
    s = cfg["split"]
    seed = cfg["seed"] if seed is None else seed
    if s.get("group_by_rating_features", True):
        draw = _uniform_from_key(rating_group_key(df).to_numpy(), seed)
    else:
        draw = _uniform_from_key(df["IDpol"].to_numpy() if "IDpol" in df else np.arange(len(df)), seed)
    out = np.where(draw < s["train"], "train", np.where(draw < s["train"] + s["validation"], "validation", "test"))
    return pd.Series(out, index=df.index, name="split")


def large_loss_cap(claims: pd.DataFrame, train_ids: pd.Index | np.ndarray, quantile: float) -> float:
    """Individual-claim cap from training claims only (no peeking at test)."""
    amounts = claims.loc[claims["IDpol"].isin(train_ids), "ClaimAmount"]
    return float(amounts.quantile(quantile)) if quantile < 1.0 else float("inf")


def apply_large_loss_cap(df: pd.DataFrame, claims: pd.DataFrame, cap: float, train_mask: pd.Series) -> tuple[pd.DataFrame, float]:
    """Add ClaimAmountCapped and return the large-loss load factor (computed on train).

    load = total train losses / total capped train losses; pure premium = capped model x load.
    """
    capped = claims.assign(ClaimAmountCapped=claims["ClaimAmount"].clip(upper=cap))
    per_pol = capped.groupby("IDpol")["ClaimAmountCapped"].sum()
    out = df.copy()
    out["ClaimAmountCapped"] = out["IDpol"].map(per_pol).fillna(0.0)
    tr = out.loc[train_mask]
    load = float(tr["ClaimAmount"].sum() / tr["ClaimAmountCapped"].sum()) if tr["ClaimAmountCapped"].sum() > 0 else 1.0
    return out, load


def make_sample(policies_raw: pd.DataFrame, sev_raw: pd.DataFrame, n: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Small reproducible sample for CI. Keeps the claims-only block rows proportionally so DQ-1 is exercised."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(policies_raw), size=min(n, len(policies_raw)), replace=False)
    f = policies_raw.iloc[np.sort(idx)].reset_index(drop=True)
    s = sev_raw[sev_raw["IDpol"].isin(f["IDpol"])].reset_index(drop=True)
    return f, s
