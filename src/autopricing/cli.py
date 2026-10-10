"""Command line: `python -m autopricing <command>`."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from . import data as D
from .config import ROOT, load_config


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="autopricing", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the full pricing + validation pipeline")
    r.add_argument("--config", default=None)
    r.add_argument("--sample", action="store_true", help="use the committed 20k-policy sample (fast, CI)")
    r.add_argument("--skip-sensitivity", action="store_true")
    sub.add_parser("download", help="download freMTPL2freq/sev from OpenML into data/raw")
    s = sub.add_parser("make-sample", help="rebuild data/sample from data/raw")
    s.add_argument("--n", type=int, default=20000)
    sub.add_parser("report", help="re-inject numbers from reports/metrics.json into README and docs/mrm")
    sc = sub.add_parser("score", help="score a CSV of policies with the saved GLM (frequency x severity)")
    sc.add_argument("input")
    sc.add_argument("output")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(getattr(args, "config", None))

    if args.cmd == "run":
        from .pipeline import run
        run(args.config, sample=args.sample, skip_sensitivity=args.skip_sensitivity)
    elif args.cmd == "download":
        print(json.dumps(D.download(cfg, ROOT), indent=2))
    elif args.cmd == "make-sample":
        f, s_ = D.load_raw(ROOT / cfg["data"]["raw_dir"])
        fs, ss = D.make_sample(f, s_, args.n, cfg["seed"])
        out = ROOT / cfg["data"]["sample_dir"]
        out.mkdir(parents=True, exist_ok=True)
        fs.to_parquet(out / "freMTPL2freq.pq", index=False)
        ss.to_parquet(out / "freMTPL2sev.pq", index=False)
        print(f"sample: {len(fs)} policies, {len(ss)} claim records -> {out}")
    elif args.cmd == "report":
        from .report import inject_all
        inject_all(ROOT, json.loads((ROOT / "reports" / "metrics.json").read_text()))
    elif args.cmd == "score":
        from .scoring import score_csv
        score_csv(Path(args.input), Path(args.output))
