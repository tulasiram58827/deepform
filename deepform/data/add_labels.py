"""
Use a manifest to add labels to the tokenized documents.
"""

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from fuzzywuzzy import fuzz
from tqdm import tqdm

from deepform.common import LABELED_DIR, TOKEN_DIR
from deepform.util import is_dollar_amount, normalize_dollars

label_cols = [
    "gross_amount"
]  # ["advertiser", "contract_number", "flight_from", "flight_to"]


def gross_amount_match(lhs, rhs):
    if not (is_dollar_amount(lhs) and is_dollar_amount(rhs)):
        return 0
    return fuzz.ratio(normalize_dollars(lhs), normalize_dollars(rhs)) / 100


def label_doc(token_file, dest_file, gross_amount):
    tokens = pd.read_parquet(token_file)
    tokens["gross_amount"] = tokens.token.apply(
        gross_amount_match, args=(gross_amount,)
    )
    tokens.to_parquet(dest_file, compression="lz4", index=False)


def label_docs(manifest, source_dir, dest_dir):
    token_files = {p.stem: p for p in source_dir.glob("*.parquet")}

    jobqueue = []
    for row in manifest.itertuples():
        slug = row.file_id
        if slug not in token_files:
            logging.error(f"No token file for {slug}")
            continue
        gross_amount = row.gross_amount
        if gross_amount == "":
            logging.warning(f"'gross_amount' for {slug} is empty, skipping")
            continue
        jobqueue.append(
            {
                "token_file": token_files[slug],
                "dest_file": dest_dir / f"{slug}.parquet",
                "gross_amount": gross_amount,
            }
        )

    label_jobs = []
    with ThreadPoolExecutor() as executor:
        for kwargs in jobqueue:
            label_jobs.append(executor.submit(label_doc, **kwargs))

        _ = [j.result() for j in tqdm(as_completed(label_jobs), total=len(label_jobs))]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="CSV with labels for each document")
    parser.add_argument(
        "indir", nargs="?", default=TOKEN_DIR, help="training data to extend",
    )
    parser.add_argument(
        "outdir", nargs="?", default=LABELED_DIR, help="directory of output files",
    )
    parser.add_argument("--log-level", dest="log_level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level.upper())

    logging.info(f"Reading {Path(args.manifest).resolve()}")
    manifest = pd.read_csv(args.manifest)

    indir, outdir = Path(args.indir), Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    label_docs(manifest, indir, outdir)
