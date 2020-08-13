"""
Process a parquet of all training data to add labels and computed features.

Final data is stored individually (per-document) to enable random access of
small samples, with an index over all the documents.
"""

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum, auto
from pathlib import Path

import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from tqdm import tqdm

from deepform.common import TOKEN_DIR, TRAINING_DIR, TRAINING_INDEX
from deepform.data.create_vocabulary import get_token_id
from deepform.util import (
    date_similarity,
    default_similarity,
    dollar_similarity,
    is_dollar_amount,
    log_dollar_amount,
)


class TokenType(Enum):
    NONE = 0
    CONTRACT_NUM = auto()
    ADVERTISER = auto()
    FLIGHT_FROM = auto()
    FLIGHT_TO = auto()
    GROSS_AMOUNT = auto()


LABEL_COLS = {
    # Each label column, and the match function that it uses.
    "contract_num": default_similarity,
    "advertiser": default_similarity,
    "flight_from": date_similarity,
    "flight_to": date_similarity,
    "gross_amount": dollar_similarity,
}


def extend_and_write_docs(source_dir, manifest, pq_index, out_path):
    """Split data into individual documents, add features, and write to parquet."""

    token_files = {p.stem: p for p in source_dir.glob("*.parquet")}

    jobqueue = []
    for row in manifest.itertuples():
        slug = row.file_id
        if slug not in token_files:
            logging.error(f"No token file for {slug}")
            continue
        labels = {}
        for label_col in LABEL_COLS:
            labels[label_col] = getattr(row, label_col)
            if not labels[label_col]:
                logging.warning(f"'{label_col}' for {slug} is empty")
        jobqueue.append(
            {
                "token_file": token_files[slug],
                "dest_file": out_path / f"{slug}.parquet",
                "labels": labels,
            }
        )

    # Spin up a bunch of jobs to do the conversion
    with ThreadPoolExecutor() as executor:
        doc_jobs = []
        for kwargs in jobqueue:
            doc_jobs.append(executor.submit(process_document_tokens, **kwargs))

        logging.debug("Waiting for jobs to complete")
        progress = tqdm(as_completed(doc_jobs), total=len(doc_jobs))
        doc_results = [j.result() for j in progress]

    logging.debug(f"Writing document index to {pq_index}...")
    doc_index = pd.DataFrame(doc_results).set_index("slug", drop=True)
    doc_index.to_parquet(pq_index, compression="lz4")


def pq_index_and_dir(pq_index, pq_path=None):
    """Get directory for sharded training data, creating if necessary."""
    pq_index = Path(pq_index).resolve()
    if pq_path is None:
        pq_path = TRAINING_DIR
    else:
        pq_path = Path(pq_path)
    pq_index.parent.mkdir(parents=True, exist_ok=True)
    pq_path.mkdir(parents=True, exist_ok=True)
    return pq_index, pq_path


def process_document_tokens(token_file, dest_file, labels):
    """Filter out short tokens, add computed features, and return index info."""
    slug = token_file.stem
    doc = pd.read_parquet(token_file)

    doc = label_tokens(doc, labels)

    # Remove tokens shorter than three characters.
    doc = doc[doc["token"].str.len() >= 3]

    # Extend with the straightforward features.
    doc = add_base_features(doc)

    # Handle the features that need the whole document.
    doc["label"] = np.zeros(len(doc), dtype="u1")
    # The "label" column stores the TokenType that correctly labels this token.
    # By default this is 0, or "NONE".
    best_matches = {}
    for feature in LABEL_COLS:
        token_value = TokenType[feature.upper()].value
        max_score = doc[feature].max()
        best_matches[f"best_match_{feature}"] = max_score
        matches = token_value * np.isclose(doc[feature], max_score)
        doc["label"] = np.maximum(doc["label"], matches)

    # Write to its final location.
    doc.to_parquet(dest_file, compression="lz4", index=False)

    # Return the summary information about the document.
    return {"slug": slug, "length": len(doc), **labels, **best_matches}


def label_tokens(tokens, labels):
    for col_name, label_value in labels.items():
        match_fn = LABEL_COLS[col_name]
        tokens[col_name] = tokens.token.apply(match_fn, args=(label_value,))
    return tokens


def fraction_digits(s):
    """Return the fraction of a string that is composed of digits."""
    return np.mean([c.isdigit() for c in s]) if isinstance(s, str) else 0.0


def match_string(a, b):
    m = fuzz.ratio(a.lower(), b.lower()) / 100.0
    return m if m >= 0.9 else 0


def add_base_features(token_df):
    """Extend a DataFrame with features that can be pre-computed."""
    df = token_df.copy()
    df["tok_id"] = df["token"].apply(get_token_id).astype("u2")
    df["length"] = df["token"].str.len().astype("i2")
    df["digitness"] = df["token"].apply(fraction_digits).astype("f4")
    df["is_dollar"] = df["token"].apply(is_dollar_amount).astype("f4")
    df["log_amount"] = df["token"].apply(log_dollar_amount).fillna(0).astype("f4")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="CSV with labels for each document")
    parser.add_argument(
        "indir", nargs="?", default=TOKEN_DIR, help="directory of document tokens",
    )
    parser.add_argument(
        "indexfile",
        nargs="?",
        default=TRAINING_INDEX,
        help="path to index of resulting parquet files",
    )
    parser.add_argument(
        "outdir", nargs="?", default=TRAINING_DIR, help="directory of parquet files",
    )
    parser.add_argument("--log-level", dest="log_level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level.upper())

    logging.info(f"Reading {Path(args.manifest).resolve()}")
    manifest = pd.read_csv(args.manifest)

    indir, index, outdir = Path(args.indir), Path(args.indexfile), Path(args.outdir)
    index.parent.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)
    extend_and_write_docs(indir, manifest, index, outdir)
