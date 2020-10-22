"""Use a model to infer predicted values for a document."""


import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from deepform.common import DATA_DIR, TOKEN_DIR
from deepform.data.add_features import TokenType, add_base_features, as_completed
from deepform.data.tokenize_pdfs import extract_doc
from deepform.document import FEATURE_COLS, Document, pad_df
from deepform.features import fix_dtypes
from deepform.model import load_model


def infer_from_pdf(pdf_path, model=None, window_len=None):
    """Extract features from a PDF and run infrence on it."""
    if not model:
        model, window_len = load_model()
    if not window_len:
        raise Exception("No window_len param provided or inferrable")

    doc = extract_doc(pdf_path, window_len)

    best_score_texts, individual_scores, _ = doc.predict_answer(model)

    # TODO: clean up the column name from the token type enum
    predictions = {
        str(column.name.lower()): {"prediction": text, "score": score}
        for text, score, column in zip(
            best_score_texts, individual_scores, np.array(TokenType)[1:]
        )
    }

    return predictions


def predict(token_file, model, window_len):
    slug = token_file.stem
    doc = tokens_to_doc(token_file, window_len)

    predict_texts, predict_scores, _ = doc.predict_answer(model, 0.5)
    fields = [tt.name.lower() for tt in TokenType if tt.value > 0]
    predictions = []
    for field, text, score in zip(fields, predict_texts, predict_scores):
        predictions.append({"slug": slug, "field": field, "text": text, "score": score})
    return pd.DataFrame(predictions)


def predict_many(token_files, model_file):
    model, window_len = load_model(args.model)
    return pd.concat(predict(t, model, window_len) for t in token_files)


def tokens_to_doc(token_file, window_len=25):
    """Create a Document with features extracted from a pdf."""
    tokens = pd.read_parquet(token_file)
    # Remove tokens shorter than three characters.
    df = tokens[tokens["token"].str.len() >= 3]
    df = add_base_features(df)
    df["tok_id"] = np.minimum(511, df["tok_id"])
    df = pad_df(df, window_len - 1)
    fix_dtypes(df)
    return Document(
        slug=token_file.stem,
        tokens=df,
        features=df[FEATURE_COLS].to_numpy(dtype=float),
        labels=np.zeros(len(df), dtype=bool),  # Dummy.
        positive_windows=np.array(0),  # Dummy.
        window_len=window_len,
        label_values={},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-m", "--model", dest="model", help="model file to use in prediction"
    )
    args = parser.parse_args()

    manifest = pd.read_csv(DATA_DIR / "fcc-data-2020-labeled-manifest.csv")
    slugs = set(manifest.file_id)
    token_files = [t for t in TOKEN_DIR.glob("*.parquet") if t.stem in slugs]
    token_files.sort()

    # Spin up a bunch of jobs to do the conversion
    with ThreadPoolExecutor() as executor:
        doc_jobs = []
        for i in range(0, len(token_files), 100):
            batch = token_files[i : i + 100]
            doc_jobs.append(executor.submit(predict_many, batch, args.model))

        doc_results = []
        for p in as_completed(doc_jobs):
            result = p.result()
            doc_results.append(result)
            print(result.to_string())

    results = pd.concat(doc_results).reset_index(drop=True)
    results.to_csv("predict_on_known.csv", index=False)
