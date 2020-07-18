from dataclasses import dataclass

import numpy as np
import pandas as pd

from deepform.features import fix_dtypes

FEATURE_COLS = [
    "tok_id",
    "page",
    "x0",
    "y0",
    "length",
    "digitness",
    "is_dollar",
    "log_amount",
]
NUM_FEATURES = len(FEATURE_COLS)

TOKEN_COLS = ["token", "x0", "y0", "x1", "y1", "page", "match"]


@dataclass
class Window:
    """A Window just holds views to the arrays held by a Document."""

    tokens: pd.DataFrame
    features: np.ndarray
    label: int

    def __len__(self):
        return len(self.features)


@dataclass(frozen=True)
class Document:
    slug: str
    # tokens, features, and labels are all aligned with the same indices.
    tokens: pd.DataFrame
    features: np.ndarray
    labels: np.ndarray
    # positive_windows is a list of which (starting) indices have a match.
    positive_windows: np.ndarray
    window_len: int
    gross_amount: str

    def random_window(self, require_positive=False):
        if require_positive:
            index = np.random.choice(self.positive_windows)
        else:
            index = np.random.randint(len(self))
        return self[index]

    def __getitem__(self, n):
        """Return the window in the document centered around the `n`th token."""
        k = n + 1 + 2 * self.window_len
        return Window(self.tokens.iloc[n:k], self.features[n:k], self.labels[n])

    def __len__(self):
        """Return the number of windows (/tokens) in the document."""
        return len(self.labels)

    def __iter__(self):
        """Iterate over all windows in the document in order."""
        for i in range(len(self)):
            yield self[i]

    @staticmethod
    def from_parquet(slug, pq_path, config):
        """Load precomputed features from a parquet file and apply a config."""
        df = pd.read_parquet(pq_path)

        df["tok_id"] = (
            np.minimum(df["tok_id"], config.vocab_size - 1) * config.use_string
        )
        df["page"] *= config.use_page
        df["x0"] *= config.use_geom
        df["y0"] *= config.use_geom
        df["log_amount"] *= config.use_amount
        df["match"] = df["gross_amount"]

        # Get labels before padding the tokens.
        labels = df["label"].to_numpy(dtype="u1")

        # Pre-compute which windows have the desired token.
        positive_windows = list(df[df["label"] > 0].index)

        if config.pad_windows:
            df = pad_df(df, config.window_len)
        fix_dtypes(df)

        return Document(
            slug=slug,
            tokens=df[TOKEN_COLS],
            features=df[FEATURE_COLS].to_numpy(dtype=float),
            labels=labels,
            positive_windows=np.array(positive_windows),
            window_len=config.window_len,
            gross_amount=actual_value(df, value_col="token", match_col="gross_amount"),
        )


def pad_df(df, num_rows):
    """Add `num_rows` NaNs to the start and end of a DataFrame."""
    zeros = pd.DataFrame(index=pd.RangeIndex(num_rows))
    return pd.concat([zeros, df, zeros]).reset_index(drop=True)


def actual_value(df, value_col, match_col):
    """Return the best value from `value_col`, as evaluated by `match_col`."""
    index = df[match_col].argmax()
    return df.iloc[index][value_col]
