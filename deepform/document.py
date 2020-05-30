from dataclasses import dataclass

import numpy as np
import pandas as pd

from deepform.features import fix_dtypes

FEATURE_COLS = [
    "hash",
    "page",
    "x0",
    "y0",
    "length",
    "digitness",
    "is_dollar",
    "log_amount",
]

TOKEN_COLS = ["token", "x0", "y0", "x1", "y1", "page", "match"]


@dataclass(frozen=True)
class Window:
    """A Window just holds views to the arrays held by a Document."""

    tokens: np.ndarray
    features: np.ndarray
    labels: np.ndarray

    def __len__(self):
        return len(self.tokens)


@dataclass(frozen=True)
class Document:
    slug: str
    # tokens, features, and labels are all aligned with the same indices.
    tokens: np.ndarray
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
        """Return the `n`th window in the document."""
        # n is the start of the window.
        m = n + self.window_len // 2  # m is the middle, actual token of interest.
        k = n + self.window_len  # k is the end of the window.
        return Window(self.tokens[n:k], self.features[n:k], self.labels[m])

    def __len__(self):
        """Return the number of windows in the document.

        Note that unless window_len=1, this is less than the number of tokens.
        """
        return len(self.labels) - self.window_len

    def __iter__(self):
        """Iterate over all windows in the document in order."""
        for i in range(len(self)):
            yield self[i]

    @staticmethod
    def from_parquet(slug, pq_path, config):
        """Load precomputed features from a parquet file and apply a config."""
        df = pd.read_parquet(pq_path)

        df["hash"] = (df["hash"] % config.vocab_size) * config.use_string
        df["page"] *= config.use_page
        df["x0"] *= config.use_geom
        df["y0"] *= config.use_geom
        df["log_amount"] *= config.use_amount
        df["match"] = df["gross_amount"]

        if config.pad_windows:
            df = pad_df(df, config.window_len // 2)
        fix_dtypes(df)

        # Pre-compute which windows have the desired token.
        positive_windows = df[df["label"]].index - config.window_len // 2
        assert len(positive_windows) > 0

        # Expand 'label' to gross_amount / void.
        df["label_gross"] = df["label"]
        df["label_void"] = 1 - df["label"]
        labels = df[["label_gross", "label_void"]].to_numpy()

        return Document(
            slug=slug,
            tokens=np.array(df[TOKEN_COLS].to_dict("records")),
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
