from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Window:
    """A Window holds a contiguous slice of a Document."""

    tokens: pd.DataFrame
    features: np.ndarray
    labels: np.ndarray

    def __len__(self):
        return len(self.labels)

    def normalize_geometry(self):
        """Alter the geometry fields to make windows always have 0 minimums."""
        self.featuers = self.features.copy()
        self.features[:, 2] -= np.min(self.features[:, 2])  # Normalize x coordinates.
        self.features[:, 3] -= np.min(self.features[:, 3])  # Normalize y coordinates.
        return self
