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

    def center_geometry(self):
        """Alter the geometry fields to center windows around 0."""
        self.featuers = self.features.copy()
        self.features[:, 2] -= np.mean(self.features[:, 2])  # Center x coordinates.
        self.features[:, 3] -= np.mean(self.features[:, 3])  # Center y coordinates.
        return self
