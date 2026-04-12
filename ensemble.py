# ensemble.py — shared model classes (must be importable from any entry point)

import numpy as np


class AveragingEnsemble:
    """Averages predictions from multiple models that share the same .predict() interface."""
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        preds = np.stack([m.predict(X) for m in self.models], axis=0)
        return preds.mean(axis=0)
