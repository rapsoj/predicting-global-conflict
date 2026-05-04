"""
models/two_stage.py
-------------------
Two-stage onset + intensity model for conflict forecasting (V3 improvement #1).

Architecture:
  Stage 1 — LGBMClassifier: P(conflict active this month)
  Stage 2 — LGBMTweedie:    E[count | conflict active]
  Output  — P(active) × E[count | active]

Edge cases:
  always_active  (>95% non-zero months): skip classifier, return regressor directly.
  always_dormant (<5% non-zero months):  return zeros.
"""

import numpy as np
import lightgbm as lgb


class TwoStageModel:
    """
    Binary onset classifier × Tweedie intensity regressor.

    Compatible with sklearn's fit/predict interface.
    Sample weights are accepted by fit() but may be None.
    """

    def fit(self, X, y, w=None):
        y_bin = (y > 0).astype(int)
        active_frac = y_bin.mean()
        self._always_active  = active_frac > 0.95
        self._always_dormant = active_frac < 0.05

        # Stage 2: intensity regressor (active months only)
        mask = y > 0
        self._has_reg = mask.sum() >= 10 and y[mask].sum() > 0
        if self._has_reg:
            self.reg_ = lgb.LGBMRegressor(
                objective='tweedie', tweedie_variance_power=1.5,
                n_estimators=200, verbose=-1)
            try:
                self.reg_.fit(X[mask], y[mask],
                              sample_weight=w[mask] if w is not None else None)
            except Exception:
                self._has_reg = False

        # Stage 1: classifier (mixed regions only)
        self._has_clf = False
        if not self._always_active and not self._always_dormant:
            self.clf_ = lgb.LGBMClassifier(n_estimators=200, verbose=-1)
            try:
                self.clf_.fit(X, y_bin, sample_weight=w)
                self._has_clf = True
            except Exception:
                pass

        return self

    def predict(self, X):
        if self._always_dormant or not self._has_reg:
            return np.zeros(len(X))
        intens = np.maximum(0, self.reg_.predict(X))
        if self._always_active or not self._has_clf:
            return intens
        prob = self.clf_.predict_proba(X)[:, 1]
        return prob * intens
