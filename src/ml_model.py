import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import LeaveOneOut
import joblib
import sys
sys.path.insert(0, "src")


# ── Feature engineering (scale-agnostic) ─────────────────────────────────────

def engineer_features(sparse_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build scale-agnostic features from sparse measurements.
    All features are normalised to initial conditions so they
    work regardless of bioreactor volume or starting concentration.
    """
    df = sparse_df.copy().sort_values("time_h").reset_index(drop=True)

    # Initial values for normalisation
    x0          = df.DCW_gL.dropna().iloc[0]      if "DCW_gL"      in df else 1.0
    od0         = df.OD660.dropna().iloc[0]        if "OD660"       in df else 1.0
    glycerol0   = df.glycerol_gL.dropna().iloc[0]  if "glycerol_gL" in df else 25.0

    feats = pd.DataFrame()
    feats["time_h"] = df.time_h

    # Normalised biomass (X/X0) — scale agnostic
    if "DCW_gL" in df.columns:
        feats["X_norm"]     = df.DCW_gL / x0
        feats["X_norm"]     = feats.X_norm.interpolate(method="linear")
    if "OD660" in df.columns:
        feats["OD_norm"]    = df.OD660 / od0
        feats["OD_norm"]    = feats.OD_norm.interpolate(method="linear")

    # Specific growth rate mu (h-1) — already scale agnostic
    if "DCW_gL" in df.columns:
        dcw = df.DCW_gL.interpolate(method="linear")
        dt  = df.time_h.diff()
        mu  = np.log(dcw / dcw.shift(1)) / dt
        feats["mu"]         = mu.clip(-0.5, 0.5)
        feats["mu_trend"]   = feats.mu.diff()   # acceleration of growth

    # Normalised glycerol (S/S0) — scale agnostic
    if "glycerol_gL" in df.columns:
        feats["S_norm"]     = df.glycerol_gL / glycerol0
        feats["S_norm"]     = feats.S_norm.interpolate(method="linear")
        feats["S_depletion_rate"] = feats.S_norm.diff() / df.time_h.diff()

    # Methanol presence (binary + amount)
    if "methanol_gL" in df.columns:
        feats["meoh_present"] = (df.methanol_gL > 0.1).astype(float)
        feats["meoh_gL"]      = df.methanol_gL.fillna(0)

    # L1 yield normalised to max observed
    if "L1_yield_mgL" in df.columns:
        l1_max = df.L1_yield_mgL.max()
        if l1_max > 0:
            feats["L1_norm"]        = df.L1_yield_mgL / l1_max
            feats["L1_rate"]        = feats.L1_norm.diff() / df.time_h.diff()

    # Time features
    feats["time_norm"]  = df.time_h / df.time_h.max()  # 0-1 normalised

    # Phase label (for classifier training)
    feats["phase_label"] = "growth"
    if "methanol_gL" in df.columns:
        feats.loc[df.methanol_gL > 0.1, "phase_label"] = "production"
    if "glycerol_gL" in df.columns:
        feats.loc[df.glycerol_gL < 1.0, "phase_label"] = "production"

    return feats


# ── Gaussian Process yield predictor ─────────────────────────────────────────

class YieldPredictor:
    """
    Predicts L1 yield trajectory using Gaussian Process Regression.
    Trained on sparse measurements, gives uncertainty bounds.
    Works with small datasets (2+ batches).
    """

    def __init__(self):
        kernel = Matern(nu=1.5) + WhiteKernel(noise_level=0.1)
        self.gpr    = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=5,
            normalize_y=True, random_state=42
        )
        self.scaler = StandardScaler()
        self.trained = False
        self._l1_max = 1.0

    def fit(self, features_list: list, sparse_list: list):
        """
        Train on multiple batches.
        features_list: list of feature DataFrames
        sparse_list:   list of sparse DataFrames (with L1_yield_mgL)
        """
        X_all, y_all = [], []

        for feats, sparse in zip(features_list, sparse_list):
            if "L1_yield_mgL" not in sparse.columns:
                continue
            # Only use rows where L1 is measured
            labeled = sparse.dropna(subset=["L1_yield_mgL"]).copy()
            if labeled.empty:
                continue

            merged = labeled.merge(feats, on="time_h", how="left")
            feature_cols = ["time_norm", "X_norm", "mu", "OD_norm"]


            merged = merged.reindex(columns=feature_cols + ["L1_yield_mgL"]).fillna(0)
            merged = merged[merged.L1_yield_mgL > 0]

            if merged.empty:
                continue

            X_all.append(merged[feature_cols].values)
            y_all.append(merged.L1_yield_mgL.values)

        if not X_all:
            print("  No training data available for yield predictor")
            return self

        X = np.vstack(X_all)
        y = np.concatenate(y_all)
        self._l1_max = y.max()
        y_norm = y / self._l1_max

        X_scaled = self.scaler.fit_transform(X)
        self.gpr.fit(X_scaled, y_norm)
        self.trained = True
        print(f"  YieldPredictor trained on {len(X)} points")
        return self

    def predict(self, feats: pd.DataFrame):
        """
        Predict L1 yield with uncertainty bounds.
        Returns DataFrame with columns: time_h, L1_pred, L1_lower, L1_upper
        """
        if not self.trained:
            return None

        feature_cols = ["time_norm", "X_norm", "mu", "OD_norm"]
        available = [c for c in feature_cols if c in feats.columns]
        # Align to exact columns seen during training
        df_pred = feats.reindex(columns=feature_cols).fillna(0)
        X = df_pred.values
        X_scaled = self.scaler.transform(X)

        y_mean, y_std = self.gpr.predict(X_scaled, return_std=True)

        return pd.DataFrame({
            "time_h":    feats.time_h.values,
            "L1_pred":   y_mean * self._l1_max,
            "L1_lower":  (y_mean - 1.96 * y_std) * self._l1_max,
            "L1_upper":  (y_mean + 1.96 * y_std) * self._l1_max,
        })


# ── Phase classifier ──────────────────────────────────────────────────────────

class PhaseClassifier:
    """
    Classifies cultivation phase (growth/production) from normalised features.
    Uses Gradient Boosting — robust to small datasets with cross-validation.
    """

    def __init__(self):
        self.clf     = GradientBoostingClassifier(
            n_estimators=50, max_depth=3,
            learning_rate=0.1, random_state=42
        )
        self.scaler  = StandardScaler()
        self.trained = False
        self.classes_ = []

    def fit(self, features_list: list):
        X_all, y_all = [], []
        feature_cols = ["time_norm", "mu", "mu_trend", "X_norm", "OD_norm"]

        for feats in features_list:
            available = [c for c in feature_cols if c in feats.columns]
            sub = feats[available + ["phase_label"]].dropna()
            if sub.empty:
                continue
            # Align to consistent feature set
            X_all.append(sub[available].values)
            if not hasattr(self, "_feature_cols"):
                self._feature_cols = available
            y_all.append(sub.phase_label.values)

        if not X_all:
            return self

        X = np.vstack(X_all)
        y = np.concatenate(y_all)

        # Only train if we have multiple classes
        if len(np.unique(y)) < 2:
            print("  PhaseClassifier: only one phase in training data — skipping")
            return self

        X_scaled = self.scaler.fit_transform(X)
        self.clf.fit(X_scaled, y)
        self.classes_ = list(self.clf.classes_)
        self.trained = True
        print(f"  PhaseClassifier trained on {len(X)} points, "
              f"classes: {self.classes_}")
        return self

    def predict(self, feats: pd.DataFrame):
        """Returns predicted phase and confidence for each timepoint."""
        if not self.trained:
            return None

        feature_cols = ["time_norm", "mu", "mu_trend", "X_norm", "OD_norm"]
        df_pred = feats.reindex(columns=feature_cols).fillna(0)
        X = df_pred.values
        X_scaled = self.scaler.transform(X)

        phases = self.clf.predict(X_scaled)
        proba  = self.clf.predict_proba(X_scaled)
        conf   = proba.max(axis=1)

        return pd.DataFrame({
            "time_h":     feats.time_h.values,
            "phase_pred": phases,
            "confidence": conf,
        })


# ── Adaptive controller wrapper ───────────────────────────────────────────────

class AdaptiveController:
    """
    Wraps RuleBasedController with ML layer.
    - PhaseClassifier overrides rule-based phase if confidence > threshold
    - YieldPredictor estimates harvest timing
    - Self-calibrates thresholds from first 12h of any new run
    """

    def __init__(self, model_dir: str = "outputs/models"):
        self.model_dir       = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.yield_predictor = YieldPredictor()
        self.phase_classifier = PhaseClassifier()
        self._calibrated     = False
        self._mu_max         = 0.16    # default from batch 7
        self._glycerol0      = 25.0    # default initial glycerol
        self._x0             = 0.5     # default initial biomass

    def calibrate(self, early_df: pd.DataFrame, window_h: float = 12.0):
        """
        Self-calibrate from first window_h hours of a new run.
        Updates mu_max, glycerol0, x0 for this specific run.
        """
        early = early_df[early_df.time_h <= window_h].copy()

        if "DCW_gL" in early.columns:
            dcw = early.DCW_gL.dropna()
            if len(dcw) >= 2:
                dt   = early.time_h.diff().dropna()
                mu_vals = np.log(dcw / dcw.shift(1)) / dt
                mu_clean = mu_vals.replace([np.inf, -np.inf], np.nan).dropna()
                self._mu_max = float(mu_clean.max()) if not mu_clean.empty else 0.16
                self._x0     = float(dcw.iloc[0])

        if "glycerol_gL" in early.columns:
            g0 = early.glycerol_gL.dropna()
            if not g0.empty:
                self._glycerol0 = float(g0.iloc[0])

        self._calibrated = True
        print(f"  Calibrated: mu_max={self._mu_max:.3f} h-1, "
              f"glycerol0={self._glycerol0:.1f} g/L, x0={self._x0:.2f} g/L")

        # Update rule thresholds based on calibration
        from control import RuleBasedController
        self.controller = RuleBasedController(
            mu_induction_threshold = 0.15 * self._mu_max,
            glycerol_depletion_gL  = 0.04 * self._glycerol0,
        )

    def train(self, sparse_list: list):
        """Train ML models on all available batches."""
        print("\nTraining ML models...")
        features_list = [engineer_features(s) for s in sparse_list]

        self.phase_classifier.fit(features_list)
        self.yield_predictor.fit(features_list, sparse_list)

        # Save models
        joblib.dump(self.phase_classifier,
                    self.model_dir / "phase_classifier.pkl")
        joblib.dump(self.yield_predictor,
                    self.model_dir / "yield_predictor.pkl")
        print(f"  Models saved to {self.model_dir}")

    def predict_yield(self, sparse_df: pd.DataFrame):
        """Predict L1 yield trajectory with uncertainty."""
        feats = engineer_features(sparse_df)
        return self.yield_predictor.predict(feats)

    def classify_phases(self, sparse_df: pd.DataFrame):
        """Classify phases with confidence scores."""
        feats = engineer_features(sparse_df)
        return self.phase_classifier.predict(feats)


if __name__ == "__main__":
    from loader import load_sparse_samples

    print("Loading data...")
    b5 = load_sparse_samples("data/result-batch_5.xlsx")
    b7 = load_sparse_samples("data/result-batch_7.xlsx")

    ctrl = AdaptiveController()

    # Train on both batches
    ctrl.train([b5, b7])

    # Self-calibrate on batch 7 early data
    print("\nCalibrating on batch 7 first 12h...")
    ctrl.calibrate(b7, window_h=12.0)

    # Predict yield for batch 7
    print("\nYield prediction for batch 7:")
    pred = ctrl.predict_yield(b7)
    if pred is not None:
        print(pred.to_string())

    # Phase classification
    print("\nPhase classification for batch 7:")
    phases = ctrl.classify_phases(b7)
    if phases is not None:
        print(phases.to_string())
