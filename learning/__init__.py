"""
learning - Block 3: Machine Learning and Deep Learning for risk prediction.

Run everything from the repository root:

    python -m learning.dataset            # load PhysioNet (2 hospitals), label, split, build samples
    python -m learning.train_ml           # train/compare ML models, save the best
    python -m learning.train_dl           # train the GRU
    python -m learning.demo_block3        # end-to-end demonstration
    python -m unittest learning.test_block3

Modules
    data_loader.py   read PhysioNet 2019 .psv files (vitals + age only)
    labeling.py      NEWS2 score and the 4 risk classes
    dataset.py       6-hour windows, imputation, split by patient
    evaluate.py      metrics, persistence baseline, confusion matrices
    train_ml.py      Logistic Regression / Decision Tree / Random Forest / k-NN
    train_dl.py      GRU on the vital-sign time series
    predictor.py     RiskPredictor: models -> ML_Risk / DL_Risk facts in the KB
"""
LEVELS = ["Low", "Medium", "High", "Critical"]


def __getattr__(name):
    # Lazy import: `python -m learning.dataset` must not import dataset twice.
    if name in ("RiskPredictor", "Prediction"):
        from . import predictor
        return getattr(predictor, name)
    raise AttributeError(name)
