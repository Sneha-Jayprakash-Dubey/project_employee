import os

try:
    import shap
except ImportError:
    shap = None

import joblib

# Load the backend model used for SHAP explanations.
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "productivity_model.pkl")

try:
    model = joblib.load(MODEL_PATH)
except Exception:
    model = None

explainer = None
if shap and model is not None:
    try:
        explainer = shap.Explainer(model)
    except Exception:
        explainer = None


def explain_prediction(df):
    """Return per-feature contribution scores using SHAP.

    If SHAP or the explainer is unavailable (missing dependencies or a loading error),
    return an empty dict so the app can continue operating.
    """

    if explainer is None:
        return {}

    shap_values = explainer(df)

    contributions = {}
    for i, feature in enumerate(df.columns):
        contributions[feature] = float(shap_values.values[0][i])

    return contributions