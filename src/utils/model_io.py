import os
import joblib
from datetime import datetime

# =====================
# CONFIG (FIXED PATH)
# =====================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
MODEL_DIR = os.path.join(BASE_DIR, "models")


# =====================
# INTERNAL HELPER
# =====================
def _get_model_path(name, versioned=False):
    if versioned:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.pkl"
    else:
        filename = f"{name}.pkl"

    return os.path.join(MODEL_DIR, filename)


# =====================
# SAVE MODEL
# =====================
def save_model(model, name, versioned=False, overwrite=True):
    os.makedirs(MODEL_DIR, exist_ok=True)

    path = _get_model_path(name, versioned)

    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"Model already exists: {path}")

    # 🔥 wrap model with metadata
    package = {
        "model": model,
        "saved_at": datetime.now().isoformat(),
        "name": name
    }

    joblib.dump(package, path)

    print(f"✅ Model saved: {path}")

    return path


# =====================
# LOAD MODEL
# =====================
def load_model(name, latest=False):

    if latest:
        if not os.path.exists(MODEL_DIR):
            raise FileNotFoundError("Models directory not found.")

        files = [
            f for f in os.listdir(MODEL_DIR)
            if f.startswith(name) and f.endswith(".pkl")
        ]

        if not files:
            raise FileNotFoundError(f"No versioned models found for: {name}")

        files.sort(reverse=True)
        path = os.path.join(MODEL_DIR, files[0])

    else:
        path = os.path.join(MODEL_DIR, f"{name}.pkl")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")

    package = joblib.load(path)

    print(f"📦 Model loaded: {path}")

    # 🔥 backward compatibility
    if isinstance(package, dict) and "model" in package:
        return package["model"]
    else:
        return package


# =====================
# CHECK MODEL EXISTS
# =====================
def model_exists(name):
    path = os.path.join(MODEL_DIR, f"{name}.pkl")
    return os.path.exists(path)


# =====================
# LIST SAVED MODELS
# =====================
def list_models():
    if not os.path.exists(MODEL_DIR):
        return []

    return [
        f.replace(".pkl", "")
        for f in os.listdir(MODEL_DIR)
        if f.endswith(".pkl")
    ]


# =====================
# DELETE MODEL
# =====================
def delete_model(name):
    path = os.path.join(MODEL_DIR, f"{name}.pkl")

    if os.path.exists(path):
        os.remove(path)
        print(f"🗑 Deleted: {path}")
    else:
        print(f"⚠ Model not found: {path}")