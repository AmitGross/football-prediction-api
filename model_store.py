# model_store.py — upload / download model.pkl to / from Supabase Storage

import io
import os
import pickle

MODEL_BUCKET          = "models"
MODEL_OBJECT          = "model.pkl"
LOCAL_MODEL_PATH      = "/tmp/model.pkl"


def upload_model(model, feature_cols: list, model_version: str) -> None:
    """Serialise model → upload to Supabase Storage bucket 'models'."""
    from supabase_client import get_client

    payload = {"model": model, "features": feature_cols, "version": model_version}
    buf = io.BytesIO()
    pickle.dump(payload, buf)
    buf.seek(0)
    data = buf.read()

    client = get_client()
    client.storage.from_(MODEL_BUCKET).upload(
        path=MODEL_OBJECT,
        file=data,
        file_options={"upsert": "true", "content-type": "application/octet-stream"},
    )
    print(f"[model_store] model.pkl uploaded ({len(feature_cols)} features, {len(data)//1024} KB)")


def download_model() -> str:
    """Download model.pkl from Supabase Storage → /tmp/model.pkl. Returns local path."""
    from supabase_client import get_client

    client = get_client()
    data = client.storage.from_(MODEL_BUCKET).download(MODEL_OBJECT)

    os.makedirs("/tmp", exist_ok=True)
    with open(LOCAL_MODEL_PATH, "wb") as f:
        f.write(data)
    print(f"[model_store] model.pkl downloaded ({len(data)//1024} KB)")
    return LOCAL_MODEL_PATH


def load_model_from_storage():
    """Download from Supabase Storage and deserialise. Returns (model, feature_cols)."""
    path = download_model()
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload["model"], payload["features"]
