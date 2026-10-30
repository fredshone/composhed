"""SR-5: model bundles with variant, spec version and configuration."""

import os

import joblib

SPEC_VERSION = "1.0"


class BundleMismatch(ValueError):
    pass


def save_bundle(
    output_dir: str,
    variant: str,
    models: dict,
    feature_names: list[str],
    config: dict,
    **extra,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    bundle = {
        "variant": variant,
        "spec_version": SPEC_VERSION,
        "models": models,
        "feature_names": feature_names,
        "config": config,
        **extra,
    }
    path = os.path.join(output_dir, f"{variant}_models.pkl")
    joblib.dump(bundle, path, compress=3)
    return path


def load_bundle(path: str, variant: str) -> dict:
    bundle = joblib.load(path)
    found = bundle.get("variant") if isinstance(bundle, dict) else None
    if found != variant:
        raise BundleMismatch(f"{path} is a {found!r} bundle, expected {variant!r}")
    return bundle
