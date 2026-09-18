"""Create a DVC-versioned bundle from the notebook-selected artifacts."""

from __future__ import annotations

import shutil

from src.config import load_settings


def package_artifacts() -> None:
    settings = load_settings()
    destination = settings.versioned_artifacts_dir
    destination.mkdir(parents=True, exist_ok=True)
    for source in (
        settings.model_path,
        settings.feature_columns_path,
        settings.metadata_path,
    ):
        shutil.copy2(source, destination / source.name)


if __name__ == "__main__":
    package_artifacts()
