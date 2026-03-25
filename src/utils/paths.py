from pathlib import Path


def find_project_root(start_file: str) -> Path:
    """
    Resolve project root by walking up until both `src` and `data` exist.
    """
    current = Path(start_file).resolve().parent
    for candidate in [current] + list(current.parents):
        if (candidate / "src").exists() and (candidate / "data").exists():
            return candidate
    return Path(start_file).resolve().parent


def project_paths(start_file: str):
    root = find_project_root(start_file)
    return {
        "root": root,
        "src": root / "src",
        "data": root / "data",
        "results": root / "results",
        "features_enhanced": root / "data" / "features_enhanced",
        "labels": root / "data" / "labels",
        "raw_videos": root / "data" / "raw_videos",
        "metadata_csv": root / "data" / "video_metadata.csv",
    }
