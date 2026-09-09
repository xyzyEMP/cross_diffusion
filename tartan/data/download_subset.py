"""Download a reproducible, storage-bounded TartanGround evaluation subset."""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import shutil
import zipfile

from huggingface_hub import HfApi, snapshot_download


PRIMARY_ENVIRONMENTS = ("ModernCityDowntown", "OldTownFall")
AUXILIARY_ENVIRONMENTS = ("SeasonalForestAutumn", "NordicHarbor", "GreatMarsh")
ROBOTS = ("omni", "diff", "anymal")


def _metadata_files(files: list[str], environments: tuple[str, ...], all_sequences: bool) -> list[str]:
    selected: list[str] = []
    defaults = {"omni": "P0000", "diff": "P1000", "anymal": "P2000"}
    for name in files:
        parts = name.split("/")
        if len(parts) != 4 or parts[0] not in environments or parts[3] != "metadata.zip":
            continue
        robot = parts[1].removeprefix("Data_")
        if robot in ROBOTS and (all_sequences or parts[2] == defaults[robot]):
            selected.append(name)
    return selected


def _extract_and_verify(archive_path: Path) -> None:
    if archive_path.is_symlink():
        materialized = archive_path.with_suffix(archive_path.suffix + ".materialized")
        shutil.copyfile(archive_path.resolve(), materialized)
        materialized.replace(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(archive_path.parent)
    if archive_path.name == "metadata.zip" and not (archive_path.parent / "pose_lcam_front.txt").exists():
        raise RuntimeError(f"Extraction did not produce pose_lcam_front.txt: {archive_path}")
    if archive_path.name.endswith("_sem_pcd.zip"):
        expected = archive_path.with_name(archive_path.name.removesuffix("_pcd.zip") + ".pcd")
        if not expected.exists() or expected.stat().st_size == 0:
            raise RuntimeError(f"Extraction did not produce semantic PCD: {archive_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/media/yzy/1E585F67585F3CA9/data_yzy/TartanGround"),
    )
    parser.add_argument(
        "--profile", choices=("compact", "town_focused"), default="town_focused",
        help="town_focused downloads every sequence in the two primary town environments",
    )
    parser.add_argument(
        "--remove-archives", action=argparse.BooleanOptionalAction, default=True,
        help="remove verified ZIP files after extraction (recoverable from Hugging Face)",
    )
    parser.add_argument("--max-size-gib", type=float, default=30.0)
    args = parser.parse_args()

    repo_files = HfApi().list_repo_files("theairlabcmu/TartanGround", repo_type="dataset")
    patterns = _metadata_files(repo_files, PRIMARY_ENVIRONMENTS, all_sequences=args.profile == "town_focused")
    patterns += _metadata_files(repo_files, AUXILIARY_ENVIRONMENTS, all_sequences=False)
    for environment in PRIMARY_ENVIRONMENTS + AUXILIARY_ENVIRONMENTS:
        patterns.extend((f"{environment}/{environment}_sem_pcd.zip", f"{environment}/seg_labels.zip"))
    patterns = sorted(set(patterns))
    snapshot_download(
        repo_id="theairlabcmu/TartanGround",
        repo_type="dataset",
        local_dir=args.data_root,
        allow_patterns=patterns,
    )

    archives = sorted(args.data_root.rglob("*.zip"))
    for path in archives:
        _extract_and_verify(path)
    if args.remove_archives:
        for path in archives:
            path.unlink()

    size_bytes = sum(path.stat().st_size for path in args.data_root.rglob("*") if path.is_file())
    if size_bytes > args.max_size_gib * 1024**3:
        raise RuntimeError(f"Dataset is {size_bytes / 1024**3:.2f} GiB, above limit {args.max_size_gib:.2f} GiB")
    manifest = {
        "profile": args.profile,
        "primary_environments": list(PRIMARY_ENVIRONMENTS),
        "auxiliary_environments": list(AUXILIARY_ENVIRONMENTS),
        "downloaded_archives": patterns,
        "archives_removed_after_verification": args.remove_archives,
        "extracted_size_bytes": size_bytes,
    }
    (args.data_root / "subset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"TartanGround profile ready: {size_bytes / 1024**3:.2f} GiB, {len(patterns)} archives")


if __name__ == "__main__":
    main()
