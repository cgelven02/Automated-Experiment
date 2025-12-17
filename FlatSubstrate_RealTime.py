#!/usr/bin/env python3
import os
import time
import glob
import argparse
import logging
import gc
from pathlib import Path

import torch
import pandas as pd

# python FlatSubstrate_RealTime.py "path to input" "C:\path\to\output" --timeout-sec 60 --poll-sec 5

# Reuse your code & defaults
from SAM2_FlatSubstrate import (
    process_image, make_dirs,
    SAM2AutomaticMaskGenerator,
    SIZE_MIN_DEFAULT, SIZE_MAX_DEFAULT, SKEW_MAX_DEFAULT, FIELD_WIDTH_UM_DEFAULT,
    INVERT_DEFAULT, LOCAL_RATIO_MAX_DEFAULT, DELTA_MIN_DEFAULT, RING_RADIUS_DEFAULT,
    RING_MIN_AREA_DEFAULT, USE_GLOBAL_INT_DEFAULT, GLOBAL_INT_MAX_DEFAULT,
    CLAHE_CLIP_DEFAULT, CLAHE_TILE_DEFAULT, SAM2_REPO_DEFAULT, SAM2_PRED_IOU_DEFAULT,
    SAM2_STAB_SCORE_DEFAULT, SAM2_PPS_DEFAULT
)

IMG_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")


def load_processed_titles(path_txt: Path) -> set[str]:
    if path_txt.exists():
        with path_txt.open("r", encoding="utf-8") as f:
            return set(line.strip() for line in f if line.strip())
    return set()


def append_processed_title(path_txt: Path, title: str) -> None:
    with path_txt.open("a", encoding="utf-8") as f:
        f.write(title + "\n")


def list_new_images(input_dir: Path, processed_by_title: set[str]) -> list[Path]:
    files = []
    for ext in IMG_EXTS:
        files.extend(Path(input_dir).glob(ext))
    # keep only those with unseen *stems* (title = filename without extension)
    new_files = [p for p in files if p.stem not in processed_by_title]
    # process in chronological order (oldest first)
    new_files.sort(key=lambda p: p.stat().st_mtime)
    return new_files


def is_file_ready(path: Path, quiescent_secs: float = 2.0, max_wait_secs: float = 30.0) -> bool:
    """
    Wait until file size is stable for `quiescent_secs` and it is openable for reading.
    Returns False if not ready within `max_wait_secs`.
    """
    start = time.monotonic()
    last_size = -1
    stable_since = time.monotonic()

    while True:
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            size = -1

        if size == last_size and size > 0:
            if (time.monotonic() - stable_since) >= quiescent_secs:
                try:
                    with path.open("rb") as f:
                        _ = f.read(1024)
                    return True
                except Exception:
                    pass
        else:
            stable_since = time.monotonic()
            last_size = size

        if (time.monotonic() - start) > max_wait_secs:
            return False
        time.sleep(0.5)


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    ap = argparse.ArgumentParser(description="Watch a folder and run SAM2 segmentation as images arrive.")
    ap.add_argument("input_dir", help="Folder to watch (e.g., Z:\\Caleb_test_Sep19)")
    ap.add_argument("output_dir", help="Folder to save overlays and CSV")

    # Same knobs as your script (defaults identical)
    ap.add_argument("--size-min", type=int, default=SIZE_MIN_DEFAULT)
    ap.add_argument("--size-max", type=int, default=SIZE_MAX_DEFAULT)
    ap.add_argument("--skew-max", type=float, default=SKEW_MAX_DEFAULT)
    ap.add_argument("--field-width-um", type=float, default=FIELD_WIDTH_UM_DEFAULT)

    ap.add_argument("--invert", action="store_true", default=INVERT_DEFAULT)
    ap.add_argument("--no-invert", dest="invert", action="store_false")

    ap.add_argument("--local-ratio-max", type=float, default=LOCAL_RATIO_MAX_DEFAULT)
    ap.add_argument("--delta-min", type=float, default=DELTA_MIN_DEFAULT)
    ap.add_argument("--ring-radius", type=int, default=RING_RADIUS_DEFAULT)
    ap.add_argument("--ring-min-area", type=int, default=RING_MIN_AREA_DEFAULT)

    ap.add_argument("--use-global-intensity", action="store_true", default=USE_GLOBAL_INT_DEFAULT)
    ap.add_argument("--global-int-max", type=float, default=GLOBAL_INT_MAX_DEFAULT)

    ap.add_argument("--clahe-clip", type=float, default=CLAHE_CLIP_DEFAULT)
    ap.add_argument("--clahe-tile", type=int, default=CLAHE_TILE_DEFAULT)

    ap.add_argument("--sam2-repo", type=str, default=SAM2_REPO_DEFAULT)
    ap.add_argument("--sam2-pred-iou", type=float, default=SAM2_PRED_IOU_DEFAULT)
    ap.add_argument("--sam2-stability", type=float, default=SAM2_STAB_SCORE_DEFAULT)
    ap.add_argument("--sam2-pps", type=int, default=SAM2_PPS_DEFAULT)

    ap.add_argument("--debug", action="store_true", help="Save per-image metrics CSVs in debug_metrics/")
    ap.add_argument("--save-raw-overlays", action="store_true", help="Save overlay of ALL raw SAM2 masks (pre-filter)")

    # Watcher-specific
    ap.add_argument("--poll-sec", type=float, default=5.0, help="Polling interval while waiting for new files")
    ap.add_argument("--timeout-sec", type=float, default=60.0, help="Exit if no new files for this many seconds")
    ap.add_argument("--fresh-csv", action="store_true", help="Delete existing CSV on start (default: append)")
    ap.add_argument("--state-file", type=str, default="processed_titles.txt",
                    help="File storing already-processed titles (persisted across runs)")

    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    overlays_dir, dbg_dir, raw_dir = make_dirs(str(output_dir), args.save_raw_overlays, args.debug)

    csv_path = output_dir / "particle_database.csv"
    state_path = output_dir / args.state_file

    if args.fresh_csv and csv_path.exists():
        csv_path.unlink()
        logging.info("Existing CSV removed (fresh run).")

    # Load persistent state of processed titles
    processed_titles = load_processed_titles(state_path)
    logging.info(f"Loaded {len(processed_titles)} previously processed titles from {state_path.name}")

    # Initialize SAM2 once
    gen = SAM2AutomaticMaskGenerator.from_pretrained(
        args.sam2_repo,
        pred_iou_thresh=args.sam2_pred_iou,
        stability_score_thresh=args.sam2_stability,
        points_per_side=args.sam2_pps
    )

    next_id = 0
    first_csv_write = not csv_path.exists()
    kept_total = 0

    # Inactivity timer
    last_progress_t = time.monotonic()

    try:
        while True:
            # Find new work
            new_files = list_new_images(input_dir, processed_titles)

            if not new_files:
                # timeout check
                if (time.monotonic() - last_progress_t) >= args.timeout_sec:
                    logging.info(f"No new files for {args.timeout_sec:.0f}s. Exiting.")
                    break
                time.sleep(args.poll_sec)
                continue

            for path in new_files:
                title = path.stem
                if title in processed_titles:
                    continue  # double-check

                # Ensure the file is fully written
                if not is_file_ready(path):
                    logging.warning(f"Skipping (not ready): {path.name}")
                    continue

                try:
                    df_image, next_id = process_image(
                        str(path), gen, overlays_dir, dbg_dir, raw_dir, next_id,
                        size_min=args.size_min, size_max=args.size_max, skew_max=args.skew_max,
                        field_width_um=args.field_width_um,
                        invert=args.invert,
                        local_ratio_max=args.local_ratio_max, delta_min=args.delta_min,
                        ring_radius=args.ring_radius, ring_min_area=args.ring_min_area,
                        use_global_intensity=args.use_global_intensity, global_int_max=args.global_int_max,
                        clahe_clip=args.clahe_clip, clahe_tile=args.clahe_tile,
                        debug=args.debug, save_raw_overlays=args.save_raw_overlays
                    )
                    if df_image is not None and not df_image.empty:
                        kept_total += len(df_image)
                        mode = 'a' if csv_path.exists() else 'w'
                        header = not csv_path.exists() or first_csv_write
                        df_image.to_csv(csv_path, mode=mode, header=header, index=False)
                        first_csv_write = False
                        del df_image
                    # mark processed regardless (we don't want to retry the same title)
                    processed_titles.add(title)
                    append_processed_title(state_path, title)
                    last_progress_t = time.monotonic()

                except Exception as e:
                    logging.error(f"Error processing {path.name}: {e}")

                finally:
                    torch.cuda.empty_cache()
                    gc.collect()

    except KeyboardInterrupt:
        logging.info("Interrupted by user. Saving state and exiting.")

    logging.info(f"Done. Particles kept total: {kept_total}. Results: {csv_path}")
    logging.info(f"Processed titles stored in: {state_path}")

if __name__ == "__main__":
    main()
