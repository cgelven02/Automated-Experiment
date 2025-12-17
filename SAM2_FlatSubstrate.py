#!/usr/bin/env python3
"""
SAM2 segmentation (inverted + LOCAL background filtering)

- Uses local ring metrics instead of global-mean to decide what's a particle.
- Keeps segments if:
    local_ratio = mask_mean / ring_mean <= LOCAL_RATIO_MAX
    and local_delta = ring_mean - mask_mean >= DELTA_MIN
- Inverts image (so visually dark particles become numerically darker than bright substrate).
- Robust 16-bit TIFF -> 8-bit conversion without resizing.
- Debug mode: saves per-image metrics for ALL SAM2 masks (before filtering) and optional raw-mask overlays.

Usage:
    python SAM2_FlatSubstrate.py "input_dir" "output_dir"

Dependencies:
    pip install opencv-python pillow numpy pandas scikit-image tqdm torch
    # plus your existing SAM2 package providing SAM2AutomaticMaskGenerator
"""
import argparse
import glob
import os
import random
import logging
import gc

import torch
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from skimage.measure import regionprops_table, label
from skimage.morphology import binary_dilation, disk
from tqdm import tqdm
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# ------------------ Defaults ------------------
SIZE_MIN_DEFAULT         = 200 #250
SIZE_MAX_DEFAULT         = 700000 #700_000
SKEW_MAX_DEFAULT         = 1.35 #1.35
FIELD_WIDTH_UM_DEFAULT   = 50

INVERT_DEFAULT           = False #True
LOCAL_RATIO_MAX_DEFAULT  = 1.1  # keep if mask_mean / ring_mean <= this
DELTA_MIN_DEFAULT        = 0.5  # keep if ring_mean - mask_mean >= this (gray levels 0..255)
RING_RADIUS_DEFAULT      = 15
RING_MIN_AREA_DEFAULT    = 10

# Optional global-intensity gate (usually OFF now)
USE_GLOBAL_INT_DEFAULT   = False
GLOBAL_INT_MAX_DEFAULT   = 9.99   # effectively disabled unless you turn it on

CLAHE_CLIP_DEFAULT       = 1.0
CLAHE_TILE_DEFAULT       = 8

SAM2_REPO_DEFAULT        = "facebook/sam2-hiera-large"
SAM2_PRED_IOU_DEFAULT    = 0.8 #0.75
SAM2_STAB_SCORE_DEFAULT  = 0.87
SAM2_PPS_DEFAULT         = 64
# ------------------------------------------------


def make_dirs(root, save_raw_overlays=False, debug=False):
    os.makedirs(root, exist_ok=True)
    overlays = os.path.join(root, "overlays")
    os.makedirs(overlays, exist_ok=True)
    dbg_dir = os.path.join(root, "debug_metrics")
    raw_dir = os.path.join(root, "raw_overlays")
    if debug:
        os.makedirs(dbg_dir, exist_ok=True)
    if save_raw_overlays:
        os.makedirs(raw_dir, exist_ok=True)
    return overlays, dbg_dir, raw_dir


def load_gray_8bit(path):
    """Load image as grayscale uint8 without resizing. Handles 16-bit gracefully."""
    im = Image.open(path)
    arr = np.array(im)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.shape[2] == 3 else cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    vmin, vmax = float(np.min(arr)), float(np.max(arr))
    if vmax <= vmin + 1e-6:
        return np.zeros_like(arr, dtype=np.uint8)
    return np.clip((arr - vmin) * (255.0 / (vmax - vmin)), 0, 255).astype(np.uint8)


def apply_clahe(gray8, clip=2.0, tile=8):
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(tile), int(tile)))
    return clahe.apply(gray8)


def build_regionprops(mask_bool):
    props = regionprops_table(
        label(mask_bool.astype(int), connectivity=1),
        properties=('centroid', 'axis_major_length', 'axis_minor_length', 'area')
    )
    if props['area'].size == 0:
        return None
    d = {k: v[0] for k, v in props.items()}
    if d['axis_minor_length'] == 0:
        d['axis_minor_length'] = 1e-6
    return d


def local_background_stats(mask_bool, proc_gray, ring_radius=7):
    """Return (mask_mean, ring_mean, ring_area) on processed grayscale (CLAHE +/- inverted)."""
    dil = binary_dilation(mask_bool, disk(int(ring_radius)))
    ring = np.logical_and(dil, ~mask_bool)
    mask_vals = proc_gray[mask_bool]
    ring_vals = proc_gray[ring]
    mask_mean = float(mask_vals.mean()) if mask_vals.size > 0 else np.nan
    ring_mean = float(ring_vals.mean()) if ring_vals.size > 0 else np.nan
    ring_area = int(ring_vals.size)
    return mask_mean, ring_mean, ring_area


def scale_properties(df, field_width_um, image_width_px, image_height_px):
    fx_um = field_width_um
    fy_um = field_width_um * (image_height_px / image_width_px)
    m_per_px_x = (fx_um * 1e-6) / image_width_px
    m_per_px_y = (fy_um * 1e-6) / image_height_px
    m_per_px_geo = np.sqrt(m_per_px_x * m_per_px_y)

    df['centroid_x_m'] = df['centroid-1'] * m_per_px_x
    df['centroid_y_m'] = df['centroid-0'] * m_per_px_y
    df['major_m']      = df['axis_major_length'] * m_per_px_geo
    df['minor_m']      = df['axis_minor_length'] * m_per_px_geo
    df['area_m2']      = df['area'] * (m_per_px_x * m_per_px_y)
    df['diameter_m']   = np.sqrt(df['major_m'] * df['minor_m'])
    return df


# def draw_overlay(base_img_gray8, masks, rows_like_df, indices, out_path, alpha=100):
#     base = Image.fromarray(base_img_gray8).convert("RGB").copy()
#     overlay = base.convert("RGBA")
#     mask_layer = Image.new("RGBA", overlay.size, (0, 0, 0, 0))

#     # Apply colored overlays for each particle
#     for i in indices:
#         ann = masks[int(rows_like_df.loc[i, 'maskNumber'])]
#         seg = ann['segmentation']
#         mask_img = Image.fromarray((seg * 255).astype(np.uint8), mode="L")
#         color = tuple(random.randint(0, 255) for _ in range(3)) + (alpha,)
#         colored = Image.new("RGBA", overlay.size, color)
#         mask_layer = Image.alpha_composite(
#             mask_layer,
#             Image.composite(colored, Image.new("RGBA", overlay.size), mask_img)
#         )

#     # Merge masks with base image
#     composite = Image.alpha_composite(overlay, mask_layer)

#     # Draw ParticleIDs on the final composite
#     draw = ImageDraw.Draw(composite)

#     try:
#     # Windows usually has Arial available
#         font = ImageFont.truetype("arial.ttf", size=16)
    
#     except OSError:
#         # Fallback if font not found
#         font = ImageFont.load_default()
    
#     for i in indices:
#         x_txt = int(rows_like_df.loc[i, 'centroid-1'])
#         y_txt = int(rows_like_df.loc[i, 'centroid-0'])
#         particle_id = int(rows_like_df.loc[i, 'ParticleID'])
#         draw.text((x_txt, y_txt), str(particle_id), fill=(255, 255, 255, 255), font=font)

#     composite.save(out_path)

def should_invert_image(gray8, blur_radius=51, thresh=0.0):
    """
    Decide whether to invert the image so that particles become darker than background.

    - Compares median intensities of the raw image and a blurred (background) version.
    - Returns True if objects appear brighter than background (so we should invert).
    - Returns False if objects already appear darker than background.

    Parameters
    ----------
    gray8 : np.ndarray
        8-bit grayscale image
    blur_radius : int, optional
        Median blur kernel size for background estimation
    thresh : float, optional
        Small tolerance margin. If |difference| <= thresh, we assume no inversion needed.

    Returns
    -------
    bool : True = invert needed, False = keep as-is
    """
    import cv2
    import numpy as np

    bg = cv2.medianBlur(gray8, blur_radius)
    med_img = float(np.median(gray8))
    med_bg  = float(np.median(bg))

    diff = med_img - med_bg
    # If image median is brighter than its local background → likely bright particles on dark substrate → invert
    invert_needed = diff > thresh
    return invert_needed


def draw_overlay(base_img_gray8, masks, rows_like_df, indices, out_path, alpha=100):
    """
    Winner-takes-all rendering: build a label canvas once (no overlaps),
    then render colored regions + text labels.
    Priority: largest area first (so large masks win contested pixels).
    """
    # Shapes
    H, W = base_img_gray8.shape[:2]

    # Priority order (largest area first). Fallback to given order if no 'area'.
    order = list(indices)
    if 'area' in rows_like_df.columns:
        order = sorted(order, key=lambda i: float(rows_like_df.loc[i, 'area']), reverse=True)

    # Label canvas: -1 = unclaimed; otherwise stores DataFrame index i
    label_img = np.full((H, W), -1, dtype=np.int32)

    # Claim pixels without overlap (winner takes all)
    for i in order:
        ann = masks[int(rows_like_df.loc[i, 'maskNumber'])]
        seg = ann['segmentation'].astype(bool)
        write_here = (label_img < 0) & seg
        if np.any(write_here):
            label_img[write_here] = int(i)

    # Render once
    base = Image.fromarray(base_img_gray8).convert("RGB")
    overlay = base.convert("RGBA")
    mask_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))

    for i in order:
        seg = (label_img == int(i))
        if not np.any(seg):
            continue
        mask_img = Image.fromarray(seg.astype(np.uint8) * 255, mode="L")
        color = tuple(random.randint(0, 255) for _ in range(3)) + (int(alpha),)
        colored = Image.new("RGBA", (W, H), color)
        mask_layer = Image.alpha_composite(
            mask_layer,
            Image.composite(colored, Image.new("RGBA", (W, H)), mask_img)
        )

    composite = Image.alpha_composite(overlay, mask_layer)

    # Draw labels
    draw = ImageDraw.Draw(composite)
    try:
        font = ImageFont.truetype("arial.ttf", size=16)
    except OSError:
        font = ImageFont.load_default()

    for i in order:
        # Only label if we actually rendered pixels for this i
        if not np.any(label_img == int(i)):
            continue
        x_txt = int(rows_like_df.loc[i, 'centroid-1'])
        y_txt = int(rows_like_df.loc[i, 'centroid-0'])
        if 'ParticleID' in rows_like_df.columns:
            label_text = str(int(rows_like_df.loc[i, 'ParticleID']))
        elif 'maskNumber' in rows_like_df.columns:
            label_text = str(int(rows_like_df.loc[i, 'maskNumber']))
        else:
            label_text = str(int(i))
        draw.text((x_txt, y_txt), label_text, fill=(255, 255, 255, 255), font=font)

    composite.save(out_path)


def process_image(path, gen, overlays_dir, dbg_dir, raw_dir, start_id,
                  size_min, size_max, skew_max, field_width_um,
                  invert, local_ratio_max, delta_min, ring_radius, ring_min_area,
                  use_global_intensity, global_int_max, clahe_clip, clahe_tile,
                  debug=False, save_raw_overlays=False):
    name = os.path.splitext(os.path.basename(path))[0]
    try:
        gray8 = load_gray_8bit(path)
    except Exception as e:
        logging.error(f"{name}: failed to open image: {e}")
        return None, start_id

    # CLAHE then (optional) invert
    gray = apply_clahe(gray8, clip=clahe_clip, tile=clahe_tile)
    auto_invert = should_invert_image(gray8) #
    proc_gray = 255 - gray if auto_invert else gray #invert
    global_mean = float(proc_gray.mean())
    h, w = proc_gray.shape[:2]

    # SAM2 input
    arr_for_sam2 = cv2.cvtColor(proc_gray, cv2.COLOR_GRAY2RGB)

    # Generate masks
    masks = gen.generate(arr_for_sam2)
    raw_count = len(masks) if masks else 0
    if raw_count == 0:
        logging.info(f"{name}: no masks detected by SAM2.")
        return None, start_id

    rows = []
    for idx, ann in enumerate(masks):
        seg = ann['segmentation'].astype(bool)
        props = build_regionprops(seg)
        if props is None:
            continue

        mask_mean, ring_mean, ring_area = local_background_stats(seg, proc_gray, ring_radius=ring_radius)
        if np.isnan(mask_mean) or np.isnan(ring_mean):
            continue

        local_ratio = mask_mean / max(ring_mean, 1e-6)
        local_delta = ring_mean - mask_mean
        intensity_ratio = mask_mean / max(global_mean, 1e-6)

        props.update({
            'maskNumber': idx,
            'skewness': props['axis_major_length'] / props['axis_minor_length'],
            'mask_mean': mask_mean,
            'ring_mean': ring_mean,
            'ring_area': ring_area,
            'local_ratio': local_ratio,
            'local_delta': local_delta,
            'Intensity_global': intensity_ratio,  # for debugging only
            'ImageOrigin': name
        })
        rows.append(props)

    if not rows:
        logging.info(f"{name}: no valid regions after props.")
        return None, start_id

    df_all = pd.DataFrame(rows)

    # Optional raw-mask overlay (pre-filter)
    if save_raw_overlays:
        out_raw = os.path.join(raw_dir, f"{name}_RAW.png")
        try:
            draw_overlay(gray8, masks, df_all, df_all.index, out_raw, alpha=70)
        except Exception as e:
            logging.warning(f"{name}: raw overlay failed: {e}")

    # Filtering (local criteria first)
    filt = (
        (df_all['area'] > size_min) &
        (df_all['area'] < size_max) &
        (df_all['skewness'] < skew_max) &
        (df_all['ring_area'] >= ring_min_area) &
        (df_all['local_delta'] >= delta_min) &
        (df_all['local_ratio'] <= local_ratio_max)
    )

    # Optional global-intensity gate (usually OFF)
    if use_global_intensity:
        filt &= (df_all['Intensity_global'] <= global_int_max)

    good = df_all[filt].copy()

    # Debug: save per-image metrics
    if debug and dbg_dir:
        df_all.to_csv(os.path.join(dbg_dir, f"{name}_metrics_all.csv"), index=False)

    kept = len(good)
    logging.info(f"{name}: SAM2 masks={raw_count}, kept_after_filters={kept}")

    if good.empty:
        return None, start_id

    # Physical units
    good['centroid_row_px'] = good['centroid-0']
    good['centroid_col_px'] = good['centroid-1']
    good = scale_properties(good, field_width_um, w, h)

    # Assign ParticleID
    ids = np.arange(start_id, start_id + kept)
    good['ParticleID'] = ids
    next_id = start_id + kept

    # Overlay for kept masks
    out_keep = os.path.join(overlays_dir, f"{name}_overlay.png")
    try:
        draw_overlay(gray8, masks, good, good.index, out_keep, alpha=100)
    except Exception as e:
        logging.warning(f"{name}: kept overlay failed: {e}")

    # Cleanup
    torch.cuda.empty_cache()
    del masks, gray8, gray, proc_gray, rows, df_all
    gc.collect()

    return good, next_id




def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    ap = argparse.ArgumentParser(description="SAM2 segmentation with inverted + local background filtering.")
    ap.add_argument("input_dir", help="Folder of input images")
    ap.add_argument("output_dir", help="Folder to save overlays and CSV")

    ap.add_argument("--size-min", type=int, default=SIZE_MIN_DEFAULT)
    ap.add_argument("--size-max", type=int, default=SIZE_MAX_DEFAULT)
    ap.add_argument("--skew-max", type=float, default=SKEW_MAX_DEFAULT)
    ap.add_argument("--field-width-um", type=float, default=FIELD_WIDTH_UM_DEFAULT)

    ap.add_argument("--invert", action="store_true", default=INVERT_DEFAULT)
    ap.add_argument("--no-invert", dest="invert", action="store_false")

    ap.add_argument("--local-ratio-max", type=float, default=LOCAL_RATIO_MAX_DEFAULT,
                    help="Keep if mask_mean / ring_mean <= local_ratio_max")
    ap.add_argument("--delta-min", type=float, default=DELTA_MIN_DEFAULT,
                    help="Keep if ring_mean - mask_mean >= delta_min")
    ap.add_argument("--ring-radius", type=int, default=RING_RADIUS_DEFAULT)
    ap.add_argument("--ring-min-area", type=int, default=RING_MIN_AREA_DEFAULT)

    ap.add_argument("--use-global-intensity", action="store_true", default=USE_GLOBAL_INT_DEFAULT)
    ap.add_argument("--global-int-max", type=float, default=GLOBAL_INT_MAX_DEFAULT,
                    help="Only used if --use-global-intensity is set")

    ap.add_argument("--clahe-clip", type=float, default=CLAHE_CLIP_DEFAULT)
    ap.add_argument("--clahe-tile", type=int, default=CLAHE_TILE_DEFAULT)

    ap.add_argument("--sam2-repo", type=str, default=SAM2_REPO_DEFAULT)
    ap.add_argument("--sam2-pred-iou", type=float, default=SAM2_PRED_IOU_DEFAULT)
    ap.add_argument("--sam2-stability", type=float, default=SAM2_STAB_SCORE_DEFAULT)
    ap.add_argument("--sam2-pps", type=int, default=SAM2_PPS_DEFAULT)

    ap.add_argument("--debug", action="store_true", help="Save per-image metrics CSV in debug_metrics/")
    ap.add_argument("--save-raw-overlays", action="store_true", help="Save overlay of ALL raw SAM2 masks (pre-filter)")

    args = ap.parse_args()

    overlays_dir, dbg_dir, raw_dir = make_dirs(args.output_dir, args.save_raw_overlays, args.debug)
    csv_path = os.path.join(args.output_dir, "particle_database.csv")

    # Start fresh CSV
    if os.path.exists(csv_path):
        os.remove(csv_path)
        logging.info("Existing CSV removed, starting fresh.")
    next_id = 0
    first = True

    # Init SAM2
    gen = SAM2AutomaticMaskGenerator.from_pretrained(
        args.sam2_repo,
        pred_iou_thresh=args.sam2_pred_iou,
        stability_score_thresh=args.sam2_stability,
        points_per_side=args.sam2_pps
    )

    # Gather images
    img_files = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
        img_files.extend(glob.glob(os.path.join(args.input_dir, ext)))
    img_files.sort()
    if not img_files:
        logging.error("No images found in input directory.")
        return

    kept_total = 0
    for path in tqdm(img_files, desc="Processing images"):
        try:
            df_image, next_id = process_image(
                path, gen, overlays_dir, dbg_dir, raw_dir, next_id,
                size_min=args.size_min, size_max=args.size_max, skew_max=args.skew_max,
                field_width_um=args.field_width_um,
                invert=args.invert,
                local_ratio_max=args.local_ratio_max, delta_min=args.delta_min,
                ring_radius=args.ring_radius, ring_min_area=args.ring_min_area,
                use_global_intensity=args.use_global_intensity, global_int_max=args.global_int_max,
                clahe_clip=args.clahe_clip, clahe_tile=args.clahe_tile,
                debug=args.debug, save_raw_overlays=args.save_raw_overlays
            )
            if df_image is not None:
                kept_total += len(df_image)
                df_image.to_csv(csv_path, mode='a', header=first, index=False)
                first = False
                del df_image
        except Exception as e:
            logging.error(f"Error processing {path}: {e}")

    logging.info(f"Done. Particles kept: {kept_total}. Results saved to {csv_path}")


if __name__ == "__main__":
    main()
