#!/usr/bin/env python3
"""
Remap particle positions into a sample-centric coordinate system
using Sample Top Left + Sample Top Right only.

Usage:
    python compute_global_positions_May13.py \
      <particle_database.csv> <stage_coordinates.csv> <output_folder>
"""
import sys, os, logging
import numpy as np
import pandas as pd
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

FIELD_WIDTH_UM = 50  # µm

def extract_sem_coords(name):
    parts = name.split("_")
    try:
        y = float(parts[parts.index("y")+1])
        x = float(parts[parts.index("x")+1])
    except Exception:
        raise ValueError(f"Cannot parse SEM coords from '{name}'")
    return np.array([x, y])

def load_sample_corners(csv_path):
    df = pd.read_csv(csv_path)
    # We know the first column is the corner name
    name_col = df.columns[0]
    corners = {
        row[name_col].strip(): np.array([row["x"], row["y"]])
        for _, row in df.iterrows()
    }
    for r in ("Sample Top Left","Sample Top Right"):
        if r not in corners:
            raise KeyError(f"Missing corner '{r}' in {csv_path}")
    return corners

def compute_transform(corners):
    tl = corners["Sample Top Left"]
    tr = corners["Sample Top Right"]
    v = tr - tl
    θ = np.arctan2(v[1], v[0])
    cos, sin = np.cos(-θ), np.sin(-θ)
    R = np.array([[ cos, -sin],
                  [ sin,  cos]])
    return R, tl

def main(pdb_csv, corners_csv, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(pdb_csv)
    logging.info(f"Loaded {len(df)} particles from {pdb_csv}")

    # Build global_x_m/global_y_m if missing
    if not {"global_x_m","global_y_m"}.issubset(df.columns):
        logging.info("Computing global_x_m/global_y_m from SEM offsets + local centroids…")
        # parse stage center coords
        offsets = {n: extract_sem_coords(n) for n in df["ImageOrigin"].unique()}

        # grab one image to get pixel dims
        first_img = df["ImageOrigin"].iloc[0]
        # ASSUME your images are in the same folder as your CSV
        # change this path if needed:
        try:
            sample_img_file = os.path.join(os.path.dirname(pdb_csv), first_img + ".tif")
            w_px, h_px = Image.open(sample_img_file).size
        except FileNotFoundError:
            logging.warning("Image not found, falling back to default size (1920,1200).")
            w_px, h_px = 1920, 1200  # Replace with your actual image resolution


        # compute meters-per-pixel
        fx_m = FIELD_WIDTH_UM * 1e-6
        fy_m = fx_m * (h_px / w_px)
        mpx = fx_m / w_px
        mpy = fy_m / h_px
        # yields same as mpx but left for clarity

        # shift SEM center → image top-left
        offset_top_left = np.array([ (w_px/2)*mpx, (h_px/2)*mpy ])
        centers = np.vstack([offsets[n] for n in df["ImageOrigin"]])
        sem_tl = centers - offset_top_left

        local = df[["centroid_x_m","centroid_y_m"]].values
        glob = sem_tl + local
        df["global_x_m"], df["global_y_m"] = glob[:,0], glob[:,1]

    # load corners, compute rotation
    corners = load_sample_corners(corners_csv)
    R, origin = compute_transform(corners)
    logging.info("Computed sample-centric transform")

    # apply
    pts = df[["global_x_m","global_y_m"]].values - origin
    sp = (R @ pts.T).T
    df["sample_x_m"], df["sample_y_m"] = sp[:,0], sp[:,1]

        # after building glob:
    df["global_x_lm_m"] = df["global_x_m"]
    df["global_y_lm_m"] = -df["global_y_m"]   # LM y is up

    # ... later, after computing sample_x_m / sample_y_m:
    df["sample_x_lm_m"] = df["sample_x_m"]
    df["sample_y_lm_m"] = -df["sample_y_m"]   # LM y is up


    # save
    out_csv = os.path.join(out_dir, "particle_sample_positions.csv")
    df.to_csv(out_csv, index=False)
    logging.info(f"✅ Written sample-centric positions to {out_csv}")

if __name__=="__main__":
    if len(sys.argv)!=4:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])