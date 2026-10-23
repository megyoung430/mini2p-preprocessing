"""Fine-tune Cellpose 4 using available GUI segmentations in sessions 216–231.

Run in your Cellpose environment (NumPy >= 1.26 is needed for NumPy 2 pickles):
    python scripts/train_cellpose_manual.py --dry-run
    python scripts/train_cellpose_manual.py

Only reviewed, complete annotations should be saved as *_seg.npy in these
folders: the suffix itself does not prove that the annotations are manual.
"""

import argparse
from pathlib import Path

import numpy as np


# SETTINGS
SUBJECT_PATH = Path(
    "/Volumes/behrens/meg/3x3_field_magnitude_bandit/derivatives/"
    "cohort-01/sub-03_id-MY_71_R"
)
FIRST_SESSION = 216
LAST_SESSION = 231
SAVE_PATH = Path(__file__).resolve().parents[1] / "results" / "cellpose_models"
MODEL_NAME = "mini2p_neurons_v1"
N_EPOCHS = 100
LEARNING_RATE = 1e-5
BATCH_SIZE = 1
MIN_TRAIN_MASKS = 5
GPU = True
IMAGE_EXTENSIONS = (".tif", ".tiff", ".png")


def find_pairs(subject_path):
    """Find exact-stem image pairs, excluding macOS metadata files."""
    if not subject_path.is_dir():
        raise FileNotFoundError(f"Subject folder is unavailable: {subject_path}")
    pairs = []
    for session in range(FIRST_SESSION, LAST_SESSION + 1):
        found = 0
        for plane in sorted(subject_path.glob(
            f"ses-{session}_*/funcimg/run-*/suite2p/plane0"
        )):
            for seg_file in sorted(plane.glob("*_seg.npy")):
                if seg_file.name.startswith("."):
                    continue
                stem = seg_file.name[:-len("_seg.npy")]
                candidates = sorted(
                    f for f in plane.iterdir()
                    if f.is_file() and f.stem == stem
                    and f.suffix.lower() in IMAGE_EXTENSIONS
                )
                if len(candidates) != 1:
                    print(f"SKIP: {seg_file}: expected one image, found {len(candidates)}")
                    continue
                pairs.append((candidates[0], seg_file))
                found += 1
                print(f"Found: {candidates[0].relative_to(subject_path)}")
        print(f"Session {session}: {found} image/segmentation pairs")
    return pairs


def load_training_data(pairs, imread):
    images, labels = [], []
    for image_file, seg_file in pairs:
        # Load only trusted Cellpose GUI files: these contain pickled dictionaries.
        seg_data = np.load(seg_file, allow_pickle=True).item()
        masks = np.asarray(seg_data["masks"])
        if (masks.ndim != 2 or not np.issubdtype(masks.dtype, np.integer)
                or np.any(masks < 0)):
            raise ValueError(f"Expected nonnegative integer 2D masks: {seg_file}")
        ids = np.unique(masks)
        ids = ids[ids > 0]
        if len(ids) < MIN_TRAIN_MASKS:
            print(f"SKIP: {seg_file}: only {len(ids)} ROIs")
            continue
        image = imread(str(image_file))
        if image is None:
            raise ValueError(f"Cannot read image: {image_file}")
        image = np.asarray(image)
        # These training images are grayscale projections, optionally saved as RGB.
        if image.ndim == 3 and image.shape[-1] in (1, 3, 4):
            if image.shape[-1] == 4:
                image = image[..., :3]  # Drop PNG alpha.
        elif image.ndim != 2:
            raise ValueError(f"Expected 2D grayscale or HWC image: {image_file}")
        if image.shape[:2] != masks.shape or not np.isfinite(image).all():
            raise ValueError(f"Image/mask shape mismatch or nonfinite image: {image_file}")
        # Compact ROI IDs; preserve background 0 even when absent in the input.
        compact = np.zeros(masks.shape, dtype=np.int32)
        foreground = masks > 0
        compact[foreground] = np.searchsorted(ids, masks[foreground]) + 1
        # Consistent HWC arrays allow one channel_axis across the training set.
        images.append(image[..., None] if image.ndim == 2 else image)
        labels.append(compact)
        print(f"Loaded: {image_file.name} ({len(ids)} ROIs)")
    if not images:
        raise ValueError("No usable training images remain after mask filtering.")
    return images, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-path", type=Path, default=SUBJECT_PATH)
    parser.add_argument("--save-path", type=Path, default=SAVE_PATH)
    parser.add_argument("--dry-run", action="store_true",
                        help="List image pairs only; do not load masks or train.")
    args = parser.parse_args()
    pairs = find_pairs(args.subject_path)
    print(f"\nTotal image/segmentation pairs: {len(pairs)}")
    if not pairs:
        raise SystemExit("No training pairs found.")
    if args.dry_run:
        return

    from importlib.metadata import version
    from cellpose import io, models, train

    if version("cellpose").split(".")[0] != "4":
        raise RuntimeError("This script requires Cellpose 4.")
    images, labels = load_training_data(pairs, io.imread)
    io.logger_setup()
    model = models.CellposeModel(gpu=GPU)
    print(f"Training on {len(images)} images; device: {model.device}")
    args.save_path.mkdir(parents=True, exist_ok=True)
    model_path, train_losses, test_losses = train.train_seg(
        model.net,
        train_data=images,
        train_labels=labels,
        channel_axis=-1,
        save_path=str(args.save_path),
        model_name=MODEL_NAME,
        n_epochs=N_EPOCHS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        normalize=True,
        min_train_masks=MIN_TRAIN_MASKS,
    )
    np.savez(args.save_path / f"{MODEL_NAME}_losses.npz",
             train_losses=train_losses, test_losses=test_losses)
    print(f"\nTraining complete. Model saved at:\n{model_path}")


if __name__ == "__main__":
    main()
