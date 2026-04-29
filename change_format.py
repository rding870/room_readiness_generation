"""
Reorganizes output/ into new_outputs/ with flat category folders.

Current:  output/{room_type}/{category_state}/{room_type}_NN__{category_state}.png
New:      new_outputs/{category}/{category}_{room_type}_NN_{state}.png
          new_outputs/originals/{room_type}_NN.png  (from base_images/)

Categories and state mappings:
  blinds:     blinds_down -> closed,  blinds_up -> open
  chairs:     chairs_messy -> messy,  chairs_neat -> clean
  tables:     tables_clean -> clean,  tables_cluttered -> messy
  whiteboard: whiteboard_clean -> clean, whiteboard_dirty -> messy
"""

import json
import shutil
from pathlib import Path

BASE = Path(__file__).parent
OUTPUT = BASE / "output"
BASE_IMAGES = BASE / "base_images"
NEW_OUTPUT = BASE / "new_outputs"

# Maps output subfolder name -> (category folder, state suffix)
SUBTASK_MAP = {
    "blinds_down":       ("blinds",     "closed"),
    "blinds_up":         ("blinds",     "open"),
    "chairs_messy":      ("chairs",     "messy"),
    "chairs_neat":       ("chairs",     "clean"),
    "tables_clean":      ("tables",     "clean"),
    "tables_cluttered":  ("tables",     "messy"),
    "whiteboard_clean":  ("whiteboard", "clean"),
    "whiteboard_dirty":  ("whiteboard", "messy"),
}

CATEGORIES = {"blinds", "chairs", "tables", "whiteboard", "originals"}


def setup_dirs():
    for cat in CATEGORIES:
        (NEW_OUTPUT / cat).mkdir(parents=True, exist_ok=True)


def copy_variants():
    for room_dir in OUTPUT.iterdir():
        if not room_dir.is_dir():
            continue
        room_type = room_dir.name  # e.g. meeting_room, open_space

        for subtask_dir in room_dir.iterdir():
            if not subtask_dir.is_dir():
                continue
            subtask = subtask_dir.name  # e.g. blinds_down

            if subtask not in SUBTASK_MAP:
                print(f"  Skipping unknown subtask: {subtask}")
                continue

            category, state = SUBTASK_MAP[subtask]

            for src in subtask_dir.glob("*.png"):
                # Extract number from e.g. meeting_room_01__blinds_down.png
                stem = src.stem  # meeting_room_01__blinds_down
                num_part = stem.split("__")[0].split("_")[-1]  # 01

                new_name = f"{category}_{room_type}_{num_part}_{state}.png"
                dst = NEW_OUTPUT / category / new_name

                shutil.copy2(src, dst)
                print(f"  {src.relative_to(BASE)} -> new_outputs/{category}/{new_name}")


def copy_originals():
    for room_dir in BASE_IMAGES.iterdir():
        if not room_dir.is_dir():
            continue
        room_type = room_dir.name

        for src in room_dir.glob("*.png"):
            dst = NEW_OUTPUT / "originals" / src.name
            shutil.copy2(src, dst)
            print(f"  base_images/{room_type}/{src.name} -> new_outputs/originals/{src.name}")


# Label mapping per category state
LABEL_MAP = {
    "blinds":     {"closed": "closed", "open": "open"},
    "chairs":     {"messy": "messy",   "clean": "clean"},
    "tables":     {"clean": "clean",   "messy": "messy"},
    "whiteboard": {"clean": "clean",   "messy": "messy"},
}


def generate_jsons():
    for category in ("blinds", "chairs", "tables", "whiteboard"):
        folder = NEW_OUTPUT / category
        images = sorted(folder.glob("*.png"))

        annotations = []
        for img in images:
            # Extract state from filename e.g. blinds_meeting_room_01_closed.png -> closed
            state = img.stem.rsplit("_", 1)[-1]
            label = LABEL_MAP[category].get(state, state)
            annotations.append({
                "image_filename": img.name,
                "label": label,
                "change_type": category,
            })

        data = {
            "dataset_info": {
                "project": "Logitech ML@B Task 2 - Environment Monitoring Dataset",
                "description": f"Paired images for VLM evaluation - {category}",
                "version": "2.0",
                "change_types": [category],
            },
            "annotations": annotations,
        }

        out_path = folder / "annotations.json"
        out_path.write_text(json.dumps(data, indent=2))
        print(f"  Wrote {out_path.relative_to(BASE)} ({len(annotations)} entries)")


if __name__ == "__main__":
    print("Setting up new_outputs/ folders...")
    setup_dirs()

    print("\nCopying variant images...")
    copy_variants()

    print("\nCopying originals...")
    copy_originals()

    print("\nGenerating annotation JSONs...")
    generate_jsons()

    print("\nDone.")
