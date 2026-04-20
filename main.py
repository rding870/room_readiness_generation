#!/usr/bin/env python3
"""
Room Readiness Image Generation Script

Generates labeled training/eval images using the Gemini API across 4 subtasks:
  - Whiteboard   (clean / with drawings)
  - Chairs       (messy / neat)
  - Window blinds (rolled up / rolled down)
  - Tables       (clean / cluttered)

Base images:
  base_images/meeting_room/   → 10 images
  base_images/open_space/     → 10 images

Each base image produces 8 output images (4 subtasks × 2 variants).
Total output: 20 base images × 8 = 160 images.

Usage:
  GEMINI_API_KEY=<key> python main.py [--dry-run] [--subtask whiteboard|chairs|blinds|tables]
"""

import argparse
import csv
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "gemini-2.5-flash-image"

# Gemini 2.5 Flash Image pricing (USD) — update if rates change
# https://ai.google.dev/pricing
PRICE_PER_1M_INPUT_TOKENS  = 0.075
PRICE_PER_1M_OUTPUT_TOKENS = 0.30
PRICE_PER_OUTPUT_IMAGE     = 0.039

BASE_DIR    = Path(__file__).parent
BASE_IMAGES = BASE_DIR / "base_images"
OUTPUT_DIR  = BASE_DIR / "output"
LABELS_JSON = OUTPUT_DIR / "labels.json"
LABELS_CSV  = OUTPUT_DIR / "labels.csv"
COST_LOG    = OUTPUT_DIR / "cost_log.json"

SUBTASKS = ["whiteboard", "chairs", "blinds", "tables"]

ROOM_TYPES = {
    "meeting_room": BASE_IMAGES / "meeting_room",
    "open_space":   BASE_IMAGES / "open_space",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# ---------------------------------------------------------------------------
# Prompts
# NOTE on chairs: prompts are intentionally more extreme / dishevelled than
# a previous run — chairs pulled far out, different angles, some askew.
# ---------------------------------------------------------------------------

PROMPTS: dict[str, dict[str, str]] = {
    "whiteboard": {
        "clean": (
            "Edit this room image: ensure there is a whiteboard on the wall. "
            "The whiteboard must be completely clean with no marks, drawings, or text. "
            "Keep every other element of the room — furniture, lighting, flooring, "
            "walls, windows, and people (if any) — exactly as they are."
        ),
        "dirty": (
            "Edit this room image: ensure there is a whiteboard on the wall. Identiy what is the wall and what is the whiteboard"
            "Draw diagrams, written text, arrows, flowcharts, and rough sketches "
            "covering most of the whiteboard surface, as if it has been actively "
            "used in a meeting. Do not draw on the walls. Only draw on the whiteboard"
            "Keep every other element of the room — furniture, lighting, flooring, "
            "walls, windows, and people (if any) — exactly as they are."
        ),
    },
    "chairs": {
        "neat": (
            "Edit this room image: arrange every chair neatly tucked in under the "
            "table, evenly spaced and aligned in an orderly configuration. "
            "Keep every other element of the room — table surface, whiteboard, "
            "blinds, lighting, flooring, walls, and windows — exactly as they are."
        ),
        # Intentionally more chaotic than the previous generation run
        "messy": (
            "Edit this room image: scatter the chairs in a highly disordered way — "
            "pull them far out from the table at varying distances, rotate them at "
            "different angles push a few against the walls, "
            "and leave at least one chair notably askew or displaced. Keep all chairs upright. No chairs should be laying on the ground."
            "The result should look like many people just left in a hurry after a "
            "long, busy meeting. "
            "Keep every other element of the room — table surface, whiteboard, "
            "blinds, lighting, flooring, walls, and windows — exactly as they are."
        ),
    },
    "blinds": {
        "up": (
            "Edit this room image: ensure there are windows with blinds. "
            "Roll all window blinds fully up so the windows are completely uncovered "
            "and maximum natural light enters the room. "
            "Keep every other element of the room — furniture, chairs, whiteboard, "
            "table surface, lighting, flooring, and walls — exactly as they are."
        ),
        "down": (
            "Edit this room image. Your only task is to close the window blinds. "
            "Find every window in the image. Replace the window area with fully closed, "
            "opaque roller blinds pulled all the way to the bottom sill. "
            "The result must look like a solid flat surface of fabric covering each window — "
            "no glass, no outside view, no sky, no daylight visible anywhere. "
            "The room should appear noticeably darker because no natural light enters. "
            "Do not move, add, or remove any furniture, chairs, tables, whiteboards, or walls. "
            "Do not change the camera angle or room layout in any way."
        ),
    },
    "tables": {
        "clean": (
            "Edit this room image: make all table surfaces completely clean and clear — "
            "no laptops, papers, cups, cables, or any objects on them. "
            "Keep every other element of the room — chairs, whiteboard, blinds, "
            "lighting, flooring, walls, and windows — exactly as they are."
        ),
        "cluttered": (
            "Edit this room image: scatter a realistic mess of office items across "
            "the table surfaces — open laptops, coffee cups, printed documents, "
            "notebooks, pens, cables, and water bottles in a disorganised arrangement. "
            "Keep every other element of the room — chairs, whiteboard, blinds, "
            "lighting, flooring, walls, and windows — exactly as they are."
        ),
    },
}

# Maps (subtask, variant_key) → label used in outputs
VARIANT_LABELS: dict[str, dict[str, str]] = {
    "whiteboard": {"clean": "whiteboard_clean",   "dirty": "whiteboard_dirty"},
    "chairs":     {"neat":  "chairs_neat",         "messy": "chairs_messy"},
    "blinds":     {"up":    "blinds_up",           "down":  "blinds_down"},
    "tables":     {"clean": "tables_clean",        "cluttered": "tables_cluttered"},
}

# For each subtask: (clean_variant_key, dirty_variant_key)
# The dirty variant is generated from the clean variant's output, not the base image.
SUBTASK_VARIANT_ORDER: dict[str, tuple[str, str]] = {
    "whiteboard": ("clean", "dirty"),
    "chairs":     ("neat",  "messy"),
    "blinds":     ("up",    "down"),
    "tables":     ("clean", "cluttered"),
}

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GenerationRecord:
    base_image:    str
    room_type:     str
    subtask:       str
    variant:       str
    label:         str
    output_path:   str
    input_tokens:  int = 0
    output_tokens: int = 0
    output_images: int = 1
    cost_usd:      float = 0.0
    timestamp:     str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    error:         Optional[str] = None


@dataclass
class CostSummary:
    total_input_tokens:  int   = 0
    total_output_tokens: int   = 0
    total_output_images: int   = 0
    total_cost_usd:      float = 0.0
    records:             list  = field(default_factory=list)


# ---------------------------------------------------------------------------
# Gemini helpers
# ---------------------------------------------------------------------------

_client: genai.Client | None = None


def init_client(api_key: str) -> None:
    global _client
    _client = genai.Client(api_key=api_key)


def compute_cost(input_tokens: int, output_tokens: int, output_images: int) -> float:
    cost = (
        (input_tokens  / 1_000_000) * PRICE_PER_1M_INPUT_TOKENS
        + (output_tokens / 1_000_000) * PRICE_PER_1M_OUTPUT_TOKENS
        + output_images * PRICE_PER_OUTPUT_IMAGE
    )
    return round(cost, 6)


def generate_variant(
    base_path: Path,
    prompt: str,
    output_path: Path,
    dry_run: bool = False,
) -> tuple[int, int, int, Optional[str]]:
    """
    Call the Gemini image-generation model with the base image + prompt.
    Returns (input_tokens, output_tokens, output_images_count, error_msg).
    Saves the generated image to output_path.
    """
    if dry_run:
        logging.info(f"[DRY RUN] Would generate: {output_path.name}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Copy base image as placeholder
        import shutil
        shutil.copy(base_path, output_path)
        return 100, 10, 1, None

    try:
        assert _client is not None
        suffix = base_path.suffix.lower().lstrip(".")
        mime = f"image/{'jpeg' if suffix in ('jpg', 'jpeg') else suffix}"
        image_part = types.Part.from_bytes(
            data=base_path.read_bytes(),
            mime_type=mime,
        )

        response = _client.models.generate_content(
            model=MODEL,
            contents=[image_part, prompt],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )

        generated_image_bytes = None
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                generated_image_bytes = part.inline_data.data
                break

        if generated_image_bytes is None:
            return 0, 0, 0, "No image returned in API response"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(generated_image_bytes)

        usage = response.usage_metadata
        input_tokens  = getattr(usage, "prompt_token_count",     0) or 0
        output_tokens = getattr(usage, "candidates_token_count", 0) or 0
        return input_tokens, output_tokens, 1, None

    except Exception as exc:
        logging.error(f"Generation failed for {output_path.name}: {exc}")
        return 0, 0, 0, str(exc)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def collect_base_images(base_dir: Path) -> list[Path]:
    images = sorted(
        p for p in base_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        logging.warning(f"No base images found in {base_dir}")
    return images


def run_pipeline(
    subtasks_to_run: list[str],
    dry_run: bool = False,
) -> CostSummary:
    summary = CostSummary()
    all_records: list[GenerationRecord] = []

    for room_type, room_dir in ROOM_TYPES.items():
        base_images = collect_base_images(room_dir)
        logging.info(f"[{room_type}] Found {len(base_images)} base images")

        for base_path in base_images:
            base_stem = base_path.stem

            for subtask in subtasks_to_run:
                clean_key, dirty_key = SUBTASK_VARIANT_ORDER[subtask]

                # Generate clean (GT) variant first from the original base image
                for variant_key in (clean_key, dirty_key):
                    prompt = PROMPTS[subtask][variant_key]
                    label  = VARIANT_LABELS[subtask][variant_key]
                    out_name = f"{base_stem}__{label}{base_path.suffix}"
                    out_path = OUTPUT_DIR / room_type / subtask / out_name

                    # Dirty variant is based on the clean variant's output
                    if variant_key == dirty_key:
                        clean_label    = VARIANT_LABELS[subtask][clean_key]
                        clean_out_name = f"{base_stem}__{clean_label}{base_path.suffix}"
                        input_path     = OUTPUT_DIR / room_type / subtask / clean_out_name
                        if not input_path.exists():
                            err_msg = f"Clean GT image not found, skipping dirty variant: {input_path}"
                            logging.error(err_msg)
                            record = GenerationRecord(
                                base_image    = str(base_path.relative_to(BASE_DIR)),
                                room_type     = room_type,
                                subtask       = subtask,
                                variant       = variant_key,
                                label         = label,
                                output_path   = str(out_path.relative_to(BASE_DIR)),
                                error         = err_msg,
                            )
                            all_records.append(record)
                            continue
                    else:
                        input_path = base_path

                    logging.info(f"Generating: {out_name} (from {input_path.name})")

                    in_tok, out_tok, out_imgs, err = generate_variant(
                        input_path, prompt, out_path, dry_run=dry_run
                    )

                    cost = compute_cost(in_tok, out_tok, out_imgs)

                    record = GenerationRecord(
                        base_image    = str(base_path.relative_to(BASE_DIR)),
                        room_type     = room_type,
                        subtask       = subtask,
                        variant       = variant_key,
                        label         = label,
                        output_path   = str(out_path.relative_to(BASE_DIR)),
                        input_tokens  = in_tok,
                        output_tokens = out_tok,
                        output_images = out_imgs,
                        cost_usd      = cost,
                        error         = err,
                    )

                    all_records.append(record)
                    summary.total_input_tokens  += in_tok
                    summary.total_output_tokens += out_tok
                    summary.total_output_images += out_imgs
                    summary.total_cost_usd      += cost

                    # Stay within per-minute rate limit
                    if not dry_run:
                        time.sleep(10)

    summary.records = [asdict(r) for r in all_records]
    summary.total_cost_usd = round(summary.total_cost_usd, 4)
    return summary


# ---------------------------------------------------------------------------
# Output: labels + cost log
# ---------------------------------------------------------------------------

def write_labels(records: list[dict], json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)
    logging.info(f"Labels written → {json_path}")

    if not records:
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "base_image", "room_type", "subtask", "variant",
        "label", "output_path", "cost_usd", "timestamp", "error",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    logging.info(f"Labels written → {csv_path}")


def write_cost_log(summary: CostSummary, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at":       datetime.now(timezone.utc).isoformat(),
        "model":              MODEL,
        "pricing": {
            "input_per_1m_tokens":  PRICE_PER_1M_INPUT_TOKENS,
            "output_per_1m_tokens": PRICE_PER_1M_OUTPUT_TOKENS,
            "per_output_image":     PRICE_PER_OUTPUT_IMAGE,
        },
        "totals": {
            "input_tokens":  summary.total_input_tokens,
            "output_tokens": summary.total_output_tokens,
            "output_images": summary.total_output_images,
            "cost_usd":      summary.total_cost_usd,
        },
        "per_record": summary.records,
    }
    with open(log_path, "w") as f:
        json.dump(payload, f, indent=2)
    logging.info(f"Cost log written → {log_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Room readiness image generation")
    parser.add_argument(
        "--subtask",
        choices=SUBTASKS,
        help="Run only a specific subtask (default: all 4)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip API calls; copy base images as placeholders to test the pipeline",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key and not args.dry_run:
        raise SystemExit("GEMINI_API_KEY environment variable is not set.")

    if not args.dry_run:
        assert api_key is not None
        init_client(api_key)

    subtasks_to_run = [args.subtask] if args.subtask else SUBTASKS
    logging.info(f"Subtasks: {subtasks_to_run}  |  dry_run={args.dry_run}")

    summary = run_pipeline(subtasks_to_run, dry_run=args.dry_run)

    write_labels(summary.records, LABELS_JSON, LABELS_CSV)
    write_cost_log(summary, COST_LOG)

    # --- Final summary ---
    success = sum(1 for r in summary.records if not r.get("error"))
    errors  = sum(1 for r in summary.records if r.get("error"))
    print("\n" + "=" * 60)
    print(f"  Images generated : {success}")
    print(f"  Errors           : {errors}")
    print(f"  Input tokens     : {summary.total_input_tokens:,}")
    print(f"  Output tokens    : {summary.total_output_tokens:,}")
    print(f"  Output images    : {summary.total_output_images}")
    print(f"  Estimated cost   : ${summary.total_cost_usd:.4f} USD")
    print(f"\n  Labels  → {LABELS_JSON.relative_to(BASE_DIR)}")
    print(f"          → {LABELS_CSV.relative_to(BASE_DIR)}")
    print(f"  Costs   → {COST_LOG.relative_to(BASE_DIR)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
