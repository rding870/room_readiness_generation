#!/usr/bin/env python3
"""
Base Image Generator

Generates 10 meeting-room and 10 open-space base images using the Gemini API
and saves them into:
  base_images/meeting_room/
  base_images/open_space/

Usage:
  GEMINI_API_KEY=<key> python generate_base_images.py [--dry-run]
"""

import argparse
import base64
import logging
import os
import time
from pathlib import Path

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "gemini-2.5-flash-image"

BASE_DIR   = Path(__file__).parent
OUT_DIRS = {
    "meeting_room": BASE_DIR / "base_images" / "meeting_room",
    "open_space":   BASE_DIR / "base_images" / "open_space",
}

IMAGES_PER_TYPE = 25

# ---------------------------------------------------------------------------
# Prompts — varied so each image looks distinct
# ---------------------------------------------------------------------------

MEETING_ROOM_PROMPTS = [
    "A photorealistic photo of a modern corporate meeting room. Large rectangular table with 8 leather chairs, floor-to-ceiling windows with roller blinds, a whiteboard on the wall, neutral carpet, recessed lighting. Empty room, no people.",
    "A photorealistic photo of a compact meeting room in a tech startup. Round table, 4 modern chairs, a wall-mounted TV screen, whiteboard, polished concrete floor. Empty, daytime.",
    "A photorealistic photo of a formal boardroom. Long dark-wood table, high-back executive chairs, framed artwork on walls, venetian blinds on tall windows, ceiling projector, carpeted floor. Empty, professional setting.",
    "A photorealistic photo of a bright Scandinavian-style meeting room. White walls, light wood table, colourful chairs, large windows with white roller blinds, whiteboard, pendant lights. Empty, clean.",
    "A photorealistic photo of a meeting room inside a busy open office. Rectangular table, 6 fabric chairs, whiteboard, roller blinds on one exterior window, vinyl flooring. Empty, natural light.",
    "A photorealistic photo of a small huddle room. 4-seat table, cushioned chairs, wall-mounted whiteboard, single window with blinds, a monitor on the table, soft lighting. Empty.",
    "A photorealistic photo of a creative agency meeting room. Mismatched chairs, long reclaimed-wood table, exposed-brick wall with a whiteboard, industrial pendant lights, roller blinds, concrete floor. Empty.",
    "A photorealistic photo of a hotel conference room. Oval table, 10 upholstered chairs, heavy curtains on tall windows, projection screen, whiteboard on side wall, patterned carpet. Empty.",
    "A photorealistic photo of a government office meeting room. Plain rectangular table, simple chairs, fluorescent lighting, venetian blinds, whiteboard, linoleum floor, neutral walls. Empty, institutional feel.",
    "A photorealistic photo of a co-working space meeting room. Glass door, compact table for 6, stackable chairs, writable wall surface, exposed ceiling ducts, large window with roller blind. Empty, modern.",
    "A photorealistic photo of a law firm meeting room. Dark mahogany table, 12 leather chairs, built-in bookshelves, venetian blinds on tall windows, whiteboard on one wall, thick carpet. Empty, formal.",
    "A photorealistic photo of a hospital administration meeting room. White laminate table, plastic chairs, fluorescent lighting, venetian blinds, whiteboard, linoleum floor, clean clinical feel. Empty.",
    "A photorealistic photo of a university seminar room repurposed as a meeting space. Rectangular folding tables arranged in a U-shape, chairs, projection screen, whiteboard, roller blinds on large windows, carpeted floor. Empty.",
    "A photorealistic photo of a high-end executive meeting room in a skyscraper. Panoramic city-view windows with motorised roller blinds, long glass table, designer chairs, interactive whiteboard, dark herringbone floor. Empty.",
    "A photorealistic photo of a small team war room. Wall-to-wall whiteboards, standing-height table, bar stools, roller blinds on one window, industrial ceiling, concrete floor, bright LED strips. Empty.",
    "A photorealistic photo of a mid-century modern meeting room. Walnut veneer table, Eames-style chairs, large window with wooden venetian blinds, whiteboard, parquet floor, warm pendant lighting. Empty.",
    "A photorealistic photo of a tech campus all-hands meeting room. Very large rectangular table for 20, ergonomic chairs, video-conferencing screens on two walls, roller blinds, whiteboard, polished concrete floor. Empty.",
    "A photorealistic photo of a heritage building meeting room. Ornate ceiling mouldings, long antique table, mismatched chairs, tall sash windows with roller blinds fitted inside, standalone whiteboard on wheels, wooden floor. Empty.",
    "A photorealistic photo of a retail head-office meeting room. White walls, bright lighting, rectangular table for 8, mesh chairs, roller blinds, wall-mounted whiteboard, laminate floor. Empty, clean.",
    "A photorealistic photo of a media production studio meeting room. Acoustic panels on walls, long table, director-style chairs, roller blinds on blacked-out window, large whiteboard, dark carpet. Empty.",
    "A photorealistic photo of a pharmaceutical company meeting room. Clean-room aesthetic, white laminate table, grey chairs, flush lighting, venetian blinds, glass whiteboard, polished floor. Empty.",
    "A photorealistic photo of a cosy library-style meeting room. Bookshelves on two walls, round table for 6, leather armchairs, sheer roller blind on one window, standalone whiteboard, warm lighting. Empty.",
    "A photorealistic photo of an outdoor-view penthouse meeting room. Wrap-around glass walls with electric roller blinds half-lowered, slim oval table, luxury chairs, interactive whiteboard, light oak floor. Empty.",
    "A photorealistic photo of a banking sector meeting room. Dark navy accent wall, white table, grey upholstered chairs, venetian blinds on full-height windows, wall-mounted whiteboard, carpeted floor. Empty.",
    "A photorealistic photo of a minimalist Japanese-style meeting room. Low rectangular table, floor cushions, shoji-screen roller blinds on wide windows, whiteboard on one wall, bamboo floor, soft diffused lighting. Empty.",
]

OPEN_SPACE_PROMPTS = [
    "A photorealistic photo of a modern open-plan office. Rows of white desks, ergonomic chairs, large windows with roller blinds letting in natural light, acoustic ceiling panels, plants along the window sill. No people.",
    "A photorealistic photo of a tech company open-plan workspace. Sit-stand desks, monitor arms, cable management trays, exposed concrete ceiling, roller blinds on floor-to-ceiling windows, polished concrete floor. No people.",
    "A photorealistic photo of a creative open office. Colourful desks grouped in clusters, pendant lights, plants on shelves, roller blinds on tall windows, whiteboard on one wall, wooden floor. No people.",
    "A photorealistic photo of a financial firm open-plan floor. Include one big table where the meeting takes place. Long rows of trader desks with multiple monitors, venetian blinds on large windows, drop ceiling, carpeted floor, task lighting. No people.",
    "A photorealistic photo of a startup open-plan workspace. Collaborative bench desks, bar-height tables along a window with roller blinds, exposed brick, Edison bulb lighting, concrete floor. No people.",
    "A photorealistic photo of a call-centre style open-plan office. Main meeting table near the camera. Many small desks with partitions, roller blinds on exterior windows, fluorescent lighting, thin carpet tiles, neutral walls. No people.",
    "A photorealistic photo of a media company newsroom-style open-plan office for meetings. One big table for a main meeting near the camera. Curved desk clusters, large screens on walls, floor-to-ceiling windows with roller blinds, vinyl floor, ceiling track lighting. No people.",
    "A photorealistic photo of a government open-plan office for meetings. One big table for a main meeting near the camera. Standard rectangular desks, fabric partition screens, venetian blinds on windows, drop ceiling with fluorescent lights, linoleum floor. No people.",
    "A photorealistic photo of a co-working open space. One big table for a main meeting near the camera. Hot-desks with power outlets, locker banks along one wall, large windows with roller blinds, plants, polished concrete floor, pendant lights. No people.",
    "A photorealistic photo of a design studio open workspace. One big table for a main meeting near the camera. Large drawing tables, drafting chairs, pinboards with sketches, roller blinds on skylights, wooden floor, track lighting. No people.",
    "A photorealistic photo of a legal firm open-plan floor. One large meeting table near the camera. Partner desks with privacy screens, filing cabinets, venetian blinds on tall windows, dark carpet, drop ceiling. No people.",
    "A photorealistic photo of a healthcare administration open-plan office. One big central meeting table near the camera. White desks, ergonomic chairs, roller blinds on large windows, clinical lighting, linoleum floor. No people.",
    "A photorealistic photo of an advertising agency bullpen. One big collaborative table near the camera. Open desks with iMacs, mood boards on walls, roller blinds on skylights, polished concrete, pendant lights. No people.",
    "A photorealistic photo of a university department open office. One large table near the camera. Staff desks with bookshelves, filing cabinets, venetian blinds on sash windows, carpet tiles, strip lighting. No people.",
    "A photorealistic photo of a retail operations open-plan office. One meeting table near the camera. Long rows of desks, monitor stands, roller blinds on strip windows, polished concrete floor, suspended LED lighting. No people.",
    "A photorealistic photo of an insurance company open-plan workspace. Central meeting table near the camera. Rows of identical workstations, fabric partition screens, venetian blinds, drop ceiling, thin carpet tiles. No people.",
    "A photorealistic photo of a biotech startup open-plan lab-office hybrid. Meeting table near the camera. Bench desks, laboratory stools, whiteboard wall, roller blinds on large windows, epoxy floor, industrial lighting. No people.",
    "A photorealistic photo of a publishing house open workspace. Meeting table near the camera. Desks covered with galley proofs, bookshelves, roller blinds on street-facing windows, parquet floor, warm lamp lighting. No people.",
    "A photorealistic photo of a logistics company operations floor. Central briefing table near the camera. Monitoring screens on the wall, standing desks, roller blinds on warehouse-style windows, concrete floor, bright overhead LEDs. No people.",
    "A photorealistic photo of a luxury fashion brand open-plan studio. Long meeting table near the camera. White-topped desks, mood board walls, sheer roller blinds on panoramic windows, pale wood floor, designer pendant lights. No people.",
    "A photorealistic photo of a government ministry open-plan office. Meeting table near the camera. Grey desks with low partitions, venetian blinds on grid windows, drop ceiling with fluorescent tubes, linoleum floor. No people.",
    "A photorealistic photo of a software consultancy open workspace. Collaborative table near the camera. Standing desks, dual monitors, cable trays, roller blinds on full-height windows, poured concrete floor, track lighting. No people.",
    "A photorealistic photo of an architecture firm open studio. Large plan table near the camera. Drafting desks, scale models on shelves, roller blinds on skylights, polished concrete, exposed structure above. No people.",
    "A photorealistic photo of a non-profit organisation open-plan office. Central meeting table near the camera. Modest desks, motivational posters, venetian blinds on modest windows, carpet tiles, budget fluorescent lighting. No people.",
    "A photorealistic photo of a pharmaceutical research open-plan floor. Central meeting table near the camera. Clean white desks, computer workstations, roller blinds on large windows, epoxy floor, bright clinical lighting. No people.",
]

PROMPTS = {
    "meeting_room": MEETING_ROOM_PROMPTS,
    "open_space":   OPEN_SPACE_PROMPTS,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_client: genai.Client | None = None


def init_client(api_key: str) -> None:
    global _client
    _client = genai.Client(api_key=api_key)


def generate_image(prompt: str, output_path: Path, dry_run: bool = False) -> bool:
    """Generate one image and save it. Returns True on success."""
    if dry_run:
        logging.info(f"[DRY RUN] Would generate: {output_path.name}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"placeholder")
        return True

    try:
        assert _client is not None
        response = _client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )

        image_bytes = None
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                image_bytes = part.inline_data.data
                break

        if image_bytes is None:
            logging.error(f"No image in response for: {output_path.name}")
            return False

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(image_bytes)
        logging.info(f"Saved: {output_path.relative_to(Path(__file__).parent)}")
        return True

    except Exception as exc:
        logging.error(f"Failed to generate {output_path.name}: {exc}")
        return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate base room images")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip API calls and write placeholder files",
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

    success = errors = 0

    for room_type, out_dir in OUT_DIRS.items():
        prompts = PROMPTS[room_type]
        logging.info(f"Generating {len(prompts)} {room_type} images → {out_dir}")

        for i, prompt in enumerate(prompts, start=1):
            out_path = out_dir / f"{room_type}_{i:02d}.png"

            if out_path.exists() and not args.dry_run:
                logging.info(f"Already exists, skipping: {out_path.name}")
                success += 1
                continue

            ok = generate_image(prompt, out_path, dry_run=args.dry_run)
            if ok:
                success += 1
            else:
                errors += 1

            if not args.dry_run:
                time.sleep(10)  # stay within per-minute rate limit

    print("\n" + "=" * 50)
    print(f"  Generated : {success}")
    print(f"  Errors    : {errors}")
    for room_type, out_dir in OUT_DIRS.items():
        print(f"  {room_type:15s} → {out_dir.relative_to(BASE_DIR)}/")
    print("=" * 50)


if __name__ == "__main__":
    main()
