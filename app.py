#!/usr/bin/env python3
"""
Room Readiness Image Studio — unified Flask UI for:
  • Mode A: Generate base images (text → image, saves to base_images/)
  • Mode B: Generate variants   (base image + subtask prompt → edited image, saves to output/)
"""

import base64
import os
from pathlib import Path

import json
import time

from flask import Flask, jsonify, render_template, request, Response, stream_with_context
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Config — mirrors main.py and generate_base_images.py
# ---------------------------------------------------------------------------

MODEL = "gemini-3.1-flash-image-preview"

BASE_DIR    = Path(__file__).parent
BASE_IMAGES = BASE_DIR / "base_images"
OUTPUT_DIR  = BASE_DIR / "output"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# ---- Base-image prompts (from generate_base_images.py) --------------------

BASE_PROMPTS = {
    "meeting_room": [
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
    ],
    "open_space": [
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
    ],
}

# ---- Variant subtask prompts (from main.py) --------------------------------

# Maps dirty variant key → its clean (GT) variant key.
# Dirty variants are generated from the clean GT output, not the base image.
CLEAN_FOR_DIRTY: dict[str, str] = {
    "whiteboard_dirty": "whiteboard_clean",
    "chairs_messy":     "chairs_neat",
    "blinds_down":      "blinds_up",
    "tables_cluttered": "tables_clean",
}

VARIANT_PROMPTS = {
    "whiteboard_clean": (
        "Edit this room image: ensure there is a whiteboard on the wall. "
        "The whiteboard must be completely clean with no marks, drawings, or text. "
        "Keep every other element of the room — furniture, lighting, flooring, "
        "walls, windows, and people (if any) — exactly as they are."
    ),
    "whiteboard_dirty": (
        "Edit this room image: ensure there is a whiteboard on the wall. "
        "Draw diagrams, written text, arrows, flowcharts, and rough sketches "
        "covering most of the whiteboard surface, as if it has been actively "
        "used in a meeting. Do not draw on the walls. Only draw on the whiteboard. "
        "Keep every other element of the room — furniture, lighting, flooring, "
        "walls, windows, and people (if any) — exactly as they are."
    ),
    "chairs_neat": (
        "Edit this room image: arrange every chair neatly tucked in under the "
        "table, evenly spaced and aligned in an orderly configuration. "
        "Keep every other element of the room — table surface, whiteboard, "
        "blinds, lighting, flooring, walls, and windows — exactly as they are."
    ),
    "chairs_messy": (
        "Edit this room image: scatter the chairs in a highly disordered way — "
        "pull them far out from the table at varying distances, rotate them at "
        "different angles, push a few against the walls, "
        "and leave at least one chair notably askew or displaced. Keep all chairs upright. "
        "The result should look like many people just left in a hurry after a long, busy meeting. "
        "Keep every other element of the room — table surface, whiteboard, "
        "blinds, lighting, flooring, walls, and windows — exactly as they are."
    ),
    "blinds_up": (
        "Edit this room image: ensure there are windows with blinds. "
        "Roll all window blinds fully up so the windows are completely uncovered "
        "and maximum natural light enters the room. "
        "Keep every other element of the room — furniture, chairs, whiteboard, "
        "table surface, lighting, flooring, and walls — exactly as they are."
    ),
    "blinds_down": (
        "Edit this room image. Your only task is to close the window blinds. "
        "Find every window in the image. Replace the window area with fully closed, "
        "opaque roller blinds pulled all the way to the bottom sill. "
        "No glass, no outside view, no sky, no daylight visible anywhere. "
        "The room should appear noticeably darker because no natural light enters. "
        "Do not move, add, or remove any furniture, chairs, tables, whiteboards, or walls."
    ),
    "tables_clean": (
        "Edit this room image: make all table surfaces completely clean and clear — "
        "no laptops, papers, cups, cables, or any objects on them. "
        "Keep every other element of the room — chairs, whiteboard, blinds, "
        "lighting, flooring, walls, and windows — exactly as they are."
    ),
    "tables_cluttered": (
        "Edit this room image: scatter a realistic mess of office items across "
        "the table surfaces — open laptops, coffee cups, printed documents, "
        "notebooks, pens, cables, and water bottles in a disorganised arrangement. "
        "Keep every other element of the room — chairs, whiteboard, blinds, "
        "lighting, flooring, walls, and windows — exactly as they are."
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    return genai.Client(api_key=api_key)


def path_to_b64(path: Path) -> dict:
    data = path.read_bytes()
    suffix = path.suffix.lower().lstrip(".")
    mime = f"image/{'jpeg' if suffix in ('jpg', 'jpeg') else suffix}"
    return {"data": base64.b64encode(data).decode(), "mime": mime}


# ---------------------------------------------------------------------------
# Data queries
# ---------------------------------------------------------------------------

def get_base_image_slots():
    """All base image slots with exists flag."""
    slots = []
    for room_type, prompts in BASE_PROMPTS.items():
        for i, prompt in enumerate(prompts, start=1):
            filename = f"{room_type}_{i:02d}.png"
            path = BASE_IMAGES / room_type / filename
            slots.append({
                "id": f"{room_type}/{filename}",
                "room_type": room_type,
                "index": i,
                "filename": filename,
                "prompt": prompt,
                "exists": path.exists(),
                "path": str(path),
            })
    return slots


def get_base_images_on_disk():
    """Base images that exist on disk (for variant mode)."""
    images = []
    for room_dir in sorted(BASE_IMAGES.iterdir()):
        if not room_dir.is_dir():
            continue
        for img_path in sorted(room_dir.glob("*")):
            if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append({
                    "id": f"{room_dir.name}/{img_path.name}",
                    "room_type": room_dir.name,
                    "filename": img_path.name,
                })
    return images


def get_variant_slots():
    """All (base_image, subtask) combinations with exists flag."""
    slots = []
    for img in get_base_images_on_disk():
        img_path = Path(img["path"]) if "path" in img else BASE_IMAGES / img["id"]
        stem = Path(img["filename"]).stem
        for subtask in VARIANT_PROMPTS:
            out_path = OUTPUT_DIR / img["room_type"] / subtask / f"{stem}__{subtask}.png"
            slots.append({
                "base_id": img["id"],
                "room_type": img["room_type"],
                "filename": img["filename"],
                "subtask": subtask,
                "exists": out_path.exists(),
                "out_path": str(out_path),
            })
    return slots


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    base_slots   = get_base_image_slots()
    base_images  = get_base_images_on_disk()
    subtask_keys = list(VARIANT_PROMPTS.keys())
    # Build set of variant keys that already exist on disk: "base_id:subtask"
    variant_done = {
        f"{s['base_id']}:{s['subtask']}"
        for s in get_variant_slots()
        if s["exists"]
    }
    return render_template(
        "index.html",
        base_slots=base_slots,
        base_images=base_images,
        subtasks=subtask_keys,
        variant_done=list(variant_done),
        clean_for_dirty=CLEAN_FOR_DIRTY,
    )


@app.route("/api/base-image/<path:image_id>")
def serve_base_image(image_id):
    path = BASE_IMAGES / image_id
    if not path.exists():
        return jsonify({"error": "Not found"}), 404
    return jsonify(path_to_b64(path))


@app.route("/api/clean-image/<path:image_id>")
def serve_clean_image(image_id):
    """Serve the clean GT output image for a given base image id and subtask.
    image_id format: <room_type>/<filename>   query param: subtask=<clean_subtask>
    """
    clean_subtask = request.args.get("subtask", "")
    if not clean_subtask:
        return jsonify({"error": "subtask query param required"}), 400
    img_path = Path(image_id)
    stem      = img_path.stem
    room_type = img_path.parent.name
    path = OUTPUT_DIR / room_type / clean_subtask / f"{stem}__{clean_subtask}.png"
    if not path.exists():
        return jsonify({"error": "Clean GT image not found"}), 404
    return jsonify(path_to_b64(path))


@app.route("/api/generate-base", methods=["POST"])
def generate_base():
    """Text → image generation (base image mode)."""
    body     = request.get_json()
    prompt   = body.get("prompt", "").strip()
    feedback = body.get("feedback", "").strip()

    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    full_prompt = f"{prompt}\n\nAdditional refinement: {feedback}" if feedback else prompt

    try:
        client   = get_client()
        response = client.models.generate_content(
            model=MODEL,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )

        img_bytes = None
        img_mime  = "image/png"
        text_out  = ""
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                img_bytes = part.inline_data.data
                img_mime  = part.inline_data.mime_type
            elif hasattr(part, "text") and part.text:
                text_out += part.text

        if img_bytes is None:
            return jsonify({"error": "No image returned", "text": text_out}), 500

        usage        = response.usage_metadata
        input_tokens = getattr(usage, "prompt_token_count", 0) or 0
        out_tokens   = getattr(usage, "candidates_token_count", 0) or 0

        return jsonify({
            "image":        base64.b64encode(img_bytes).decode(),
            "mime":         img_mime,
            "input_tokens": input_tokens,
            "out_tokens":   out_tokens,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/generate-variant", methods=["POST"])
def generate_variant():
    """Base image + subtask prompt → edited image (variant mode)."""
    body     = request.get_json()
    image_id = body.get("image_id", "")
    subtask  = body.get("subtask", "")
    feedback = body.get("feedback", "").strip()

    if not (BASE_IMAGES / image_id).exists():
        return jsonify({"error": "Base image not found"}), 404

    base_prompt = VARIANT_PROMPTS.get(subtask)
    if not base_prompt:
        return jsonify({"error": f"Unknown subtask: {subtask}"}), 400

    # For dirty variants, use the clean GT output as the input image
    clean_subtask = CLEAN_FOR_DIRTY.get(subtask)
    if clean_subtask:
        img_path_obj = Path(image_id)
        stem      = img_path_obj.stem
        room_type = img_path_obj.parent.name
        clean_path = OUTPUT_DIR / room_type / clean_subtask / f"{stem}__{clean_subtask}.png"
        if not clean_path.exists():
            return jsonify({"error": f"Clean GT image not found. Generate '{clean_subtask}' first."}), 400
        img_path = clean_path
    else:
        img_path = BASE_IMAGES / image_id

    full_prompt = f"{base_prompt}\n\nAdditional feedback: {feedback}" if feedback else base_prompt

    try:
        client  = get_client()
        suffix  = img_path.suffix.lower().lstrip(".")
        mime    = f"image/{'jpeg' if suffix in ('jpg', 'jpeg') else suffix}"
        img_part = types.Part.from_bytes(data=img_path.read_bytes(), mime_type=mime)

        response = client.models.generate_content(
            model=MODEL,
            contents=[img_part, full_prompt],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )

        img_bytes = None
        img_mime  = "image/png"
        text_out  = ""
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                img_bytes = part.inline_data.data
                img_mime  = part.inline_data.mime_type
            elif hasattr(part, "text") and part.text:
                text_out += part.text

        if img_bytes is None:
            return jsonify({"error": "No image returned", "text": text_out}), 500

        usage        = response.usage_metadata
        input_tokens = getattr(usage, "prompt_token_count", 0) or 0
        out_tokens   = getattr(usage, "candidates_token_count", 0) or 0

        return jsonify({
            "image":        base64.b64encode(img_bytes).decode(),
            "mime":         img_mime,
            "input_tokens": input_tokens,
            "out_tokens":   out_tokens,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/accept-base", methods=["POST"])
def accept_base():
    """Save accepted base image to base_images/<room_type>/<filename>."""
    body     = request.get_json()
    slot_id  = body.get("slot_id")   # e.g. "meeting_room/meeting_room_01.png"
    img_b64  = body.get("image")
    img_mime = body.get("mime", "image/png")

    if not slot_id or not img_b64:
        return jsonify({"error": "slot_id and image are required"}), 400

    out_path = BASE_IMAGES / slot_id
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(base64.b64decode(img_b64))
    return jsonify({"saved": str(out_path.relative_to(BASE_DIR))})


@app.route("/api/accept-variant", methods=["POST"])
def accept_variant():
    """Save accepted variant to output/<room_type>/<subtask>/<stem>__<subtask>.png."""
    body     = request.get_json()
    image_id = body.get("image_id")   # e.g. "meeting_room/meeting_room_01.png"
    subtask  = body.get("subtask")
    img_b64  = body.get("image")

    if not image_id or not subtask or not img_b64:
        return jsonify({"error": "image_id, subtask, and image are required"}), 400

    img_path = Path(image_id)
    stem     = img_path.stem
    room_type = img_path.parent.name if img_path.parent.name else image_id.split("/")[0]

    out_path = OUTPUT_DIR / room_type / subtask / f"{stem}__{subtask}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(base64.b64decode(img_b64))
    return jsonify({"saved": str(out_path.relative_to(BASE_DIR))})


@app.route("/api/batch-generate-subtask")
def batch_generate_subtask():
    """SSE stream: generate + save all variants for a subtask, one by one."""
    subtask = request.args.get("subtask", "")
    if not subtask or subtask not in VARIANT_PROMPTS:
        return jsonify({"error": "Invalid subtask"}), 400

    def generate_stream():
        images       = get_base_images_on_disk()
        total        = len(images)
        clean_subtask = CLEAN_FOR_DIRTY.get(subtask)
        base_prompt  = VARIANT_PROMPTS[subtask]

        yield f"data: {json.dumps({'type': 'start', 'total': total})}\n\n"

        try:
            client = get_client()
        except Exception as e:
            yield f"data: {json.dumps({'type': 'fatal', 'error': str(e)})}\n\n"
            return

        for i, img in enumerate(images):
            image_id  = img["id"]
            stem      = Path(img["filename"]).stem
            room_type = img["room_type"]
            out_path  = OUTPUT_DIR / room_type / subtask / f"{stem}__{subtask}.png"

            if out_path.exists():
                yield f"data: {json.dumps({'type': 'progress', 'index': i, 'total': total, 'image_id': image_id, 'skipped': True})}\n\n"
                continue

            # Resolve input image path (done once, outside retry loop)
            if clean_subtask:
                clean_path = OUTPUT_DIR / room_type / clean_subtask / f"{stem}__{clean_subtask}.png"
                if not clean_path.exists():
                    yield f"data: {json.dumps({'type': 'progress', 'index': i, 'total': total, 'image_id': image_id, 'error': f'Clean image not found for {clean_subtask}'})}\n\n"
                    continue
                img_path = clean_path
            else:
                img_path = BASE_IMAGES / image_id

            suffix   = img_path.suffix.lower().lstrip(".")
            mime     = f"image/{'jpeg' if suffix in ('jpg', 'jpeg') else suffix}"

            # Retry loop: up to 4 attempts with exponential backoff
            last_err = None
            for attempt in range(4):
                if attempt > 0:
                    wait = 15 * (2 ** (attempt - 1))  # 15s, 30s, 60s
                    yield f"data: {json.dumps({'type': 'retrying', 'index': i, 'total': total, 'image_id': image_id, 'attempt': attempt + 1, 'wait': wait})}\n\n"
                    time.sleep(wait)
                try:
                    img_part = types.Part.from_bytes(data=img_path.read_bytes(), mime_type=mime)
                    response = client.models.generate_content(
                        model=MODEL,
                        contents=[img_part, base_prompt],
                        config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"]),
                    )

                    img_bytes    = None
                    img_mime_out = "image/png"
                    for part in response.candidates[0].content.parts:
                        if part.inline_data is not None:
                            img_bytes    = part.inline_data.data
                            img_mime_out = part.inline_data.mime_type
                            break

                    if img_bytes is None:
                        last_err = "No image returned"
                        continue  # retry

                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_bytes(img_bytes)

                    b64 = base64.b64encode(img_bytes).decode()
                    yield f"data: {json.dumps({'type': 'progress', 'index': i, 'total': total, 'image_id': image_id, 'image': b64, 'mime': img_mime_out})}\n\n"
                    last_err = None
                    break  # success

                except Exception as e:
                    last_err = str(e)
                    # Only retry on transient errors (503, 429, timeout)
                    err_lower = last_err.lower()
                    if not any(x in err_lower for x in ("503", "429", "unavailable", "rate", "timeout", "quota")):
                        break  # non-retryable, give up immediately

            if last_err is not None:
                yield f"data: {json.dumps({'type': 'progress', 'index': i, 'total': total, 'image_id': image_id, 'error': last_err})}\n\n"

        yield f"data: {json.dumps({'type': 'complete', 'total': total})}\n\n"

    return Response(
        stream_with_context(generate_stream()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/variant-image/<path:image_id>")
def serve_variant_image(image_id):
    """Serve a saved variant image (output dir) as base64."""
    subtask = request.args.get("subtask", "")
    if not subtask:
        return jsonify({"error": "subtask required"}), 400
    img_path  = Path(image_id)
    stem      = img_path.stem
    room_type = img_path.parent.name
    path      = OUTPUT_DIR / room_type / subtask / f"{stem}__{subtask}.png"
    if not path.exists():
        return jsonify({"error": "Not found"}), 404
    return jsonify(path_to_b64(path))


if __name__ == "__main__":
    app.run(debug=True, port=5000)
