# room_readiness_generationRoom Readiness Dataset Generation: Process & Scalability
Current State
50 base images generated (25 meeting room, 25 open space) — sitting in base_images/
0 variant images generated — output/ and new_outputs/ are empty; Stage 2 has not been run yet
Pipeline Overview

Stage 1: generate_base_images.py   →   base_images/{room_type}/*.png
Stage 2: main.py                   →   output/{room_type}/{subtask}/{file}.png
                                       output/labels.json, labels.csv, cost_log.json
Stage 3: change_format.py          →   new_outputs/{category}/  (flat, ML-ready)
Stage 1 — Base Image Generation (generate_base_images.py)
What it does: Calls Gemini (gemini-2.5-flash-image) once per text prompt, saves a PNG.

Dataset size is controlled entirely by the prompt lists — there is no IMAGES_PER_TYPE constant; the docstring says "10" but the actual lists have 25 prompts each (50 total). The count is just len(MEETING_ROOM_PROMPTS) and len(OPEN_SPACE_PROMPTS).

Rate: time.sleep(10) after every call → ~6 calls/min
Time to regenerate 50 images: ~8 min
Cost: 50 images × $0.039/image = ~$1.95 (token costs are negligible for text→image)

Stage 2 — Variant Generation (main.py)
What it does: For every base image, runs 4 subtasks × 2 variants = 8 API calls per base image.

Subtask	Variant A	Variant B
whiteboard	clean	dirty
chairs	neat	messy
blinds	up	down
tables	clean	cluttered
Important: Variant B is generated from Variant A's output (not the original), creating a guaranteed paired difference. If A fails, B is skipped.

output/meeting_room/whiteboard/meeting_room_01__whiteboard_clean.png
output/meeting_room/whiteboard/meeting_room_01__whiteboard_dirty.png
...
output/labels.json   ← full metadata with token counts + cost per image
output/labels.csv
output/cost_log.json
Stage 3 — Format Reorganization (change_format.py)
Flattens the nested structure into category folders and generates per-category annotations.json. Purely local — no API calls, no cost. Run once after Stage 2.

Scalability Analysis
Expansion levers and their cost/time:

Lever	What to change	New images	Added time	Added cost
+25 more base images per type	Add 25 prompts to each list	+50 base, +400 variants	+8 min base, +67 min variants	+$1.95 + $16
+1 new room type (25 images)	Add a third entry to OUT_DIRS/ROOM_TYPES	+25 base, +200 variants	+4 min base, +33 min variants	+$0.98 + $8
+1 new subtask	Add entry to SUBTASKS and VARIANT_PROMPTS	+100 variants (for 50 base)	+17 min	+$4
+1 new variant within a subtask	Extend variant prompt dict	+50 images	+8 min	+$2
The dominant cost driver is $0.039/output image — token costs are small by comparison (~$0.001/call).

Can We Expand with Available Resources?
Yes, with the following constraints:

$25 for 400 images
