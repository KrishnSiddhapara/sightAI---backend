"""
debug_boxes.py - Diagnostic tool for SightAI bounding box pipeline.
Runs each pipeline stage and saves visualization PNGs on the EXACT image bytes sent to Gemini.
Prints raw model JSON and a detailed audit of which instances lose or change boxes.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any, Optional
from PIL import Image, ImageDraw, ImageFont
import io

from dotenv import load_dotenv
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from google import genai
from google.genai import types
from utils.image_validation import encode_vlm_image_part
from utils.config import MAX_ANALYSIS_DIMENSION, PRIMARY_VLM_MODEL
from utils.coordinate_utils import (
    parse_gemini_box_2d,
    parse_standard_bbox,
    calculate_iou,
    calculate_containment,
    calculate_center_distance
)
from utils.json_utils import clean_json_text, normalize_grounded_analysis
from services.schemas import GroundedAnalysisResult, BoundingBox
from services.vision import (
    sanitize_bounding_boxes,
    refine_bounding_boxes,
    reconcile_unboxed_instances,
    localize_objects,
    GROUNDED_VISION_SYSTEM_PROMPT,
    GENERIC_OBJECT_NAMES,
    SURFACE_OBJECT_NAMES,
    SUBPART_OBJECT_NAMES
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("debug_boxes")


COLOR_PALETTE = [
    (99, 102, 241),   # indigo
    (16, 185, 129),   # green
    (245, 158, 11),   # amber
    (236, 72, 153),   # pink
    (6, 182, 212),    # cyan
    (139, 92, 246),   # purple
    (239, 68, 68),    # red
    (34, 197, 94),    # emerald
    (59, 130, 246),   # blue
]


def draw_boxes_on_image(
    base_img: Image.Image,
    boxes_with_labels: List[Dict[str, Any]],
    output_path: str,
    title: str = ""
):
    """
    Draws bounding boxes on base_img (0-1000 scale) and saves to output_path.
    boxes_with_labels: list of {'label': str, 'box': {'x_min','y_min','x_max','y_max'}, 'color_idx': int}
    """
    img_copy = base_img.copy().convert("RGB")
    draw = ImageDraw.Draw(img_copy)
    w, h = img_copy.size

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for idx, item in enumerate(boxes_with_labels):
        b = item["box"]
        label = item.get("label", "object")
        color = COLOR_PALETTE[item.get("color_idx", idx) % len(COLOR_PALETTE)]

        # Map 0-1000 to pixel coordinates
        x1 = int(round(b["x_min"] * w / 1000.0))
        y1 = int(round(b["y_min"] * h / 1000.0))
        x2 = int(round(b["x_max"] * w / 1000.0))
        y2 = int(round(b["y_max"] * h / 1000.0))

        # Clamp
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        # Draw rectangle (3px width)
        for offset in range(3):
            draw.rectangle(
                [x1 + offset, y1 + offset, x2 - offset, y2 - offset],
                outline=color
            )

        # Draw label tag
        text = f"{label} [{int(b['x_min'])},{int(b['y_min'])}-{int(b['x_max'])},{int(b['y_max'])}]"
        bbox = draw.textbbox((x1, max(0, y1 - 15)), text, font=font) if font else (x1, y1, x1 + 80, y1 + 12)
        draw.rectangle([bbox[0] - 2, bbox[1] - 1, bbox[2] + 2, bbox[3] + 1], fill=color)
        draw.text((x1, max(0, y1 - 15)), text, fill=(255, 255, 255), font=font)

    # Title watermark at top
    if title:
        draw.rectangle([0, 0, w, 22], fill=(15, 23, 42))
        draw.text((10, 4), f"{title} (Boxes: {len(boxes_with_labels)})", fill=(248, 250, 252), font=font)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img_copy.save(output_path)
    logger.info(f"Saved stage visualization: {output_path} ({len(boxes_with_labels)} boxes)")


def run_debug_pipeline(image_path: str, output_dir: str = "docs/debug_output") -> Dict[str, Any]:
    api_key = os.getenv("VLM_API_KEY")
    if not api_key:
        raise ValueError("VLM_API_KEY environment variable is required.")

    img = Image.open(image_path)
    logger.info(f"Processing image: {image_path} (Original: {img.size})")

    # Encode exactly as sightAI does
    vlm_bytes, mime_type, opt_w, opt_h = encode_vlm_image_part(img, MAX_ANALYSIS_DIMENSION)
    logger.info(f"Encoded for Gemini: {opt_w}x{opt_h}, {len(vlm_bytes)/1024:.1f} KB")

    # Load canonical base image from the exact bytes Gemini sees
    gemini_base_img = Image.open(io.BytesIO(vlm_bytes))

    image_part = types.Part.from_bytes(data=vlm_bytes, mime_type=mime_type)
    client = genai.Client(api_key=api_key)

    # -------------------------------------------------------------
    # STAGE 1: Main Analysis Call
    # -------------------------------------------------------------
    logger.info("Executing STAGE 1: Main Grounded Analysis...")
    user_prompt = (
        "Perform strict visual verification, physical instance counting, independent attribute analysis, "
        "TIGHT bounding box localization, and scene classification on this image. "
        "Provide 'box_2d': [ymin, xmin, ymax, xmax] normalized to 0-1000 for each detected object instance "
        "(ymin/ymax vertical, xmin/xmax horizontal). Each box MUST tightly fit the visible object."
    )
    config = types.GenerateContentConfig(
        system_instruction=GROUNDED_VISION_SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema=GroundedAnalysisResult,
        temperature=0.0,
        max_output_tokens=8192,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    response = client.models.generate_content(
        model=PRIMARY_VLM_MODEL,
        contents=[image_part, user_prompt],
        config=config,
    )
    raw_text = response.text
    logger.info(f"Raw Main Response Length: {len(raw_text)} chars")

    clean_json = clean_json_text(raw_text)
    raw_data = json.loads(clean_json)
    normalized_data = normalize_grounded_analysis(raw_data)
    stage1_result = GroundedAnalysisResult.model_validate(normalized_data)

    stage1_boxes = []
    for cat_idx, cat in enumerate(stage1_result.objects):
        for inst in cat.instances:
            if inst.bounding_box:
                stage1_boxes.append({
                    "label": f"{cat.name}:{inst.id}",
                    "box": {
                        "x_min": inst.bounding_box.x_min,
                        "y_min": inst.bounding_box.y_min,
                        "x_max": inst.bounding_box.x_max,
                        "y_max": inst.bounding_box.y_max
                    },
                    "color_idx": cat_idx
                })

    stem = os.path.splitext(os.path.basename(img_path))[0]
    draw_boxes_on_image(
        gemini_base_img,
        stage1_boxes,
        os.path.join(output_dir, f"{stem}_stage1_main_raw.png"),
        title="Stage 1: Raw Main Analysis Output"
    )
    draw_boxes_on_image(
        gemini_base_img,
        stage1_boxes,
        os.path.join(output_dir, "stage1_main_raw.png"),
        title="Stage 1: Raw Main Analysis Output"
    )

    # -------------------------------------------------------------
    # STAGE 2: Sanitization (Deduplication & Containment Rules)
    # -------------------------------------------------------------
    logger.info("Executing STAGE 2: Sanitization...")
    # Clone for stage 2
    stage2_result = GroundedAnalysisResult.model_validate(stage1_result.model_dump())
    stage2_result = sanitize_bounding_boxes(stage2_result)

    stage2_boxes = []
    stage2_loss_report = []
    for cat_idx, cat in enumerate(stage2_result.objects):
        for inst in cat.instances:
            orig_match = next((b for b in stage1_boxes if b["label"] == f"{cat.name}:{inst.id}"), None)
            if inst.bounding_box:
                stage2_boxes.append({
                    "label": f"{cat.name}:{inst.id}",
                    "box": {
                        "x_min": inst.bounding_box.x_min,
                        "y_min": inst.bounding_box.y_min,
                        "x_max": inst.bounding_box.x_max,
                        "y_max": inst.bounding_box.y_max
                    },
                    "color_idx": cat_idx
                })
            elif orig_match is not None:
                stage2_loss_report.append({
                    "instance": f"{cat.name}:{inst.id}",
                    "reason": "Nullified in sanitize_bounding_boxes (NMS/Containment/Tiny Box)",
                    "original_box": orig_match["box"]
                })

    draw_boxes_on_image(
        gemini_base_img,
        stage2_boxes,
        os.path.join(output_dir, f"{stem}_stage2_sanitized.png"),
        title="Stage 2: Sanitized Boxes"
    )
    draw_boxes_on_image(
        gemini_base_img,
        stage2_boxes,
        os.path.join(output_dir, "stage2_sanitized.png"),
        title="Stage 2: Sanitized Boxes"
    )

    # -------------------------------------------------------------
    # STAGE 3: Dedicated Localization Call
    # -------------------------------------------------------------
    logger.info("Executing STAGE 3: Dedicated Localization Call...")
    categories = [(c.name.strip().lower(), max(len(c.instances), c.confirmed_count)) for c in stage1_result.objects if c.instances]
    dedicated_candidates = []
    try:
        raw_candidates = localize_objects(image_part, api_key, categories, PRIMARY_VLM_MODEL)
        for idx, cand in enumerate(raw_candidates):
            dedicated_candidates.append({
                "label": cand["label"],
                "box": cand["box"],
                "color_idx": idx
            })
    except Exception as e:
        logger.error(f"Dedicated localization failed: {e}")

    draw_boxes_on_image(
        gemini_base_img,
        dedicated_candidates,
        os.path.join(output_dir, f"{stem}_stage3_dedicated_detect.png"),
        title="Stage 3: Dedicated Detection Candidates"
    )
    draw_boxes_on_image(
        gemini_base_img,
        dedicated_candidates,
        os.path.join(output_dir, "stage3_dedicated_detect.png"),
        title="Stage 3: Dedicated Detection Candidates"
    )

    # -------------------------------------------------------------
    # STAGE 4: Final Refinement & Reconciliation
    # -------------------------------------------------------------
    logger.info("Executing STAGE 4: Refinement & Reconciliation...")
    stage4_result = refine_bounding_boxes(stage2_result, image_part, api_key, PRIMARY_VLM_MODEL)
    stage4_result = sanitize_bounding_boxes(stage4_result)
    stage4_result = reconcile_unboxed_instances(stage4_result)

    stage4_boxes = []
    for cat_idx, cat in enumerate(stage4_result.objects):
        for inst in cat.instances:
            if inst.bounding_box:
                stage4_boxes.append({
                    "label": f"{cat.name}:{inst.id}",
                    "box": {
                        "x_min": inst.bounding_box.x_min,
                        "y_min": inst.bounding_box.y_min,
                        "x_max": inst.bounding_box.x_max,
                        "y_max": inst.bounding_box.y_max
                    },
                    "color_idx": cat_idx
                })

    draw_boxes_on_image(
        gemini_base_img,
        stage4_boxes,
        os.path.join(output_dir, f"{stem}_stage4_final_api.png"),
        title="Stage 4: Final API Output Boxes"
    )
    draw_boxes_on_image(
        gemini_base_img,
        stage4_boxes,
        os.path.join(output_dir, "stage4_final_api.png"),
        title="Stage 4: Final API Output Boxes"
    )

    # Print Loss Audit Table
    print("\n" + "="*80)
    print("PIPELINE AUDIT: INSTANCE BOX LOSS REPORT")
    print("="*80)
    print(f"Stage 1 (Raw Main Analysis):     {len(stage1_boxes)} boxes")
    print(f"Stage 2 (After Sanitize):         {len(stage2_boxes)} boxes")
    print(f"Stage 3 (Dedicated Candidates):   {len(dedicated_candidates)} boxes")
    print(f"Stage 4 (Final API Result):       {len(stage4_boxes)} boxes")
    print("-" * 80)
    if stage2_loss_report:
        print("Instances dropped between Stage 1 and Stage 2:")
        for r in stage2_loss_report:
            print(f"  - {r['instance']}: {r['reason']} | Box: {r['original_box']}")
    else:
        print("Zero instances dropped between Stage 1 and Stage 2.")
    print("="*80 + "\n")

    return {
        "stage1_boxes": stage1_boxes,
        "stage2_boxes": stage2_boxes,
        "stage3_boxes": dedicated_candidates,
        "stage4_boxes": stage4_boxes,
        "loss_report": stage2_loss_report,
        "raw_json": raw_data
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tools/debug_boxes.py <path_to_image> [output_dir]")
        sys.exit(1)

    img_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "docs/debug_output"
    run_debug_pipeline(img_path, out_dir)
