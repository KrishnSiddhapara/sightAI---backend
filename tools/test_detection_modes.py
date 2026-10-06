"""
test_detection_modes.py - Evaluates different dedicated detection architectures on our 3 benchmark scenes.
"""

import os
import sys
import json
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from google import genai
from google.genai import types
from PIL import Image
import io
from pydantic import BaseModel, Field

from utils.coordinate_utils import calculate_iou, parse_gemini_box_2d
from utils.image_validation import encode_vlm_image_part
from utils.config import MAX_ANALYSIS_DIMENSION
from tests.eval.eval_pipeline import TEST_CASES, evaluate_predictions


class SingleDetection(BaseModel):
    label: str = Field(description="Object name")
    box_2d: List[int] = Field(description="[ymin, xmin, ymax, xmax] 0-1000")


class DetectionEnvelope(BaseModel):
    detections: List[SingleDetection]


def run_mode_eval(mode_name: str, config_fn) -> Dict[str, Any]:
    api_key = os.getenv("VLM_API_KEY")
    client = genai.Client(api_key=api_key)

    total_gt = 0
    total_pred = 0
    total_matched = 0
    all_ious = []
    t_start = time.perf_counter()

    for tc in TEST_CASES:
        img = Image.open(tc["image_path"])
        vlm_bytes, mime_type, opt_w, opt_h = encode_vlm_image_part(img, MAX_ANALYSIS_DIMENSION)
        image_part = types.Part.from_bytes(data=vlm_bytes, mime_type=mime_type)

        with open(tc["gt_path"], "r") as f:
            gt_data = json.load(f)

        prompt, config = config_fn(tc)
        try:
            res = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[image_part, prompt],
                config=config
            )
            raw = res.text
            data = json.loads(raw)
            if isinstance(data, dict):
                data = data.get("detections") or data.get("objects") or []

            predictions = []
            for item in data:
                b2d = item.get("box_2d")
                box = parse_gemini_box_2d(b2d)
                if box:
                    predictions.append({
                        "label": item.get("label", "object"),
                        "box": box
                    })

            metrics = evaluate_predictions(predictions, gt_data)
            total_gt += metrics["n_gt"]
            total_pred += metrics["n_pred"]
            total_matched += metrics["matched"]
            all_ious.extend(metrics["matched_ious"])
        except Exception as e:
            print(f"Error in {mode_name} on {tc['name']}: {e}")

    elapsed = time.perf_counter() - t_start
    recall = total_matched / total_gt if total_gt else 0.0
    mean_iou = sum(all_ious) / len(all_ious) if all_ious else 0.0
    pct_05 = sum(1 for i in all_ious if i >= 0.5) / len(all_ious) if all_ious else 0.0

    print(f"\n[{mode_name}] ({elapsed:.1f}s)")
    print(f"  Recall: {recall*100:.1f}% | Mean IoU: {mean_iou:.3f} | IoU>=0.5: {pct_05*100:.1f}% | Pred: {total_pred} (GT: {total_gt})")
    return {
        "mode": mode_name,
        "recall": recall,
        "mean_iou": mean_iou,
        "pct_05": pct_05,
        "elapsed": elapsed
    }


# Configuration Variations to A/B Test:
def config_freeform_unconstrained(tc):
    prompt = (
        "Detect all distinct physical objects in this image.\n"
        "Return a JSON list. Each item must have: {\"box_2d\": [ymin, xmin, ymax, xmax], \"label\": str} "
        "where coordinates are integers normalized to 0-1000 (ymin/ymax vertical, xmin/xmax horizontal).\n"
        "Rules:\n"
        "1. Provide tight bounding boxes hugging the visible physical edges.\n"
        "2. One box per unique physical object.\n"
        "3. Detect all items including small items, accessories, figurines, containers, and tools.\n"
        "4. Do NOT detect empty background surfaces or tables."
    )
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.0,
        thinking_config=types.ThinkingConfig(thinking_budget=0)
    )
    return prompt, config


def config_pydantic_envelope(tc):
    prompt = (
        "Detect all distinct physical objects in this image.\n"
        "Return a JSON object with 'detections': list of objects with 'label' and 'box_2d' [ymin, xmin, ymax, xmax] 0-1000.\n"
        "Provide tight bounding boxes for each physical object."
    )
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=DetectionEnvelope,
        temperature=0.0,
        thinking_config=types.ThinkingConfig(thinking_budget=0)
    )
    return prompt, config


def config_category_guided(tc):
    # Load categories from GT
    with open(tc["gt_path"], "r") as f:
        gt = json.load(f)
    labels = list(set(item["label"] for item in gt))
    prompt = (
        f"Detect every physical instance of these categories in this image: {', '.join(labels)}.\n"
        "Return a JSON list. Each item: {\"box_2d\": [ymin, xmin, ymax, xmax], \"label\": str} (0-1000 scale, Y-first).\n"
        "Provide tight bounding boxes around every instance."
    )
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.0,
        thinking_config=types.ThinkingConfig(thinking_budget=0)
    )
    return prompt, config


if __name__ == "__main__":
    print("Running A/B Detection Architecture Experiments...")
    run_mode_eval("Free-form JSON (Detect All)", config_freeform_unconstrained)
    run_mode_eval("Pydantic Envelope Schema", config_pydantic_envelope)
    run_mode_eval("Category-Guided Detection", config_category_guided)
