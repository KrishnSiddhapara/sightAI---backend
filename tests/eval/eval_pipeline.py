"""
eval_pipeline.py - Quantitative evaluation script for SightAI bounding box pipeline.
Computes Recall, Mean IoU, % IoU >= 0.5, and False Boxes against ground truth annotations.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any, Tuple

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from utils.coordinate_utils import calculate_iou
from services.vision import labels_match
from utils.image_validation import encode_vlm_image_part
from utils.config import MAX_ANALYSIS_DIMENSION, PRIMARY_VLM_MODEL
from services.vision import analyze_image_grounded
from google.genai import types
from PIL import Image

logging.basicConfig(level=logging.WARNING)

TEST_CASES = [
    {
        "name": "Scene 1 (Tabletop Flatlay)",
        "image_path": "tests/eval/images/scene1_tabletop.png",
        "gt_path": "tests/eval/ground_truth/scene1_tabletop.json"
    },
    {
        "name": "Scene 2 (B&W Desk Arrangement)",
        "image_path": "tests/eval/images/scene2_desk.png",
        "gt_path": "tests/eval/ground_truth/scene2_desk.json"
    },
    {
        "name": "Scene 3 (Figurines & Accessories)",
        "image_path": "tests/eval/images/scene3_figurines.png",
        "gt_path": "tests/eval/ground_truth/scene3_figurines.json"
    }
]


def evaluate_predictions(
    predictions: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    iou_thresh: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluates predicted boxes against ground truth.
    predictions: list of {'label': str, 'box': {'x_min','y_min','x_max','y_max'}}
    ground_truth: list of {'label': str, 'box': {'x_min','y_min','x_max','y_max'}}
    """
    n_gt = len(ground_truth)
    n_pred = len(predictions)

    if n_gt == 0:
        return {"recall": 1.0, "mean_iou": 0.0, "pct_iou_ge_05": 0.0, "false_boxes": n_pred}

    matched_gt = set()
    matched_preds = set()
    ious = []

    # Greedy bipartite matching by maximum IoU
    candidate_matches = []
    for pred_idx, pred in enumerate(predictions):
        p_box = pred["box"]
        p_label = pred.get("label", "").lower()
        for gt_idx, gt in enumerate(ground_truth):
            g_box = gt["box"]
            g_label = gt.get("label", "").lower()
            iou = calculate_iou(p_box, g_box)
            if iou > 0.05:
                # Prioritize label agreement
                label_bonus = 0.2 if labels_match(p_label, g_label) else 0.0
                candidate_matches.append((iou + label_bonus, iou, pred_idx, gt_idx))

    # Sort descending by match score
    candidate_matches.sort(key=lambda x: x[0], reverse=True)

    for _, iou, pred_idx, gt_idx in candidate_matches:
        if pred_idx not in matched_preds and gt_idx not in matched_gt:
            matched_preds.add(pred_idx)
            matched_gt.add(gt_idx)
            ious.append(iou)

    matched_count = len(matched_gt)
    recall = matched_count / n_gt
    mean_iou = sum(ious) / len(ious) if ious else 0.0
    pct_ge_05 = sum(1 for i in ious if i >= iou_thresh) / len(ious) if ious else 0.0
    false_boxes = n_pred - len(matched_preds)

    return {
        "n_gt": n_gt,
        "n_pred": n_pred,
        "matched": matched_count,
        "recall": recall,
        "mean_iou": mean_iou,
        "pct_iou_ge_05": pct_ge_05,
        "false_boxes": false_boxes,
        "matched_ious": ious
    }


def run_evaluation() -> Dict[str, Any]:
    api_key = os.getenv("VLM_API_KEY")
    if not api_key:
        raise ValueError("VLM_API_KEY required.")

    results = []
    for tc in TEST_CASES:
        img = Image.open(tc["image_path"])
        vlm_bytes, mime_type, opt_w, opt_h = encode_vlm_image_part(img, MAX_ANALYSIS_DIMENSION)
        image_part = types.Part.from_bytes(data=vlm_bytes, mime_type=mime_type)

        with open(tc["gt_path"], "r") as f:
            gt_data = json.load(f)

        # Run pipeline
        res = analyze_image_grounded(image_part, api_key, image_width=opt_w, image_height=opt_h, pil_image=img)

        predictions = []
        for cat in res.objects:
            for inst in cat.instances:
                if inst.bounding_box:
                    predictions.append({
                        "label": cat.name,
                        "box": {
                            "x_min": inst.bounding_box.x_min,
                            "y_min": inst.bounding_box.y_min,
                            "x_max": inst.bounding_box.x_max,
                            "y_max": inst.bounding_box.y_max
                        }
                    })

        metrics = evaluate_predictions(predictions, gt_data)
        metrics["name"] = tc["name"]
        results.append(metrics)

    print("\n" + "=" * 90)
    print("SIGHTAI BOUNDING BOX EVALUATION METRICS REPORT")
    print("=" * 90)
    print(f"{'Image / Test Scene':<35} | {'GT':<4} | {'Pred':<5} | {'Recall':<8} | {'Mean IoU':<9} | {'IoU>=0.5':<9} | {'False':<5}")
    print("-" * 90)
    total_gt = sum(r["n_gt"] for r in results)
    total_pred = sum(r["n_pred"] for r in results)
    total_matched = sum(r["matched"] for r in results)
    all_ious = [iou for r in results for iou in r["matched_ious"]]

    for r in results:
        print(f"{r['name']:<35} | {r['n_gt']:<4} | {r['n_pred']:<5} | {r['recall']*100:6.1f}% | {r['mean_iou']:8.3f} | {r['pct_iou_ge_05']*100:7.1f}% | {r['false_boxes']:<5}")
    print("-" * 90)
    overall_recall = total_matched / total_gt if total_gt else 0.0
    overall_mean_iou = sum(all_ious) / len(all_ious) if all_ious else 0.0
    overall_pct_05 = sum(1 for i in all_ious if i >= 0.5) / len(all_ious) if all_ious else 0.0
    total_false = sum(r["false_boxes"] for r in results)
    print(f"{'OVERALL AVERAGE':<35} | {total_gt:<4} | {total_pred:<5} | {overall_recall*100:6.1f}% | {overall_mean_iou:8.3f} | {overall_pct_05*100:7.1f}% | {total_false:<5}")
    print("=" * 90 + "\n")

    return {
        "overall_recall": overall_recall,
        "overall_mean_iou": overall_mean_iou,
        "overall_pct_05": overall_pct_05,
        "results": results
    }


if __name__ == "__main__":
    run_evaluation()
