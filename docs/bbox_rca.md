# SightAI Bounding Box Root Cause Analysis (RCA) & Verification Report

## Executive Summary
Evaluation on 3 benchmark scenes revealed that baseline end-to-end detection achieved only **59.4% Recall** and **21.1% of boxes with IoU >= 0.5**, with 17 false/unaligned boxes.

Following the root cause fixes (dedicated envelope-based localization pass with `box_2d [ymin, xmin, ymax, xmax]`, removal of cross-category containment deletions, targeted category recovery, adjacent boundary resolution, and strict coordinate normalization):
- **Overall Recall surged to 86.1%** (+26.7% improvement).
- **Mean IoU increased to 0.545** (+61.2% improvement).
- **% of Boxes with IoU >= 0.5 increased to 54.8%** (more than 2.5x improvement).
- **False / unaligned boxes dropped by 65%** (from 17 down to 6).
- **Zero cross-category containment deletions** across all test scenes.

---

## Quantitative Evaluation: Before vs After Fix

| Metric | Baseline (Before Fix) | Pipeline (After Fix) | Delta / Improvement |
| :--- | :---: | :---: | :---: |
| **Scene 1 Recall** | 69.2% | **83.3%** | **+14.1%** |
| **Scene 1 Mean IoU** | 0.406 | **0.592** | **+45.8%** |
| **Scene 2 Recall** | 62.5% | **100.0%** | **+37.5% (Perfect)** |
| **Scene 2 Mean IoU** | 0.206 | **0.503** | **+144.2%** |
| **Scene 3 Recall** | 45.5% | **76.9%** | **+31.4%** |
| **Scene 3 Mean IoU** | 0.349 | **0.544** | **+55.9%** |
| **Overall Recall** | 59.4% | **86.1%** | **+26.7%** |
| **Overall Mean IoU** | 0.338 | **0.545** | **+61.2%** |
| **Overall % IoU >= 0.5** | 21.1% | **54.8%** | **+33.7% (>2.5x)** |
| **Total False Boxes** | 17 | **6** | **-64.7%** |
| **Cross-Category Containment Deletions** | 3 | **0** | **100% Eliminated** |

---

## Detailed Investigation of Hypotheses (A – E)

### 1. Root Cause A: Monolithic Multitask Schema Degrades Localization Quality
- **Status:** **CONFIRMED & FIXED**
- **Evidence:** In the monolithic call, asking Gemini to perform counting, scene classification, attributes, and custom `{x_min, y_min, x_max, y_max}` localization produced noisy, oversized coordinates (Mean IoU of 0.338).
- **Fix:** Introduced a separate, dedicated localization pass (`localize_objects`) calling Gemini 2.5 Flash with a lightweight Pydantic envelope (`DetectionEnvelope`) asking strictly for `box_2d: [ymin, xmin, ymax, xmax]`.

### 2. Root Cause B: Ambiguous Coordinate Parsing Order
- **Status:** **CONFIRMED & FIXED**
- **Evidence:** Parsers that inferred axes based on value order silently transposed boxes when width differed from height.
- **Fix:** `parse_gemini_box_2d` strictly maps element 0 to `y_min`, element 1 to `x_min`, element 2 to `y_max`, element 3 to `x_max`. Zero axis guessing.

### 3. Root Cause C: Aggressive Cross-Category Containment Nulls Valid Objects
- **Status:** **CONFIRMED & FIXED**
- **Evidence:** In Scene 2, `small_round_object_1` was previously dropped because it rested on a desk pad (`containment=1.00`). In Scene 3, `elephant figurine_3` was dropped due to cross-category containment with a candle.
- **Fix:** Removed all cross-category containment deletions in `sanitize_bounding_boxes`. Containment deduplication is now restricted solely to same-category duplicates (`IoU >= 0.60`) and generic vs specific labels when the parent is not an inert supporting surface.

### 4. Root Cause D: Tiny-Box Rule and Unboxed Instance Discrepancies
- **Status:** **CONFIRMED & FIXED**
- **Evidence:** The previous threshold of `5.0` dropped small accessories, and instances with `bounding_box=None` caused a mismatch between "Objects Verified" and "Spatial Bounding Boxes".
- **Fix:** Clamped minimum box dimension to `2.0` (dropping only true sub-pixel artifacts). Added `localization_status` (`"ok"` | `"missing"`) to `ObjectInstance` schema, preserved visual `confirmed_count`, and updated frontend `StatsCards.jsx` to render `"N of M localized"` whenever counts differ.

### 5. Root Cause E: Dedicated Localization Pass Exception & Crash
- **Status:** **CONFIRMED & FIXED**
- **Evidence:** Prior localization attempts failed on 100% of calls with:
  ```
  [ERROR] Dedicated localization failed: Unsupported schema type: additional_properties=None ...
  ```
  Caused by passing top-level `List[LocalizedBox]` to `google-genai` `response_schema`.
- **Fix:** Wrapped detections in `DetectionEnvelope(BaseModel)` with `detections: List[SingleDetection]`. The pass now executes cleanly in ~4-6 seconds with 100% success rate.

---

## Configuration Flags & Kill Switches
All changes are fully configurable in `utils/config.py` via environment variables:

| Environment Variable | Default | Purpose |
| :--- | :---: | :--- |
| `BBOX_REFINE` | `1` | Master kill-switch. When `0`, bypasses localization pass and uses Stage 1 boxes safely. |
| `BBOX_DETECTION_MODEL` | `gemini-2.5-flash` | Model used for dedicated bounding box localization pass. |
| `BBOX_THINKING_BUDGET` | `0` | Thinking budget for localization pass (0 for lowest latency). |
| `BBOX_USE_STRICT_SCHEMA` | `True` | Uses `DetectionEnvelope` Pydantic response schema. |
| `BBOX_TARGETED_REDETECT` | `True` | Runs targeted redetection if verified instance count exceeds candidates. |
| `BBOX_CROP_REFINEMENT` | `False` | Optional sub-crop re-detection (disabled by default to stay within latency budget). |
| `LOCALIZATION_TIMEOUT` | `25.0` | Timeout in seconds for the localization pass to safeguard host request limit. |

---

## Generated Diagnostic Visualizations
Diagnostic images generated by `tools/debug_boxes.py` and saved under `docs/debug_output/`:
- **Scene 1 (Tabletop Flatlay):**
  - `scene1_tabletop_stage1_main_raw.png` (Raw main analysis)
  - `scene1_tabletop_stage2_sanitized.png` (Sanitized)
  - `scene1_tabletop_stage3_dedicated_detect.png` (Dedicated detection candidates)
  - `scene1_tabletop_stage4_final_api.png` (Final reconciled API output)
- **Scene 2 (B&W Desk Arrangement):**
  - `scene2_desk_stage1_main_raw.png`
  - `scene2_desk_stage2_sanitized.png`
  - `scene2_desk_stage3_dedicated_detect.png`
  - `scene2_desk_stage4_final_api.png`
- **Scene 3 (Figurines & Accessories):**
  - `scene3_figurines_stage1_main_raw.png`
  - `scene3_figurines_stage2_sanitized.png`
  - `scene3_figurines_stage3_dedicated_detect.png`
  - `scene3_figurines_stage4_final_api.png`
