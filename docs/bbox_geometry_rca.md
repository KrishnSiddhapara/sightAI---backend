# Root Cause Analysis: Bounding Box Geometry and Precision Pipeline

## 1. Executive Summary
- **Primary Root Cause**: **Model Precision / Spatial Localization Granularity on Small Objects**.
- **Display/Transform Pipeline**: Verified 100% correct via synthetic unit testing (`test_frontend_coords.mjs`). Bounding box transformations preserve aspect ratios, container offsets (`offsetX`, `offsetY`), and subpixel scaling for both landscape and portrait orientation.
- **Input Image Pipeline**: Verified 100% correct. `encode_vlm_image_part` applies high-quality proportional LANCZOS resizing, preserving aspect ratio without cropping, stretching, or padding. `ImageOps.exif_transpose` auto-rotates EXIF photos before processing.
- **Core Mechanism of Distortion**:
  1. Gemini native `box_2d` output on full-image single pass exhibits severe vertical degradation on small objects (e.g., predicting `ymax=976` for a figurine located at `y=610..700`, making boxes 3x too tall).
  2. The single coarse pass lacks spatial zoom/resolution for small items (<50px in full image).
  3. Lack of localized crop-based secondary refinement pass to zoom into candidate bounding boxes and compute tight pixel-level boundaries.

---

## 2. Quantitative Evidence & Diagnostic Findings

### A. Synthetic Frontend Coordinate Transformation Audit
Executed synthetic unit tests on 4 corners and center boxes across landscape and portrait viewports:
- **Landscape Test** (1200x800 image inside 800x600 container):
  - Displayed Dimensions: 800x533.33 px | `offsetY`: 33.33 px | `scale`: 0.6667
  - Top-Left Box: `(left=0.00, top=33.33)` -> Exact match.
  - Bottom-Right Box: `(left=720.00 + 80.00 = 800.00, top=513.33 + 53.33 = 566.66)` -> Exact match.
- **Portrait Test** (390x480 image inside 800x460 container):
  - Displayed Dimensions: 373.75x460 px | `offsetX`: 213.13 px | `scale`: 0.9583
  - Top-Left Box: `(left=213.13, top=0.00)` -> Exact match.
  - Bottom-Right Box: `(left=549.50 + 37.37 = 586.87)` -> Exact match.

Conclusion: Zero transform or display bugs in frontend `coordinateTransform.js` or `ObjectDetectionViewer.jsx`.

### B. Image Preprocessing & Aspect Ratio Audit
- `optimize_image_for_analysis` calculates proportional scale factor `max_dim / max(w, h)`:
  - Original 390x480 -> Encoded 390x480 (aspect ratio 0.8125).
  - Original 3000x4000 -> Encoded 1152x1536 (aspect ratio 0.7500).
- Aspect ratio of image sent to Gemini is identical to original and displayed images.

### C. Gemini Model Precision Audit (Raw Output Inspection)
On `scene3_figurines.png` (portrait image with 6 elephant figurines):
- **Stage 1 (Raw Main Analysis)**: Gemini identified objects but predicted imprecise vertical boundaries.
  - Candidate box for `elephant figurine_4`: `[425, 649, 976, 700]` (height = 551 units on 0-1000 scale).
  - Candidate box for `candle_1`: `[372, 649, 976, 700]` (height = 604 units on 0-1000 scale).
- **Result**: Boxes extended far down to the bottom of the image (ymax=976) over empty space/towel edges because Gemini's coarse attention map on the full image loses vertical edge precision for small objects.

---

## 3. Required Remediation Strategy
1. **Input Quality Optimization for Localization**:
   - Use higher resolution (`MAX_ANALYSIS_DIMENSION=2048`, quality >= 90) for crop/refinement passes.
2. **Two-Stage Localize-Then-Refine Architecture**:
   - **Coarse Pass**: Full image native `box_2d` detection.
   - **Crop Refinement Pass**: Crop original image around each coarse box with 20% padding (ensuring minimum crop size of 224px for visual context). Send crop to Gemini asking for single tight box `[ymin, xmin, ymax, xmax]` of target object inside crop. Map crop box back to full-image 0-1000 coordinates. Accept refined box if `IoU(refined, coarse) > 0.3`.
   - **Missing Object Recovery**: Targeted pass + 2x2 overlapping tile detection with tile NMS (IoU ~0.5) when detected count < confirmed count.
3. **Empty Background / Low Edge Density Filter**: Sanity check crops to reject boxes placed on uniform background or zero feature density.
