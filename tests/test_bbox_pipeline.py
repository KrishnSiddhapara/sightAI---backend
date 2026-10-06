import unittest
from unittest.mock import MagicMock, patch

from services.schemas import (
    GroundedAnalysisResult,
    DetectedObjectCategory,
    ObjectInstance,
    InstanceAttributes,
    BoundingBox,
    SceneDescription
)
from utils.coordinate_utils import (
    validate_bbox_coords,
    parse_gemini_box_2d,
    parse_standard_bbox,
    calculate_iou,
    calculate_center_distance
)
from utils.json_utils import normalize_grounded_analysis
from services.vision import (
    refine_bounding_boxes,
    sanitize_bounding_boxes,
    reconcile_unboxed_instances,
    labels_match,
    normalize_label
)


class TestBoundingBoxPipeline(unittest.TestCase):

    def test_gemini_box_2d_parsing_explicit_order(self):
        """
        Gemini box_2d is strictly [ymin, xmin, ymax, xmax] on a 0-1000 grid.
        Verify that element 0 is ALWAYS ymin and element 1 is ALWAYS xmin,
        and no heuristic format guessing occurs even when both orderings are geometrically valid.
        """
        # [ymin, xmin, ymax, xmax]
        # In this example: ymin=150, xmin=50, ymax=450, xmax=600
        # If incorrectly parsed as [xmin, ymin, xmax, ymax], x_min would be 150 and y_min would be 50.
        raw_box = [150, 50, 450, 600]
        parsed = parse_gemini_box_2d(raw_box)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed['x_min'], 50.0)
        self.assertEqual(parsed['y_min'], 150.0)
        self.assertEqual(parsed['x_max'], 600.0)
        self.assertEqual(parsed['y_max'], 450.0)

    def test_gemini_box_2d_scale_0_to_1_normalization(self):
        """Verify 0.0-1.0 normalized coordinates scale properly to 0-1000."""
        raw_box = [0.15, 0.05, 0.45, 0.60]
        parsed = parse_gemini_box_2d(raw_box)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed['x_min'], 50.0)
        self.assertEqual(parsed['y_min'], 150.0)
        self.assertEqual(parsed['x_max'], 600.0)
        self.assertEqual(parsed['y_max'], 450.0)

    def test_standard_bbox_dictionary_parsing(self):
        """Verify dictionary format with explicit keys parses correctly."""
        raw_dict = {"x_min": 100, "y_min": 200, "x_max": 500, "y_max": 600}
        parsed = parse_standard_bbox(raw_dict)
        self.assertEqual(parsed, {'x_min': 100.0, 'y_min': 200.0, 'x_max': 500.0, 'y_max': 600.0})

        # Test camelCase and alternate casing
        raw_camel = {"xmin": 80, "ymin": 90, "xmax": 300, "ymax": 400}
        parsed_camel = parse_standard_bbox(raw_camel)
        self.assertEqual(parsed_camel, {'x_min': 80.0, 'y_min': 90.0, 'x_max': 300.0, 'y_max': 400.0})

    def test_json_utils_explicit_box_2d_vs_bounding_box(self):
        """
        Verify that in json_utils.normalize_grounded_analysis:
        - 'box_2d' is explicitly parsed as Gemini [ymin, xmin, ymax, xmax]
        - 'bounding_box' is explicitly parsed as standard {x_min, y_min, ...}
        """
        payload = {
            "objects": [
                {
                    "name": "pen",
                    "instances": [
                        {
                            "id": "pen_1",
                            "box_2d": [100, 200, 300, 400]  # ymin=100, xmin=200, ymax=300, xmax=400
                        },
                        {
                            "id": "pen_2",
                            "bounding_box": {"x_min": 200, "y_min": 100, "x_max": 400, "y_max": 300}
                        }
                    ]
                }
            ],
            "scene": {"environment": "Desk"},
            "overall_summary": "Two pens."
        }
        normalized = normalize_grounded_analysis(payload)
        res = GroundedAnalysisResult.model_validate(normalized)
        inst1_box = res.objects[0].instances[0].bounding_box
        inst2_box = res.objects[0].instances[1].bounding_box

        # Both representations should produce identical canonical coordinates
        self.assertEqual(inst1_box.x_min, 200.0)
        self.assertEqual(inst1_box.y_min, 100.0)
        self.assertEqual(inst1_box.x_max, 400.0)
        self.assertEqual(inst1_box.y_max, 300.0)

        self.assertEqual(inst2_box.x_min, 200.0)
        self.assertEqual(inst2_box.y_min, 100.0)
        self.assertEqual(inst2_box.x_max, 400.0)
        self.assertEqual(inst2_box.y_max, 300.0)

    def test_safe_label_matching_prevents_unrelated_matches(self):
        """Verify strict label matching prevents false cross-category associations."""
        self.assertTrue(labels_match("pen", "pen"))
        self.assertTrue(labels_match("pens", "pen"))
        self.assertTrue(labels_match("cellphone", "phone"))
        self.assertTrue(labels_match("elephant figurine", "elephant"))

        # MUST NOT match unrelated categories
        self.assertFalse(labels_match("cup", "bottle"))
        self.assertFalse(labels_match("phone", "laptop"))
        self.assertFalse(labels_match("open book", "pen"))
        self.assertFalse(labels_match("car", "scarf"))

    @patch("services.vision.localize_objects")
    def test_single_object_refinement_accepted(self, mock_localize):
        """Test 1: Single object refinement is accepted when consistent."""
        mock_localize.return_value = [
            {"label": "laptop", "box": {'x_min': 105.0, 'y_min': 95.0, 'x_max': 495.0, 'y_max': 505.0}}
        ]
        inst = ObjectInstance(
            id="laptop_1",
            attributes=InstanceAttributes(),
            bounding_box=BoundingBox(x_min=100.0, y_min=100.0, x_max=500.0, y_max=500.0)
        )
        cat = DetectedObjectCategory(name="laptop", confirmed_count=1, instances=[inst])
        res = GroundedAnalysisResult(
            objects=[cat],
            scene=SceneDescription(),
            overall_summary="A laptop."
        )

        refined = refine_bounding_boxes(res, MagicMock(), "fake_key", "fake_model")
        refined_box = refined.objects[0].instances[0].bounding_box
        self.assertEqual(refined_box.x_min, 105.0)
        self.assertEqual(refined_box.y_min, 95.0)
        self.assertEqual(refined_box.x_max, 495.0)
        self.assertEqual(refined_box.y_max, 505.0)

    @patch("services.vision.localize_objects")
    def test_multiple_same_category_objects_geometric_matching(self, mock_localize):
        """
        Test 2: Multiple objects of the same category are matched geometrically
        using IoU and center distance, NOT by reading order.
        """
        # 3 pens at distinct positions
        inst1 = ObjectInstance(id="pen_1", attributes=InstanceAttributes(), bounding_box=BoundingBox(x_min=100, y_min=100, x_max=200, y_max=200))
        inst2 = ObjectInstance(id="pen_2", attributes=InstanceAttributes(), bounding_box=BoundingBox(x_min=500, y_min=500, x_max=600, y_max=600))
        inst3 = ObjectInstance(id="pen_3", attributes=InstanceAttributes(), bounding_box=BoundingBox(x_min=800, y_min=200, x_max=900, y_max=300))

        # Localization returns candidates in reversed/scrambled order
        mock_localize.return_value = [
            {"label": "pen", "box": {'x_min': 802.0, 'y_min': 198.0, 'x_max': 898.0, 'y_max': 302.0}},  # matches pen_3
            {"label": "pen", "box": {'x_min': 98.0, 'y_min': 102.0, 'x_max': 202.0, 'y_max': 198.0}},   # matches pen_1
            {"label": "pen", "box": {'x_min': 502.0, 'y_min': 498.0, 'x_max': 598.0, 'y_max': 602.0}}   # matches pen_2
        ]

        cat = DetectedObjectCategory(name="pen", confirmed_count=3, instances=[inst1, inst2, inst3])
        res = GroundedAnalysisResult(objects=[cat], scene=SceneDescription(), overall_summary="3 pens.")

        refined = refine_bounding_boxes(res, MagicMock(), "fake_key", "fake_model")
        r_insts = refined.objects[0].instances

        # Verify geometric matching correctly assigned each candidate to the right instance
        self.assertAlmostEqual(r_insts[0].bounding_box.x_min, 98.0, delta=1.0)
        self.assertAlmostEqual(r_insts[1].bounding_box.x_min, 502.0, delta=1.0)
        self.assertAlmostEqual(r_insts[2].bounding_box.x_min, 802.0, delta=1.0)

    @patch("services.vision.localize_objects")
    def test_different_objects_no_cross_matching(self, mock_localize):
        """Test 3: Different object categories never cross-match candidates."""
        inst_laptop = ObjectInstance(id="laptop_1", attributes=InstanceAttributes(), bounding_box=BoundingBox(x_min=100, y_min=100, x_max=400, y_max=400))
        inst_phone = ObjectInstance(id="phone_1", attributes=InstanceAttributes(), bounding_box=BoundingBox(x_min=500, y_min=500, x_max=600, y_max=600))
        inst_bottle = ObjectInstance(id="bottle_1", attributes=InstanceAttributes(), bounding_box=BoundingBox(x_min=700, y_min=100, x_max=800, y_max=400))

        mock_localize.return_value = [
            {"label": "bottle", "box": {'x_min': 702, 'y_min': 98, 'x_max': 798, 'y_max': 402}},
            {"label": "laptop", "box": {'x_min': 98, 'y_min': 102, 'x_max': 402, 'y_max': 398}},
            {"label": "phone", "box": {'x_min': 502, 'y_min': 498, 'x_max': 598, 'y_max': 602}},
        ]

        cat1 = DetectedObjectCategory(name="laptop", confirmed_count=1, instances=[inst_laptop])
        cat2 = DetectedObjectCategory(name="phone", confirmed_count=1, instances=[inst_phone])
        cat3 = DetectedObjectCategory(name="bottle", confirmed_count=1, instances=[inst_bottle])
        res = GroundedAnalysisResult(objects=[cat1, cat2, cat3], scene=SceneDescription(), overall_summary="Desk items.")

        refined = refine_bounding_boxes(res, MagicMock(), "fake_key", "fake_model")
        self.assertAlmostEqual(refined.objects[0].instances[0].bounding_box.x_min, 98.0, delta=1.0)
        self.assertAlmostEqual(refined.objects[1].instances[0].bounding_box.x_min, 502.0, delta=1.0)
        self.assertAlmostEqual(refined.objects[2].instances[0].bounding_box.x_min, 702.0, delta=1.0)

    def test_overlapping_objects_preserved_by_conservative_nms(self):
        """
        Test 4: Partially overlapping objects of the same category (e.g. 2 overlapping pens, IoU=0.65)
        are NOT dropped by NMS.
        """
        inst1 = ObjectInstance(id="pen_1", attributes=InstanceAttributes(), bounding_box=BoundingBox(x_min=100, y_min=100, x_max=300, y_max=300))
        inst2 = ObjectInstance(id="pen_2", attributes=InstanceAttributes(), bounding_box=BoundingBox(x_min=150, y_min=150, x_max=350, y_max=350))

        cat = DetectedObjectCategory(name="pen", confirmed_count=2, instances=[inst1, inst2])
        res = GroundedAnalysisResult(objects=[cat], scene=SceneDescription(), overall_summary="2 pens.")

        sanitized = sanitize_bounding_boxes(res)
        self.assertIsNotNone(sanitized.objects[0].instances[0].bounding_box)
        self.assertIsNotNone(sanitized.objects[0].instances[1].bounding_box)

    @patch("services.vision.localize_objects")
    def test_localization_failure_preserves_original_boxes(self, mock_localize):
        """Test 8: If localization raises an error, all original valid boxes are preserved."""
        mock_localize.side_effect = RuntimeError("Gemini API localization timeout")

        inst = ObjectInstance(id="item_1", attributes=InstanceAttributes(), bounding_box=BoundingBox(x_min=100, y_min=100, x_max=400, y_max=400))
        cat = DetectedObjectCategory(name="item", confirmed_count=1, instances=[inst])
        res = GroundedAnalysisResult(objects=[cat], scene=SceneDescription(), overall_summary="Item.")

        refined = refine_bounding_boxes(res, MagicMock(), "fake_key", "fake_model")
        # Original box must remain untouched
        self.assertIsNotNone(refined.objects[0].instances[0].bounding_box)
        self.assertEqual(refined.objects[0].instances[0].bounding_box.x_min, 100.0)

    @patch("services.vision.localize_objects")
    def test_localization_fewer_objects_preserves_unmatched_original(self, mock_localize):
        """
        Test 9: When localization returns fewer objects than the main analysis
        (e.g., 3 pens originally, but localization only finds 2),
        the 3rd pen MUST NOT have its box deleted (must NOT become None)!
        """
        inst1 = ObjectInstance(id="pen_1", attributes=InstanceAttributes(), bounding_box=BoundingBox(x_min=100, y_min=100, x_max=200, y_max=200))
        inst2 = ObjectInstance(id="pen_2", attributes=InstanceAttributes(), bounding_box=BoundingBox(x_min=400, y_min=400, x_max=500, y_max=500))
        inst3 = ObjectInstance(id="pen_3", attributes=InstanceAttributes(), bounding_box=BoundingBox(x_min=700, y_min=700, x_max=800, y_max=800))

        # Localization only detected 2 of the 3 pens
        mock_localize.return_value = [
            {"label": "pen", "box": {'x_min': 102, 'y_min': 98, 'x_max': 198, 'y_max': 202}},
            {"label": "pen", "box": {'x_min': 402, 'y_min': 398, 'x_max': 498, 'y_max': 502}}
        ]

        cat = DetectedObjectCategory(name="pen", confirmed_count=3, instances=[inst1, inst2, inst3])
        res = GroundedAnalysisResult(objects=[cat], scene=SceneDescription(), overall_summary="3 pens.")

        refined = refine_bounding_boxes(res, MagicMock(), "fake_key", "fake_model")
        r_insts = refined.objects[0].instances

        # The first two are refined
        self.assertAlmostEqual(r_insts[0].bounding_box.x_min, 102.0, delta=1.0)
        self.assertAlmostEqual(r_insts[1].bounding_box.x_min, 402.0, delta=1.0)
        # The 3rd pen MUST RETAIN its original valid box!
        self.assertIsNotNone(r_insts[2].bounding_box)
        self.assertEqual(r_insts[2].bounding_box.x_min, 700.0)
        self.assertEqual(r_insts[2].bounding_box.y_min, 700.0)

    @patch("services.vision.localize_objects")
    def test_inconsistent_localization_box_rejected_keeps_original(self, mock_localize):
        """
        Test 10: If a localization candidate is geometrically inconsistent (e.g. far away, IoU=0),
        the candidate is rejected and the original box is preserved.
        """
        inst = ObjectInstance(id="mug_1", attributes=InstanceAttributes(), bounding_box=BoundingBox(x_min=100, y_min=100, x_max=200, y_max=200))

        # Candidate is at bottom-right of the image (center distance > 700px, IoU=0)
        mock_localize.return_value = [
            {"label": "mug", "box": {'x_min': 800, 'y_min': 800, 'x_max': 900, 'y_max': 900}}
        ]

        cat = DetectedObjectCategory(name="mug", confirmed_count=1, instances=[inst])
        res = GroundedAnalysisResult(objects=[cat], scene=SceneDescription(), overall_summary="A mug.")

        refined = refine_bounding_boxes(res, MagicMock(), "fake_key", "fake_model")
        # Inconsistent candidate rejected -> original box preserved
        self.assertEqual(refined.objects[0].instances[0].bounding_box.x_min, 100.0)
        self.assertEqual(refined.objects[0].instances[0].bounding_box.y_min, 100.0)

    def test_bounding_box_coordinate_validation(self):
        """Validate bbox coords helper rejects invalid ranges and inversions."""
        # Valid
        self.assertTrue(validate_bbox_coords(100.0, 100.0, 200.0, 200.0))
        # Inverted X
        self.assertFalse(validate_bbox_coords(300.0, 100.0, 200.0, 200.0))
        # Inverted Y
        self.assertFalse(validate_bbox_coords(100.0, 300.0, 200.0, 200.0))
        # Out of bounds (< 0)
        self.assertFalse(validate_bbox_coords(-10.0, 100.0, 200.0, 200.0))
        # Out of bounds (> 1000)
        self.assertFalse(validate_bbox_coords(100.0, 100.0, 1050.0, 200.0))
        # NaN / Inf
        self.assertFalse(validate_bbox_coords(float('nan'), 100.0, 200.0, 200.0))
        self.assertFalse(validate_bbox_coords(100.0, 100.0, float('inf'), 200.0))

    def test_expanded_aliases_and_stemming(self):
        """Verify expanded aliases and plural/stemming match correctly."""
        self.assertTrue(labels_match("sunglasses", "glasses"))
        self.assertTrue(labels_match("sunglass", "glasses"))
        self.assertTrue(labels_match("computer keyboard", "keyboard"))
        self.assertTrue(labels_match("laptop computer", "laptop"))
        self.assertTrue(labels_match("notebook", "laptop"))
        self.assertTrue(labels_match("headphones", "headphone"))
        self.assertTrue(labels_match("earphones", "headphones"))
        self.assertTrue(labels_match("smartwatch", "watch"))
        self.assertTrue(labels_match("watches", "watch"))

    @patch("services.vision.localize_objects")
    def test_single_pair_skew_recovery(self, mock_localize):
        """
        Test that a single instance of a category recovers from initial coordinate skew
        (e.g. laptop shifted horizontally by 280px) when localization returns the true box.
        """
        # Original shifted box (e.g. squished to center-left)
        inst = ObjectInstance(
            id="laptop_1",
            attributes=InstanceAttributes(),
            bounding_box=BoundingBox(x_min=540.0, y_min=240.0, x_max=740.0, y_max=640.0)
        )
        cat = DetectedObjectCategory(name="laptop", confirmed_count=1, instances=[inst])
        res = GroundedAnalysisResult(objects=[cat], scene=SceneDescription(), overall_summary="Laptop on desk.")

        # True localized box on the right
        mock_localize.return_value = [
            {"label": "laptop", "box": {'x_min': 700.0, 'y_min': 240.0, 'x_max': 980.0, 'y_max': 640.0}}
        ]

        refined = refine_bounding_boxes(res, MagicMock(), "fake_key", "fake_model")
        refined_box = refined.objects[0].instances[0].bounding_box
        # Should be refined to true coordinates
        self.assertEqual(refined_box.x_min, 700.0)
        self.assertEqual(refined_box.x_max, 980.0)

    def test_resolve_adjacent_box_overlaps_horizontal(self):
        """
        Verify that when two adjacent objects (e.g. keyboard and sunglasses) have an accidental
        horizontal boundary overlap (e.g. 20px), their boundary is split to eliminate the overlap.
        """
        inst_glasses = ObjectInstance(
            id="sunglasses_1",
            attributes=InstanceAttributes(),
            bounding_box=BoundingBox(x_min=400.0, y_min=400.0, x_max=520.0, y_max=600.0)
        )
        inst_keyboard = ObjectInstance(
            id="keyboard_1",
            attributes=InstanceAttributes(),
            bounding_box=BoundingBox(x_min=500.0, y_min=400.0, x_max=700.0, y_max=700.0)
        )
        cat1 = DetectedObjectCategory(name="sunglasses", confirmed_count=1, instances=[inst_glasses])
        cat2 = DetectedObjectCategory(name="keyboard", confirmed_count=1, instances=[inst_keyboard])
        res = GroundedAnalysisResult(objects=[cat1, cat2], scene=SceneDescription(), overall_summary="Desk items.")

        sanitized = sanitize_bounding_boxes(res)
        g_box = sanitized.objects[0].instances[0].bounding_box
        k_box = sanitized.objects[1].instances[0].bounding_box

        # Split point should be (520 + 500) / 2 = 510.0
        self.assertEqual(g_box.x_max, 510.0)
        self.assertEqual(k_box.x_min, 510.0)
        # Verify zero horizontal overlap
        self.assertLessEqual(g_box.x_max, k_box.x_min)

    def test_subpart_suppression_coil_in_headphones(self):
        """
        Verify that a small nested sub-part (e.g. 'coil' or 'cable') heavily contained
        inside a whole object (e.g. 'headphones') is suppressed in favor of the whole object.
        """
        inst_headphones = ObjectInstance(
            id="headphones_1",
            attributes=InstanceAttributes(),
            bounding_box=BoundingBox(x_min=200.0, y_min=500.0, x_max=450.0, y_max=750.0)
        )
        inst_coil = ObjectInstance(
            id="coil_1",
            attributes=InstanceAttributes(),
            bounding_box=BoundingBox(x_min=280.0, y_min=550.0, x_max=380.0, y_max=700.0)
        )
        cat1 = DetectedObjectCategory(name="headphones", confirmed_count=1, instances=[inst_headphones])
        cat2 = DetectedObjectCategory(name="coil", confirmed_count=1, instances=[inst_coil])
        res = GroundedAnalysisResult(objects=[cat1, cat2], scene=SceneDescription(), overall_summary="Headphones on desk.")

        sanitized = sanitize_bounding_boxes(res)
        # Headphones box must be preserved
        self.assertIsNotNone(sanitized.objects[0].instances[0].bounding_box)
        # Sub-part coil box must be suppressed
        self.assertIsNone(sanitized.objects[1].instances[0].bounding_box)

    def test_alias_deduplication_usb_adapter(self):
        """
        Verify that overlapping duplicate boxes for the same item under alias names
        (e.g. 'usb' and 'usb adapter') are deduplicated.
        """
        inst_usb = ObjectInstance(
            id="usb_1",
            attributes=InstanceAttributes(),
            bounding_box=BoundingBox(x_min=500.0, y_min=200.0, x_max=550.0, y_max=240.0)
        )
        inst_adapter = ObjectInstance(
            id="usb adapter_1",
            attributes=InstanceAttributes(),
            bounding_box=BoundingBox(x_min=495.0, y_min=198.0, x_max=555.0, y_max=242.0)
        )
        cat1 = DetectedObjectCategory(name="usb", confirmed_count=1, instances=[inst_usb])
        cat2 = DetectedObjectCategory(name="usb adapter", confirmed_count=1, instances=[inst_adapter])
        res = GroundedAnalysisResult(objects=[cat1, cat2], scene=SceneDescription(), overall_summary="Adapter on desk.")

        sanitized = sanitize_bounding_boxes(res)
        boxes = [
            sanitized.objects[0].instances[0].bounding_box,
            sanitized.objects[1].instances[0].bounding_box
        ]
        active_boxes = [b for b in boxes if b is not None]
        # Exactly ONE box must remain
        self.assertEqual(len(active_boxes), 1)

    def test_cross_category_physical_exclusion_candle_vs_figurine(self):
        """
        Verify that when two distinct solid physical objects overlap heavily
        (e.g. candle and an erroneous elephant figurine overlapping on the candle),
        the spurious multi-instance box is dropped and the unique candle is preserved.
        """
        inst_candle = ObjectInstance(
            id="candle_1",
            attributes=InstanceAttributes(),
            bounding_box=BoundingBox(x_min=400.0, y_min=500.0, x_max=500.0, y_max=650.0)
        )
        # Elephant figurine 2 heavily overlaps the exact candle location
        inst_elephant1 = ObjectInstance(
            id="elephant figurine_1",
            attributes=InstanceAttributes(),
            bounding_box=BoundingBox(x_min=700.0, y_min=100.0, x_max=800.0, y_max=300.0)
        )
        inst_elephant2 = ObjectInstance(
            id="elephant figurine_2",
            attributes=InstanceAttributes(),
            bounding_box=BoundingBox(x_min=395.0, y_min=510.0, x_max=495.0, y_max=645.0)
        )
        cat_candle = DetectedObjectCategory(name="candle", confirmed_count=1, instances=[inst_candle])
        cat_elephant = DetectedObjectCategory(name="elephant figurine", confirmed_count=2, instances=[inst_elephant1, inst_elephant2])
        res = GroundedAnalysisResult(objects=[cat_candle, cat_elephant], scene=SceneDescription(), overall_summary="Candle and figurines.")

        sanitized = sanitize_bounding_boxes(res)
        # Candle box must remain intact
        self.assertIsNotNone(sanitized.objects[0].instances[0].bounding_box)
        # Legitimate elephant figurine 1 must remain intact
        self.assertIsNotNone(sanitized.objects[1].instances[0].bounding_box)
        # Spurious elephant figurine 2 on candle must be suppressed
        self.assertIsNone(sanitized.objects[1].instances[1].bounding_box)

    def test_reconcile_unboxed_instances_synchronizes_counts(self):
        """
        Verify that reconcile_unboxed_instances verifies and synchronizes confirmed_count.
        """
        inst1 = ObjectInstance(id="elephant_1", attributes=InstanceAttributes(), bounding_box=BoundingBox(x_min=100, y_min=100, x_max=200, y_max=200))
        inst2 = ObjectInstance(id="elephant_2", attributes=InstanceAttributes(), bounding_box=BoundingBox(x_min=200, y_min=200, x_max=300, y_max=300))
        inst3 = ObjectInstance(id="elephant_3", attributes=InstanceAttributes(), bounding_box=BoundingBox(x_min=300, y_min=300, x_max=400, y_max=400))

        # Category with 3 boxed instances but confirmed_count was erroneously 1
        cat = DetectedObjectCategory(name="elephant figurine", confirmed_count=1, instances=[inst1, inst2, inst3])
        res = GroundedAnalysisResult(objects=[cat], scene=SceneDescription(), overall_summary="3 figurines.")

        reconciled = reconcile_unboxed_instances(res)
        # Confirmed count must be updated to 3 to match the boxed instances
        self.assertEqual(reconciled.objects[0].confirmed_count, 3)
        self.assertEqual(len(reconciled.objects[0].instances), 3)

    def test_compound_and_head_noun_label_matching(self):
        """
        Verify that compound category labels match by head noun or token subset safely.
        """
        self.assertTrue(labels_match("snake plant", "plant"))
        self.assertTrue(labels_match("plant", "snake plant"))
        self.assertTrue(labels_match("white towel", "towel"))
        self.assertTrue(labels_match("tea light candle", "candle"))
        self.assertTrue(labels_match("elephant figurine", "figurine"))
        self.assertTrue(labels_match("gaming mouse", "mouse"))

        # Distinct objects must NOT match
        self.assertFalse(labels_match("plant", "candle"))
        self.assertFalse(labels_match("towel", "keyboard"))
        self.assertFalse(labels_match("mouse", "elephant"))

    def test_normalize_grounded_analysis_synthesizes_missing_instances(self):
        """
        Verify that if confirmed_count > len(instances), normalize_grounded_analysis
        synthesizes missing instances so they can be localized.
        """
        raw = {
            "objects": [
                {
                    "name": "candle",
                    "confirmed_count": 3,
                    "instances": [
                        {"id": "candle_1", "box_2d": [100, 100, 200, 200]}
                        # Missing candle_2 and candle_3
                    ]
                }
            ],
            "scene": {"summary": "A room"},
            "overall_summary": "Summary"
        }
        normalized = normalize_grounded_analysis(raw)
        instances = normalized["objects"][0]["instances"]
        self.assertEqual(len(instances), 3)
        self.assertEqual(instances[0]["id"], "candle_1")
        self.assertEqual(instances[1]["id"], "candle_2")
        self.assertEqual(instances[2]["id"], "candle_3")

    def test_object_on_supporting_surface_preserved(self):
        """
        Verify that an independent object (e.g. smartwatch) sitting on a supporting surface
        (e.g. white towel) is NOT suppressed by containment or collision exclusion.
        """
        # Large white towel
        inst_towel = ObjectInstance(
            id="towel_1",
            attributes=InstanceAttributes(),
            bounding_box=BoundingBox(x_min=100.0, y_min=100.0, x_max=500.0, y_max=900.0)
        )
        # Smartwatch resting completely on top of the towel
        inst_watch = ObjectInstance(
            id="smartwatch_1",
            attributes=InstanceAttributes(),
            bounding_box=BoundingBox(x_min=150.0, y_min=400.0, x_max=280.0, y_max=650.0)
        )
        cat_towel = DetectedObjectCategory(name="towel", confirmed_count=1, instances=[inst_towel])
        cat_watch = DetectedObjectCategory(name="smartwatch", confirmed_count=1, instances=[inst_watch])
        res = GroundedAnalysisResult(objects=[cat_towel, cat_watch], scene=SceneDescription(), overall_summary="Watch on towel.")

        sanitized = sanitize_bounding_boxes(res)
        # BOTH towel and smartwatch bounding boxes MUST be preserved!
        self.assertIsNotNone(sanitized.objects[0].instances[0].bounding_box)
        self.assertIsNotNone(sanitized.objects[1].instances[0].bounding_box)
        self.assertEqual(sanitized.objects[1].instances[0].bounding_box.x_min, 150.0)


if __name__ == "__main__":
    unittest.main()
