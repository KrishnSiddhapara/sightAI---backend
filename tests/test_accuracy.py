import io
import unittest
from unittest.mock import MagicMock, patch
from PIL import Image

from utils.image_validation import validate_image_file, compute_image_hash
from services.schemas import (
    GroundedAnalysisResult,
    DetectedObjectCategory,
    ObjectInstance,
    InstanceAttributes,
    SceneDescription,
)
from services.vision import analyze_image_grounded


class MockUploadedFile:
    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


class TestVisualAccuracyAndGrounding(unittest.TestCase):

    def setUp(self):
        self.img1 = Image.new("RGB", (150, 150), color="red")
        buf1 = io.BytesIO()
        self.img1.save(buf1, format="PNG")
        self.bytes1 = buf1.getvalue()
        self.upload1 = MockUploadedFile("sample1.png", self.bytes1)

        self.img2 = Image.new("RGB", (150, 150), color="blue")
        buf2 = io.BytesIO()
        self.img2.save(buf2, format="PNG")
        self.bytes2 = buf2.getvalue()
        self.upload2 = MockUploadedFile("sample2.png", self.bytes2)

    def test_pydantic_grounded_schema_instance_isolation(self):
        """Verify that person instances maintain completely distinct attributes without cross-contamination."""
        attr1 = InstanceAttributes(
            clothing="red t-shirt and blue jeans",
            clothing_color="red",
            pose="standing",
            action="looking left",
        )
        attr2 = InstanceAttributes(
            clothing="green jacket and black shorts",
            clothing_color="green",
            pose="sitting",
            action="reading book",
        )

        inst1 = ObjectInstance(id="person_1", attributes=attr1)
        inst2 = ObjectInstance(id="person_2", attributes=attr2)

        cat = DetectedObjectCategory(
            name="person",
            confirmed_count=2,
            instances=[inst1, inst2],
        )

        scene = SceneDescription(
            environment="Park bench",
            primary_activity="Two people relaxing",
            summary="Park scene with two people.",
        )

        result = GroundedAnalysisResult(
            objects=[cat],
            scene=scene,
            overall_summary="Two people sitting and standing in a park.",
        )

        # Assert independent attribute isolation
        self.assertEqual(result.objects[0].confirmed_count, 2)
        self.assertEqual(result.objects[0].instances[0].attributes.clothing_color, "red")
        self.assertEqual(result.objects[0].instances[1].attributes.clothing_color, "green")
        self.assertNotEqual(
            result.objects[0].instances[0].attributes.clothing_color,
            result.objects[0].instances[1].attributes.clothing_color,
        )

    @patch("services.vision.genai.Client")
    def test_analyze_image_grounded_mock(self, mock_client_cls):
        """Verify analyze_image_grounded correctly parses structured VLM responses."""
        mock_client = MagicMock()
        mock_res = MagicMock()
        json_payload = """{
            "objects": [
                {
                    "name": "person",
                    "confirmed_count": 3,
                    "uncertain_count": 0,
                    "instances": [
                        {
                            "id": "person_1",
                            "attributes": {
                                "clothing": "red t-shirt",
                                "clothing_color": "red",
                                "pose": "standing",
                                "action": "waving",
                                "accessories": "none visible"
                            },
                            "bounding_box": {
                                "x_min": 100,
                                "y_min": 150,
                                "x_max": 300,
                                "y_max": 800
                            }
                        },
                        {
                            "id": "person_2",
                            "attributes": {
                                "clothing": "blue shirt",
                                "clothing_color": "blue",
                                "pose": "sitting",
                                "action": "talking",
                                "accessories": "glasses"
                            },
                            "bounding_box": {
                                "x_min": 350,
                                "y_min": 200,
                                "x_max": 550,
                                "y_max": 750
                            }
                        },
                        {
                            "id": "person_3",
                            "attributes": {
                                "clothing": "white hoodie",
                                "clothing_color": "white",
                                "pose": "standing",
                                "action": "smiling",
                                "accessories": "hat"
                            },
                            "bounding_box": null
                        }
                    ]
                },
                {
                    "name": "car",
                    "confirmed_count": 1,
                    "uncertain_count": 0,
                    "instances": [
                        {
                            "id": "car_1",
                            "attributes": {
                                "object_color": "black",
                                "type_or_subtype": "sedan",
                                "visible_details": "parked on left",
                                "pose": "parked"
                            },
                            "bounding_box": {
                                "x_min": 600,
                                "y_min": 400,
                                "x_max": 950,
                                "y_max": 850
                            }
                        }
                    ]
                }
            ],
            "scene": {
                "category": "Street",
                "environment": "Paved street near sidewalk",
                "primary_activity": "People gathered near parked car",
                "summary": "Outdoor street scene with 3 people and 1 parked car."
            },
            "overall_summary": "Three people wearing distinct clothing gathered on a street next to a black car."
        }"""
        mock_res.text = json_payload
        mock_client.models.generate_content.return_value = mock_res
        mock_client_cls.return_value = mock_client

        res = analyze_image_grounded(self.img1, "fake_key")
        self.assertEqual(len(res.objects), 2)
        self.assertEqual(res.objects[0].name, "person")
        self.assertEqual(res.objects[0].confirmed_count, 3)
        self.assertEqual(len(res.objects[0].instances), 3)
        self.assertEqual(res.objects[0].instances[0].attributes.clothing_color, "red")
        self.assertIsNotNone(res.objects[0].instances[0].bounding_box)
        self.assertEqual(res.objects[0].instances[0].bounding_box.x_min, 100)
        self.assertIsNone(res.objects[0].instances[2].bounding_box)

    def test_bounding_box_validation_and_sanitization(self):
        """Verify invalid or inverted bounding boxes are safely repaired or discarded to prevent rendering bugs."""
        from services.vision import sanitize_bounding_boxes
        from services.schemas import BoundingBox

        inst_valid = ObjectInstance(
            id="item_1",
            attributes=InstanceAttributes(),
            bounding_box=BoundingBox(x_min=50, y_min=50, x_max=200, y_max=300)
        )
        inst_inverted = ObjectInstance(
            id="item_2",
            attributes=InstanceAttributes(),
            bounding_box=BoundingBox(x_min=200, y_min=50, x_max=50, y_max=300)
        )
        inst_zero_area = ObjectInstance(
            id="item_3",
            attributes=InstanceAttributes(),
            bounding_box=BoundingBox(x_min=100, y_min=100, x_max=102, y_max=102)
        )

        cat = DetectedObjectCategory(name="box", confirmed_count=3, instances=[inst_valid, inst_inverted, inst_zero_area])
        scene = SceneDescription(environment="Room", primary_activity="Testing", summary="Test")
        res = GroundedAnalysisResult(objects=[cat], scene=scene, overall_summary="Test")

        sanitized = sanitize_bounding_boxes(res)
        self.assertIsNotNone(sanitized.objects[0].instances[0].bounding_box)
        # Verify auto-repair of inverted x_min and x_max
        self.assertEqual(sanitized.objects[0].instances[1].bounding_box.x_min, 50)
        self.assertEqual(sanitized.objects[0].instances[1].bounding_box.x_max, 200)
        # Verify zero-area box (< 5 units) is discarded
        self.assertIsNone(sanitized.objects[0].instances[2].bounding_box)

    def test_image_preprocessing_rgb_and_hash(self):
        """Verify image validation handles RGB mode and computes distinct hashes for state invalidation."""
        is_valid, err, pil_img = validate_image_file(self.upload1)
        self.assertTrue(is_valid)
        self.assertEqual(pil_img.mode, "RGB")

        hash1 = compute_image_hash(self.bytes1)
        hash2 = compute_image_hash(self.bytes2)
        self.assertNotEqual(hash1, hash2)


if __name__ == "__main__":
    unittest.main()
