import unittest
from pydantic import ValidationError
from services.schemas import GroundedAnalysisResult
from utils.json_utils import clean_json_text, normalize_grounded_analysis, format_pydantic_validation_error

class TestSchemaNormalizationAndValidation(unittest.TestCase):

    def test_valid_response(self):
        payload = {
            "objects": [
                {
                    "name": "person",
                    "confirmed_count": 1,
                    "uncertain_count": 0,
                    "instances": [
                        {
                            "id": "person_1",
                            "attributes": {"clothing": "blue shirt"},
                            "bounding_box": {"x_min": 100, "y_min": 200, "x_max": 500, "y_max": 800}
                        }
                    ]
                }
            ],
            "scene": {
                "environment": "Desk area",
                "primary_activity": "Working",
                "summary": "Office room"
            },
            "overall_summary": "Single person working at desk."
        }
        norm = normalize_grounded_analysis(payload)
        res = GroundedAnalysisResult.model_validate(norm)
        self.assertEqual(len(res.objects), 1)
        self.assertEqual(res.objects[0].name, "person")
        self.assertEqual(res.scene.environment, "Desk area")

    def test_missing_optional_fields_and_defaults(self):
        payload = {
            "objects": [{"name": "chair"}],
            "scene": {"environment": "Room"},
            "overall_summary": "A room with a chair."
        }
        norm = normalize_grounded_analysis(payload)
        res = GroundedAnalysisResult.model_validate(norm)
        self.assertEqual(res.objects[0].confirmed_count, 1)
        self.assertEqual(res.objects[0].uncertain_count, 0)
        self.assertEqual(res.scene.environment, "Room")

    def test_empty_objects_image(self):
        payload = {
            "objects": [],
            "scene": {"environment": "Forest", "primary_activity": "None", "summary": "Forest"},
            "overall_summary": "Empty forest."
        }
        norm = normalize_grounded_analysis(payload)
        res = GroundedAnalysisResult.model_validate(norm)
        self.assertEqual(len(res.objects), 0)

    def test_camelcase_normalization(self):
        payload = {
            "objects": [
                {
                    "name": "dog",
                    "confirmedCount": 2,
                    "uncertainCount": 1,
                    "instances": [
                        {
                            "id": "dog_1",
                            "attributes": {"objectColor": "brown", "typeOrSubtype": "golden retriever"},
                            "boundingBox": {"xMin": 100, "yMin": 150, "xMax": 400, "yMax": 500}
                        }
                    ]
                }
            ],
            "scene": {"sceneEnvironment": "Grass field", "primaryActivity": "Dogs playing", "sceneSummary": "Park"},
            "overallSummary": "Dogs playing in park."
        }
        norm = normalize_grounded_analysis(payload)
        res = GroundedAnalysisResult.model_validate(norm)
        self.assertEqual(res.objects[0].confirmed_count, 2)
        self.assertEqual(res.objects[0].instances[0].attributes.object_color, "brown")
        self.assertEqual(res.objects[0].instances[0].bounding_box.x_min, 100)

    def test_google_2d_box_array_normalization(self):
        payload = {
            "objects": [
                {
                    "name": "car",
                    "instances": [
                        {
                            "id": "car_1",
                            "bounding_box": [150, 50, 450, 600]  # [ymin, xmin, ymax, xmax]
                        }
                    ]
                }
            ],
            "scene": {"environment": "Road"},
            "overall_summary": "Car on road."
        }
        norm = normalize_grounded_analysis(payload)
        res = GroundedAnalysisResult.model_validate(norm)
        self.assertEqual(res.objects[0].instances[0].bounding_box.x_min, 50)
        self.assertEqual(res.objects[0].instances[0].bounding_box.y_min, 150)
        self.assertEqual(res.objects[0].instances[0].bounding_box.x_max, 600)
        self.assertEqual(res.objects[0].instances[0].bounding_box.y_max, 450)

    def test_float_coordinate_scale_0_to_1_normalization(self):
        payload = {
            "objects": [
                {
                    "name": "bottle",
                    "instances": [
                        {
                            "id": "bottle_1",
                            "bounding_box": {"x_min": 0.1, "y_min": 0.2, "x_max": 0.4, "y_max": 0.8}
                        }
                    ]
                }
            ],
            "scene": {"environment": "Table"},
            "overall_summary": "Bottle on table."
        }
        norm = normalize_grounded_analysis(payload)
        res = GroundedAnalysisResult.model_validate(norm)
        self.assertEqual(res.objects[0].instances[0].bounding_box.x_min, 100)
        self.assertEqual(res.objects[0].instances[0].bounding_box.y_min, 200)
        self.assertEqual(res.objects[0].instances[0].bounding_box.x_max, 400)
        self.assertEqual(res.objects[0].instances[0].bounding_box.y_max, 800)

    def test_markdown_json_cleaning(self):
        raw_md = "```json\n{\"overall_summary\": \"Test summary\"}\n```"
        clean = clean_json_text(raw_md)
        self.assertEqual(clean, "{\"overall_summary\": \"Test summary\"}")

    def test_validation_error_formatter(self):
        try:
            # Force validation failure by passing invalid int
            GroundedAnalysisResult.model_validate({"objects": "not a list"})
        except ValidationError as exc:
            formatted = format_pydantic_validation_error(exc)
            self.assertIn("Field: 'objects'", formatted)

if __name__ == "__main__":
    unittest.main()
