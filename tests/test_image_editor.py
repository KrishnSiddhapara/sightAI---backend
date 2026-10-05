import io
import unittest
from unittest.mock import MagicMock, patch
from PIL import Image

from services.schemas import (
    GroundedAnalysisResult,
    DetectedObjectCategory,
    ObjectInstance,
    InstanceAttributes,
    SceneDescription,
)
from services.image_editor import (
    validate_edit_instruction,
    check_edit_instruction_ambiguity,
    build_editing_prompt,
    edit_image,
)

class TestImageEditorService(unittest.TestCase):

    def setUp(self):
        self.sample_img = Image.new("RGB", (200, 200), color="blue")
        buf = io.BytesIO()
        self.sample_img.save(buf, format="PNG")
        self.sample_img_bytes = buf.getvalue()

    def test_validate_edit_instruction(self):
        """Test instruction validation logic."""
        self.assertIsNotNone(validate_edit_instruction(""))
        self.assertIsNotNone(validate_edit_instruction("   "))
        self.assertIsNotNone(validate_edit_instruction("a" * 350))
        self.assertIsNone(validate_edit_instruction("Add a football next to the person."))

    def test_check_edit_instruction_ambiguity_no_context(self):
        """Without context, ambiguity check returns False."""
        is_ambig, msg = check_edit_instruction_ambiguity("Change shirt to red", None)
        self.assertFalse(is_ambig)
        self.assertIsNone(msg)

    def test_check_edit_instruction_ambiguity_multiple_people(self):
        """With multiple people and non-specific shirt edit, returns True with warning."""
        inst1 = ObjectInstance(id="person_1", attributes=InstanceAttributes(clothing_color="red"))
        inst2 = ObjectInstance(id="person_2", attributes=InstanceAttributes(clothing_color="green"))
        cat = DetectedObjectCategory(name="person", confirmed_count=2, instances=[inst1, inst2])
        scene = SceneDescription(environment="Park", primary_activity="Walking", summary="Two people.")
        vision_ctx = GroundedAnalysisResult(objects=[cat], scene=scene, overall_summary="Two people in park.")

        # Ambiguous instruction
        is_ambig, msg = check_edit_instruction_ambiguity("Change the shirt to blue", vision_ctx)
        self.assertTrue(is_ambig)
        self.assertIn("Ambiguous target", msg)

        # Unambiguous instruction specifying person 1
        is_ambig_specific, msg_specific = check_edit_instruction_ambiguity("Change person 1's shirt to blue", vision_ctx)
        self.assertFalse(is_ambig_specific)
        self.assertIsNone(msg_specific)

    def test_build_editing_prompt(self):
        """Test editing prompt assembly."""
        prompt = build_editing_prompt("Add sunglasses to person")
        self.assertIn("USER EDIT REQUEST: Add sunglasses to person", prompt)
        self.assertIn("CONCEPTUAL EDIT INSTRUCTIONS:", prompt)

    @patch("services.image_editor.genai.Client")
    def test_edit_image_mock_success(self, mock_client_cls):
        """Test edit_image successfully decodes returned inline_data PIL image."""
        mock_client = MagicMock()
        mock_res = MagicMock()
        
        # Create fake inline_data part returning valid PNG bytes
        out_img = Image.new("RGB", (100, 100), color="green")
        out_buf = io.BytesIO()
        out_img.save(out_buf, format="PNG")
        
        mock_part = MagicMock()
        mock_part.inline_data.data = out_buf.getvalue()
        
        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]
        mock_res.candidates = [mock_candidate]
        
        mock_client.models.generate_content.return_value = mock_res
        mock_client_cls.return_value = mock_client

        edited = edit_image(self.sample_img, "Add a football", "fake_key")
        self.assertIsInstance(edited, Image.Image)
        self.assertEqual(edited.size, (100, 100))

    def test_edit_image_missing_key(self):
        """Test edit_image raises ValueError if API key is missing."""
        with self.assertRaises(ValueError):
            edit_image(self.sample_img, "Add a football", "")
        with self.assertRaises(ValueError):
            edit_image(self.sample_img, "Add a football", "your_api_key_here")

if __name__ == "__main__":
    unittest.main()
