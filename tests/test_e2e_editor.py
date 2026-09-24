import os
import io
import sys
import unittest
import dotenv

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

dotenv.load_dotenv()
api_key = os.getenv("VLM_API_KEY")

from PIL import Image
from utils.image_validation import validate_image_file, compute_image_hash
from services.safety import check_image_safety
from services.vision import analyze_image_grounded
from services.user_query import answer_image_query
from services.image_editor import (
    edit_image,
    check_edit_instruction_ambiguity,
    validate_edit_instruction,
)
from services.version_manager import (
    create_initial_version,
    add_new_version,
    get_version_by_number,
    get_latest_version,
    export_image_bytes,
)

class TestE2EMultiVersionEditor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not api_key or api_key == "your_api_key_here":
            raise unittest.SkipTest("Skipping E2E tests: VLM_API_KEY not configured in .env")

    def setUp(self):
        # Create a simple safe test image
        self.img = Image.new("RGB", (300, 300), color="lightblue")

    def test_01_safety_check_input(self):
        """TEST 1: Verify safety check works on safe input image."""
        safety = check_image_safety(self.img, api_key)
        self.assertTrue(safety.get("is_safe"))

    def test_02_multi_version_sequential_edits(self):
        """TEST 2: Sequential edits v0 -> v1 -> v2 -> v3 create immutable versions."""
        history = [create_initial_version(self.img)]
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["version_number"], 0)

        # Edit 1: v0 -> v1
        v1_img = edit_image(history[0]["image"], "Add a football next to the person", api_key)
        v1 = add_new_version(history, v1_img, "Add a football next to the person", source_version_num=0)
        self.assertEqual(v1["version_number"], 1)
        self.assertEqual(v1["source_version_number"], 0)

        # Edit 2: v1 -> v2
        v2_img = edit_image(v1["image"], "Change football color to red", api_key)
        v2 = add_new_version(history, v2_img, "Change football color to red", source_version_num=1)
        self.assertEqual(v2["version_number"], 2)
        self.assertEqual(v2["source_version_number"], 1)

        # Edit 3: v2 -> v3
        v3_img = edit_image(v2["image"], "Replace background with a beach", api_key)
        v3 = add_new_version(history, v3_img, "Replace background with a beach", source_version_num=2)
        self.assertEqual(v3["version_number"], 3)
        self.assertEqual(v3["source_version_number"], 2)

        self.assertEqual(len(history), 4) # v0, v1, v2, v3
        self.assertEqual(get_latest_version(history)["version_number"], 3)

    def test_03_download_previous_version_isolation(self):
        """TEST 3: Downloading Version 1 returns Version 1 bytes, not Version 3."""
        history = [create_initial_version(self.img)]
        v1_img = edit_image(history[0]["image"], "Add a football", api_key)
        add_new_version(history, v1_img, "Add a football", source_version_num=0) # v1

        v2_img = edit_image(history[1]["image"], "Make background beach", api_key)
        add_new_version(history, v2_img, "Make background beach", source_version_num=1) # v2

        v1_bytes = export_image_bytes(history[1]["image"], "JPEG")
        v2_bytes = export_image_bytes(history[2]["image"], "JPEG")

        self.assertNotEqual(v1_bytes, v2_bytes)
        self.assertEqual(history[1]["version_number"], 1)

    def test_04_edit_previous_version_branching(self):
        """TEST 4: Editing Version 1 when Version 2 exists creates Version 3 based on Version 1."""
        history = [create_initial_version(self.img)]
        v1_img = edit_image(history[0]["image"], "Add a football", api_key)
        add_new_version(history, v1_img, "Add a football", source_version_num=0) # v1

        v2_img = edit_image(history[1]["image"], "Make football red", api_key)
        add_new_version(history, v2_img, "Make football red", source_version_num=1) # v2

        # Edit v1 to create v3
        v3_img = edit_image(history[1]["image"], "Add sunglasses to person", api_key)
        v3 = add_new_version(history, v3_img, "Add sunglasses to person", source_version_num=1)

        self.assertEqual(v3["version_number"], 3)
        self.assertEqual(v3["source_version_number"], 1)
        self.assertEqual(len(history), 4)

        # Verify v1 and v2 prompts remain intact
        self.assertEqual(history[1]["edit_prompt"], "Add a football")
        self.assertEqual(history[2]["edit_prompt"], "Make football red")

    def test_05_failed_edit_history_preservation(self):
        """TEST 5: Failed edit does not mutate or corrupt existing version history."""
        history = [create_initial_version(self.img)]
        v1_img = edit_image(history[0]["image"], "Add a football", api_key)
        add_new_version(history, v1_img, "Add a football", source_version_num=0) # v1

        # Attempt invalid empty edit prompt
        with self.assertRaises(ValueError):
            edit_image(history[1]["image"], "", api_key)

        # History remains intact with 2 items (v0, v1)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[1]["version_number"], 1)

    def test_06_ambiguity_handling(self):
        """TEST 6: Ambiguous instruction with multiple people triggers ambiguity flag."""
        from services.schemas import GroundedAnalysisResult, DetectedObjectCategory, ObjectInstance, InstanceAttributes, SceneDescription
        cat = DetectedObjectCategory(
            name="person",
            confirmed_count=3,
            instances=[
                ObjectInstance(id="person_1", attributes=InstanceAttributes()),
                ObjectInstance(id="person_2", attributes=InstanceAttributes()),
                ObjectInstance(id="person_3", attributes=InstanceAttributes()),
            ]
        )
        ctx = GroundedAnalysisResult(
            objects=[cat],
            scene=SceneDescription(category="Outdoor", environment="Park", primary_activity="Walk", summary="3 people"),
            overall_summary="3 people in park"
        )

        is_ambig, msg = check_edit_instruction_ambiguity("Change the shirt to red", ctx)
        self.assertTrue(is_ambig)
        self.assertIn("Ambiguous target", msg)

        is_ambig_specific, _ = check_edit_instruction_ambiguity("Change person 2's shirt to black", ctx)
        self.assertFalse(is_ambig_specific)

if __name__ == "__main__":
    unittest.main()
