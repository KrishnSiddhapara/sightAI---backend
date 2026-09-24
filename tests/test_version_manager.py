import os
import sys
import unittest
from PIL import Image

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.version_manager import (
    create_initial_version,
    add_new_version,
    get_version_by_number,
    get_latest_version,
    export_image_bytes,
)

class TestVersionManager(unittest.TestCase):

    def setUp(self):
        self.orig_img = Image.new("RGB", (100, 100), color="blue")
        self.edit1_img = Image.new("RGB", (100, 100), color="red")
        self.edit2_img = Image.new("RGB", (100, 100), color="green")
        self.edit3_img = Image.new("RGB", (100, 100), color="yellow")
        self.edit4_img = Image.new("RGB", (100, 100), color="purple")

    def test_initial_version_creation(self):
        """Test initial Version 0 creation."""
        v0 = create_initial_version(self.orig_img)
        self.assertEqual(v0["version_number"], 0)
        self.assertEqual(v0["version_id"], "v0")
        self.assertEqual(v0["edit_prompt"], "Original Image")
        self.assertEqual(v0["source_version_number"], 0)
        self.assertIsInstance(v0["image"], Image.Image)

    def test_sequential_version_addition(self):
        """Test sequential additions v0 -> v1 -> v2 -> v3."""
        history = [create_initial_version(self.orig_img)]
        
        v1 = add_new_version(history, self.edit1_img, "Add football", source_version_num=0)
        self.assertEqual(v1["version_number"], 1)
        self.assertEqual(v1["source_version_number"], 0)

        v2 = add_new_version(history, self.edit2_img, "Change football to red", source_version_num=1)
        self.assertEqual(v2["version_number"], 2)
        self.assertEqual(v2["source_version_number"], 1)

        v3 = add_new_version(history, self.edit3_img, "Change background to stadium", source_version_num=2)
        self.assertEqual(v3["version_number"], 3)
        self.assertEqual(v3["source_version_number"], 2)

        self.assertEqual(len(history), 4) # v0, v1, v2, v3
        latest = get_latest_version(history)
        self.assertEqual(latest["version_number"], 3)

    def test_branching_version_lineage(self):
        """Test editing v1 when v3 exists creates v4 based on v1 without mutating v1, v2, v3."""
        history = [create_initial_version(self.orig_img)]
        add_new_version(history, self.edit1_img, "Add football", source_version_num=0) # v1
        add_new_version(history, self.edit2_img, "Make red", source_version_num=1) # v2
        add_new_version(history, self.edit3_img, "Add stadium", source_version_num=2) # v3

        # Edit v1 to create v4
        v4 = add_new_version(history, self.edit4_img, "Add hat to person", source_version_num=1)
        self.assertEqual(v4["version_number"], 4)
        self.assertEqual(v4["source_version_number"], 1)
        self.assertEqual(len(history), 5)

        # Verify v1, v2, v3 remain unchanged
        v1_rec = get_version_by_number(history, 1)
        self.assertIsNotNone(v1_rec)
        self.assertEqual(v1_rec["edit_prompt"], "Add football")
        v3_rec = get_version_by_number(history, 3)
        self.assertEqual(v3_rec["edit_prompt"], "Add stadium")

    def test_export_image_bytes(self):
        """Test conversion of PIL image to JPEG and PDF bytes."""
        jpg_bytes = export_image_bytes(self.orig_img, format_type="JPEG")
        self.assertGreater(len(jpg_bytes), 0)

        pdf_bytes = export_image_bytes(self.orig_img, format_type="PDF")
        self.assertGreater(len(pdf_bytes), 0)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))

if __name__ == "__main__":
    unittest.main()
