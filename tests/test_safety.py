import io
import os
import unittest
from unittest.mock import MagicMock, patch
from PIL import Image
from dotenv import load_dotenv

from utils.image_validation import validate_image_file, compute_image_hash
from services.safety import check_image_safety, get_fail_closed_response, SafetyCategory
from services.vision import parse_vlm_response

load_dotenv()

class MockUploadedFile:
    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


class TestImageSafetyAndValidation(unittest.TestCase):

    def setUp(self):
        # Create a valid test image in memory
        self.img = Image.new("RGB", (100, 100), color="blue")
        buf = io.BytesIO()
        self.img.save(buf, format="PNG")
        self.valid_png_bytes = buf.getvalue()
        self.valid_upload = MockUploadedFile("test_sample.png", self.valid_png_bytes)

    def test_compute_image_hash(self):
        hash1 = compute_image_hash(self.valid_png_bytes)
        hash2 = compute_image_hash(self.valid_png_bytes)
        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 64)

        # Different bytes -> different hash
        hash3 = compute_image_hash(b"different_bytes")
        self.assertNotEqual(hash1, hash3)

    def test_validate_image_file_valid(self):
        is_valid, err, usable_img = validate_image_file(self.valid_upload, max_size_mb=10.0)
        self.assertTrue(is_valid)
        self.assertIsNone(err)
        self.assertIsNotNone(usable_img)

    def test_validate_image_file_invalid_ext(self):
        invalid_upload = MockUploadedFile("test.exe", self.valid_png_bytes)
        is_valid, err, usable_img = validate_image_file(invalid_upload, max_size_mb=10.0)
        self.assertFalse(is_valid)
        self.assertIn("Unsupported file format", err)
        self.assertIsNone(usable_img)

    def test_validate_image_file_oversized(self):
        is_valid, err, usable_img = validate_image_file(self.valid_upload, max_size_mb=0.00001)
        self.assertFalse(is_valid)
        self.assertIn("File size exceeds limit", err)
        self.assertIsNone(usable_img)

    def test_validate_image_file_corrupted(self):
        corrupted_upload = MockUploadedFile("corrupted.jpg", b"NOT_AN_IMAGE_HEADER_DATA")
        is_valid, err, usable_img = validate_image_file(corrupted_upload, max_size_mb=10.0)
        self.assertFalse(is_valid)
        self.assertIn("corrupted", err)
        self.assertIsNone(usable_img)

    def test_fail_closed_response(self):
        result = get_fail_closed_response("Test error scenario")
        self.assertFalse(result["is_safe"])
        self.assertEqual(result["category"], "UNKNOWN")
        self.assertEqual(result["confidence"], 0.0)
        self.assertIn("Safety check unavailable", result["error"])

    def test_check_image_safety_missing_key(self):
        result = check_image_safety(self.img, api_key="")
        self.assertFalse(result["is_safe"])
        self.assertEqual(result["category"], "UNKNOWN")

    @patch("services.safety.genai.Client")
    def test_check_image_safety_api_exception(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("API connection timed out")
        mock_client_cls.return_value = mock_client

        result = check_image_safety(self.img, api_key="fake_key")
        self.assertFalse(result["is_safe"])
        self.assertEqual(result["category"], "UNKNOWN")
        self.assertIn("Safety check unavailable", result["error"])

    @patch("services.safety.genai.Client")
    def test_check_image_safety_violence_response(self, mock_client_cls):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"is_safe": false, "category": "VIOLENCE_AND_CIVIL_UNREST", "confidence": 0.99, "reasoning": "Riot and street violence with smoke and sticks detected"}'
        mock_client.models.generate_content.return_value = mock_response
        mock_client_cls.return_value = mock_client

        result = check_image_safety(self.img, api_key="fake_key")
        self.assertFalse(result["is_safe"])
        self.assertEqual(result["category"], "VIOLENCE_AND_CIVIL_UNREST")
        self.assertEqual(result["confidence"], 0.99)
        self.assertIsNotNone(result["error"])

    @patch("services.safety.genai.Client")
    def test_check_image_safety_safe_response(self, mock_client_cls):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"is_safe": true, "category": "SAFE", "confidence": 0.98, "reasoning": "Benign solid color image"}'
        mock_client.models.generate_content.return_value = mock_response
        mock_client_cls.return_value = mock_client

        result = check_image_safety(self.img, api_key="fake_key")
        self.assertTrue(result["is_safe"])
        self.assertEqual(result["category"], "SAFE")
        self.assertEqual(result["confidence"], 0.98)
        self.assertIsNone(result["error"])

    def test_parse_vlm_response(self):
        sample_text = """Objects:
1. Laptop
2. Coffee cup
3. Notebook

Description:
A clean modern desk workspace with a laptop and coffee cup."""
        objects, desc = parse_vlm_response(sample_text)
        self.assertEqual(objects, ["Laptop", "Coffee cup", "Notebook"])
        self.assertEqual(desc, "A clean modern desk workspace with a laptop and coffee cup.")


if __name__ == "__main__":
    unittest.main()
