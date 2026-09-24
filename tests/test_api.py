import os
import io
import sys
import base64
import unittest
# pyrefly: ignore [missing-import]
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.server import app

client = TestClient(app)

class TestFastAPIBackend(unittest.TestCase):

    def setUp(self):
        # Create a small RGB test image in BytesIO
        self.img = Image.new("RGB", (100, 100), color="blue")
        buf = io.BytesIO()
        self.img.save(buf, format="JPEG")
        self.img_bytes = buf.getvalue()
        
        # Base64 representation
        b64 = base64.b64encode(self.img_bytes).decode("utf-8")
        self.img_b64 = f"data:image/jpeg;base64,{b64}"

    def test_health_check(self):
        response = client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("api_key_configured", data)

    def test_export_endpoint(self):
        response = client.post(
            "/api/export",
            data={"image_base64": self.img_b64, "format_type": "JPEG"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/jpeg")
        self.assertTrue(len(response.content) > 0)

    def test_invalid_image_upload(self):
        response = client.post(
            "/api/safety-check",
            files={"file": ("test.txt", b"not an image", "text/plain")}
        )
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("detail", data)

if __name__ == "__main__":
    unittest.main()
