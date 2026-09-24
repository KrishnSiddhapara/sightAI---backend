import io
import socket
import unittest
from unittest.mock import MagicMock, patch
from PIL import Image
from dotenv import load_dotenv

from services.user_query import validate_user_prompt, answer_image_query

load_dotenv()

class TestAskAIFeature(unittest.TestCase):

    def setUp(self):
        self.img = Image.new("RGB", (100, 100), color="blue")

    def test_validate_user_prompt(self):
        self.assertIsNotNone(validate_user_prompt(""))
        self.assertIsNotNone(validate_user_prompt("   "))
        self.assertIsNotNone(validate_user_prompt("a" * 501))
        self.assertIsNone(validate_user_prompt("What color is the sky?"))

    @patch("services.user_query.genai.Client")
    def test_answer_image_query(self, mock_client_cls):
        mock_client = MagicMock()
        mock_res = MagicMock()
        mock_res.text = "The sky is blue in the image."
        mock_client.models.generate_content.return_value = mock_res
        mock_client_cls.return_value = mock_client

        ans = answer_image_query(self.img, "What color is the sky?", "fake_key")
        self.assertEqual(ans, "The sky is blue in the image.")

    @patch("services.user_query.time.sleep")
    @patch("services.user_query.genai.Client")
    def test_network_dns_error_handling(self, mock_client_cls, mock_sleep):
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = socket.gaierror(11001, "getaddrinfo failed")
        mock_client_cls.return_value = mock_client

        with self.assertRaises(ConnectionError) as ctx:
            answer_image_query(self.img, "Where is the smallest tree?", "fake_key", max_retries=2)

        self.assertIn("Network connection failed", str(ctx.exception))
        self.assertIn("DNS lookup failed", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
