import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from main import ResearchResponse, _format_for_save, _is_anthropic_credit_error  # noqa: E402

class TestMain(unittest.TestCase):
    def test_format_for_save_includes_fields(self):
        response = ResearchResponse(
            topic="Topic A",
            summary="Some summary.",
            sources=["https://example.com/1", "https://example.com/2"],
            tools_used=["wikipedia", "search"],
        )

        text = _format_for_save(response, original_query="my query", mode="fallback")
        self.assertIn("MODE: fallback", text)
        self.assertIn("ORIGINAL_QUERY: my query", text)
        self.assertIn("TOPIC: Topic A", text)
        self.assertIn("SUMMARY:\nSome summary.", text)
        self.assertIn("- https://example.com/1", text)
        self.assertIn("- https://example.com/2", text)
        self.assertIn("TOOLS_USED: wikipedia, search", text)

    def test_is_anthropic_credit_error_detects_message(self):
        err = Exception(
            "Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', "
            "'message': 'Your credit balance is too low to access the Anthropic API.'}}"
        )
        self.assertTrue(_is_anthropic_credit_error(err))
        self.assertFalse(_is_anthropic_credit_error(Exception("Some other error")))


if __name__ == "__main__":
    unittest.main()
