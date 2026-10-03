import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from backend.server import app
from tools.web_search import WebSearchTool, classify_source_type, extract_domain
from services.schemas import AgentResearchRequest, AgentResearchResponse, SourceItem

client = TestClient(app)

class TestResearchAgent(unittest.TestCase):

    def test_domain_extraction_and_classification(self):
        self.assertEqual(extract_domain("https://www.amazon.com/dp/1234"), "amazon.com")
        self.assertEqual(extract_domain("https://en.wikipedia.org/wiki/Atomic_Habits"), "en.wikipedia.org")

        self.assertEqual(classify_source_type("https://www.amazon.com/dp/1234"), "retailer")
        self.assertEqual(classify_source_type("https://en.wikipedia.org/wiki/Test"), "reference")
        self.assertEqual(classify_source_type("https://jamesclear.com/atomic-habits", "James Clear Official Site"), "official")

    @patch("ddgs.DDGS")
    def test_web_search_tool_mock(self, mock_ddgs_cls):
        mock_instance = MagicMock()
        mock_instance.text.return_value = [
            {
                "title": "Atomic Habits on Amazon",
                "href": "https://www.amazon.com/Atomic-Habits/dp/0735211299",
                "body": "Buy Atomic Habits by James Clear."
            },
            {
                "title": "Atomic Habits Wikipedia",
                "href": "https://en.wikipedia.org/wiki/Atomic_Habits",
                "body": "Atomic Habits is a 2018 book."
            }
        ]
        mock_ddgs_cls.return_value = mock_instance

        tool = WebSearchTool()
        results = tool.execute("Atomic Habits James Clear", max_results=2)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["domain"], "amazon.com")
        self.assertEqual(results[0]["source_type"], "retailer")
        self.assertEqual(results[1]["source_type"], "reference")

    @patch("services.research_agent.genai.Client")
    def test_agent_research_flow_visual_only(self, mock_client_cls):
        """Verify that visual-only questions (e.g. shirt color) do NOT trigger web search."""
        mock_client = MagicMock()
        
        # Intent model response: requires_research = False
        mock_res_intent = MagicMock()
        mock_res_intent.text = '{"requires_research": false, "entity_type": "clothing", "identified_entities": ["shirt"], "search_query": "", "reasoning": "Visual question"}'

        # Answer model response
        mock_res_ans = MagicMock()
        mock_res_ans.text = "The person is wearing a red shirt."

        mock_client.models.generate_content.side_effect = [mock_res_intent, mock_res_ans]
        mock_client_cls.return_value = mock_client

        from services.research_agent import execute_agent_research
        req = AgentResearchRequest(question="What color is the shirt?", image_context={"objects": [{"name": "person"}]})
        res = execute_agent_research(req, "fake_key")

        self.assertTrue(res.success)
        self.assertFalse(res.requires_research)
        self.assertEqual(len(res.used_tools), 0)
        self.assertEqual(len(res.sources), 0)
        self.assertIn("red shirt", res.answer)

    @patch("services.research_agent.WebSearchTool.execute")
    @patch("services.research_agent.genai.Client")
    def test_agent_research_flow_external_search(self, mock_client_cls, mock_search):
        """Verify that purchasing questions trigger external search and return sources."""
        mock_client = MagicMock()
        
        # Intent response
        mock_res_intent = MagicMock()
        mock_res_intent.text = '{"requires_research": true, "entity_type": "book", "identified_entities": ["Atomic Habits"], "search_query": "Atomic Habits James Clear purchase", "reasoning": "Purchase info needed"}'

        # Synthesis response
        mock_res_synth = MagicMock()
        mock_res_synth.text = "You can purchase Atomic Habits from Amazon and local bookstores."

        mock_client.models.generate_content.side_effect = [mock_res_intent, mock_res_synth]
        mock_client_cls.return_value = mock_client

        mock_search.return_value = [
            {
                "title": "Atomic Habits - Amazon",
                "url": "https://www.amazon.com/Atomic-Habits/dp/0735211299",
                "domain": "amazon.com",
                "snippet": "Available for purchase.",
                "source_type": "retailer"
            }
        ]

        from services.research_agent import execute_agent_research
        req = AgentResearchRequest(question="Where can I purchase this book?", image_context={"overall_summary": "Atomic Habits book cover"})
        res = execute_agent_research(req, "fake_key")

        self.assertTrue(res.success)
        self.assertTrue(res.requires_research)
        self.assertIn("web_search", res.used_tools)
        self.assertEqual(len(res.sources), 1)
        self.assertEqual(res.sources[0].domain, "amazon.com")
        self.assertEqual(res.sources[0].source_type, "retailer")

if __name__ == "__main__":
    unittest.main()
