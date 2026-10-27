import os
import unittest
from unittest.mock import patch

with patch.dict(os.environ, {"OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY") or "test-only"}):
    from .gtm_agent import run_agent, send_prospect_email


class SendProspectEmailTests(unittest.TestCase):
    def test_plain_text_and_rep_supplied_link(self):
        for body in ("Are you available for a demo?", "Pricing deck: https://example.com/pricing.pdf"):
            with self.subTest(body=body):
                result = send_prospect_email.func(
                    prospect={"name": "Prospect", "email": "prospect@example.com"},
                    subject="Follow up",
                    body=body,
                    runtime=None,
                    from_rep={"name": "Rep", "email": "rep@example.com"},
                )
                self.assertEqual(result["status"], "sent")
                self.assertEqual(result["body"], body)
                self.assertEqual(result["to"], "prospect@example.com")


@unittest.skipUnless(os.environ.get("OPENAI_API_KEY"), "Requires model credentials")
class EmailAgentTests(unittest.TestCase):
    def run_request(self, request):
        with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}):
            return run_agent(request, user_id="rep_sbrown")

    def sent_emails(self, result):
        import json

        return [json.loads(message.content) for message in result["messages"]
                if message.type == "tool" and message.name == "send_prospect_email"]

    def test_pricing_deck_without_link(self):
        result = self.run_request("Email LEAD-50003 to send over our pricing deck.")
        self.assertEqual(self.sent_emails(result), [])
        reply = result["reply"].lower()
        self.assertRegex(reply, r"cannot|can't|unable|don't have")
        self.assertIn("attach", reply)
        self.assertIn("link", reply)
        self.assertNotRegex(reply, r"(?:i've|i have|has been|successfully) attached")

    def test_pricing_deck_link(self):
        url = "https://example.com/pricing.pdf"
        result = self.run_request(f"Email LEAD-50003 our pricing deck using this link: {url}")
        emails = self.sent_emails(result)
        self.assertTrue(emails)
        for email in emails:
            self.assertEqual(email["status"], "sent")
            self.assertIn(url, email["body"])
            self.assertNotIn("attached", email["body"].lower())
        self.assertIn("link", result["reply"].lower())
        self.assertNotIn("attached", result["reply"].lower())

    def test_plain_text_email(self):
        result = self.run_request("Email LEAD-50003 to ask about availability for a demo.")
        emails = self.sent_emails(result)
        self.assertTrue(emails)
        self.assertTrue(all(email["status"] == "sent" for email in emails))


if __name__ == "__main__":
    unittest.main()
