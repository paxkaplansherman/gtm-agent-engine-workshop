import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

with patch.dict(os.environ, {
    "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "test-key"),
    "LANGSMITH_TRACING": "false",
}):
    from gtm_agent import gtm_agent as agent


class SendProspectEmailTests(unittest.TestCase):
    def setUp(self):
        self.prospect = {
            "prospect_id": "LEAD-50001",
            "name": "Test Prospect",
            "email": "prospect@example.com",
            "disqualified": False,
        }
        self.rep = {"name": "Test Rep", "email": "rep@example.com"}
        self.runtime = SimpleNamespace(config={"metadata": {"user_id": "REP-TEST"}})

    def send(self, **kwargs):
        return agent.send_prospect_email.func(
            self.prospect, "Introduction", "Hello", self.runtime, **kwargs
        )

    def test_disqualified_crm_record_blocks_send_despite_supplied_flag(self):
        with patch.dict(agent.data_service.PROSPECTS, {
            "LEAD-50001": {**self.prospect, "disqualified": True},
        }):
            result = self.send(from_rep=self.rep)
        self.assertEqual(result, {
            "status": "blocked",
            "error": "Prospect LEAD-50001 is disqualified; email not sent.",
        })
        self.assertNotIn("message_id", result)

    def test_eligible_records_preserve_email_flow(self):
        for flag in (False, None, 1, "true"):
            for explicit_rep in (False, True):
                with self.subTest(flag=flag, explicit_rep=explicit_rep):
                    record = dict(self.prospect)
                    if flag is None:
                        record.pop("disqualified")
                    else:
                        record["disqualified"] = flag
                    with patch.dict(agent.data_service.PROSPECTS, {"LEAD-50001": record}), \
                            patch.object(agent.data_service, "get_rep", return_value=self.rep):
                        result = self.send(**({"from_rep": self.rep} if explicit_rep else {}))
                    self.assertTrue(result.pop("message_id").startswith("msg-"))
                    self.assertEqual(result, {
                        "status": "sent",
                        "to": self.prospect["email"],
                        "to_name": self.prospect["name"],
                        "from": self.rep["email"],
                        "from_name": self.rep["name"],
                        "subject": "Introduction",
                        "body": "Hello",
                    })

    def test_eligible_prospect_without_email_still_fails(self):
        self.prospect.pop("email")
        with patch.dict(agent.data_service.PROSPECTS, {"LEAD-50001": self.prospect}):
            self.assertEqual(self.send(from_rep=self.rep), {
                "status": "failed", "error": "Prospect record has no email address.",
            })


@unittest.skipUnless(os.environ.get("GTM_RUN_MODEL_TESTS") == "1", "Requires live model credentials")
class EmailAgentTests(unittest.TestCase):
    def test_disqualified_email_request_explains_block_and_asks_next_step(self):
        with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}):
            result = agent.run_agent(
                "Email LEAD-50001 with subject Introduction and body Hello.",
                user_id=agent.REP_IDS[0]["rep_id"],
            )
        reply = result["reply"].lower()
        self.assertIn("disqualified", reply)
        self.assertIn("?", reply)
        self.assertTrue(any(word in reply for word in ("proceed", "instead", "next")), reply)
        sends = [message for message in result["messages"]
                 if getattr(message, "name", None) == "send_prospect_email"]
        self.assertTrue(all(json.loads(message.content)["status"] == "blocked" for message in sends))
        self.assertNotRegex(reply, r"(?:i have|i've|successfully) sent|(?:^|[.!]\s+)the email (?:has been|was) sent")


if __name__ == "__main__":
    unittest.main()
