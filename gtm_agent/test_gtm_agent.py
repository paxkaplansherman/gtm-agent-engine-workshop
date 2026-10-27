import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage

with patch.dict(os.environ, {"OPENAI_API_KEY": "test", "LANGSMITH_TRACING": "false"}):
    from gtm_agent import gtm_agent as agent


class ToolCallingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class AgentToolTests(unittest.TestCase):
    def setUp(self):
        self.tracing = patch.dict(os.environ, {"LANGSMITH_TRACING": "false"})
        self.tracing.start()
        self.addCleanup(self.tracing.stop)
        self.profiles = patch.dict(agent.data_service._PROFILES, {}, clear=True)
        self.profiles.start()
        self.addCleanup(self.profiles.stop)
        self.prospect_id = next(iter(agent.data_service.PROSPECTS))
        self.offering_id = "OFFER-10001"
        self.rep = agent.REP_IDS[0]
        self.runtime = SimpleNamespace(config={"metadata": {"user_id": self.rep["rep_id"]}})
        self.score = agent.ProspectScore(
            score=75, justification="Fit assessment", rubric_breakdown={
                "revenue_fit": 80, "tech_stack_match": 70, "segment_fit": 75,
            },
        )

    def test_scoring_resolves_records_and_preserves_prompt_and_result(self):
        with patch.object(agent, "_scoring_llm") as scoring:
            scoring.invoke.return_value = self.score
            result = agent.score_prospect.invoke({
                "prospect_id": self.prospect_id, "offering_id": self.offering_id,
            })
        self.assertEqual(result, self.score.model_dump())
        messages = scoring.invoke.call_args.args[0]
        self.assertEqual(messages[0], {"role": "system", "content": agent.SCORING_PROMPT})
        offering_text, profile_text = messages[1]["content"].removeprefix("Offering:\n").split(
            "\n\nProspect profile:\n"
        )
        self.assertEqual(json.loads(offering_text), agent.data_service.get_offering(self.offering_id))
        expected = {"prospect_id": self.prospect_id, **agent.data_service.get_prospect_record(self.prospect_id)}
        self.assertEqual(json.loads(profile_text), expected)
        self.assertEqual(agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id}), {
            "prospect_profile": expected, "found": True,
        })

    def test_scoring_uses_current_tech_stack_for_cached_profile(self):
        agent.data_service._PROFILES[self.prospect_id] = {
            **agent._build_prospect_profile(self.prospect_id)["prospect_profile"],
            "tech_stack": ["stale"],
        }
        with patch.object(agent, "_scoring_llm") as scoring:
            scoring.invoke.return_value = self.score
            agent.score_prospect.invoke({"prospect_id": self.prospect_id, "offering_id": self.offering_id})
        profile = json.loads(scoring.invoke.call_args.args[0][1]["content"].split("Prospect profile:\n")[1])
        self.assertEqual(profile["tech_stack"], agent.data_service.fetch_tech_stack(self.prospect_id))

    def test_invalid_offering(self):
        offering = agent.data_service.get_offering(self.offering_id)
        for field in ("required_tech_stack", "min_annual_revenue", "description"):
            with self.subTest(field=field), patch.object(
                agent.data_service, "get_offering", return_value={k: v for k, v in offering.items() if k != field}
            ):
                self.assertEqual(agent.score_prospect.invoke({
                    "prospect_id": self.prospect_id, "offering_id": self.offering_id,
                }), {"score": None, "error": "Cannot score without a valid offering."})

    def test_email_resolves_contact_and_sender_and_preserves_content(self):
        result = agent.send_prospect_email.func(self.prospect_id, "A subject", "Exact body\nThanks!", self.runtime)
        prospect = agent.data_service.get_prospect_record(self.prospect_id)
        self.assertEqual(result["status"], "sent")
        self.assertTrue(result["message_id"].startswith("msg-"))
        self.assertEqual({k: v for k, v in result.items() if k not in ("status", "message_id")}, {
            "to": prospect["email"], "to_name": prospect["name"],
            "from": self.rep["email"], "from_name": self.rep["name"],
            "subject": "A subject", "body": "Exact body\nThanks!",
        })

    def test_missing_prospect(self):
        self.assertEqual(agent.send_prospect_email.func("missing", "Subject", "Body", self.runtime), {
            "status": "failed", "error": "Prospect record not found.",
        })
        self.assertEqual(agent.score_prospect.invoke({
            "prospect_id": "missing", "offering_id": self.offering_id,
        }), {"score": None, "error": "Prospect record not found."})

    def test_missing_email(self):
        for email in (None, ""):
            with self.subTest(email=email), patch.object(agent.data_service, "get_prospect_record", return_value={"email": email}):
                self.assertEqual(agent.send_prospect_email.func(self.prospect_id, "Subject", "Body", self.runtime), {
                    "status": "failed", "error": "Prospect record has no email address.",
                })

    def run_tool_request(self, name, args, request):
        model = ToolCallingModel(responses=[
            AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "call-1"}]),
            AIMessage(content="Done"),
        ])
        graph = agent.create_deep_agent(
            model=model,
            tools=[agent.score_prospect, agent.send_prospect_email, agent.get_current_rep],
            system_prompt=agent.SYSTEM_PROMPT,
        )
        result = graph.invoke({"messages": [{"role": "user", "content": request}]}, config=self.runtime.config)
        calls = [call for message in result["messages"] if isinstance(message, AIMessage) for call in message.tool_calls]
        self.assertEqual([call["name"] for call in calls], [name])
        self.assertEqual(len([m for m in result["messages"] if isinstance(m, AIMessage)]), 2)
        return json.loads(next(m.content for m in result["messages"] if isinstance(m, ToolMessage)))

    def test_graph_supports_direct_scoring_and_email_and_explicit_identity(self):
        with patch.object(agent, "_scoring_llm") as scoring:
            scoring.invoke.return_value = self.score
            result = self.run_tool_request("score_prospect", {
                "prospect_id": self.prospect_id, "offering_id": self.offering_id,
            }, f"Score {self.prospect_id} for {self.offering_id}")
        self.assertEqual(result, self.score.model_dump())
        result = self.run_tool_request("send_prospect_email", {
            "prospect_id": self.prospect_id, "subject": "Hello", "body": "Model-written content",
        }, f"Email {self.prospect_id}")
        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["from"], self.rep["email"])
        self.assertEqual(result["subject"], "Hello")
        self.assertEqual(result["body"], "Model-written content")
        self.assertEqual(self.run_tool_request("get_current_rep", {}, "Who am I?"), {
            "rep": self.rep, "found": True,
        })


if __name__ == "__main__":
    unittest.main()
