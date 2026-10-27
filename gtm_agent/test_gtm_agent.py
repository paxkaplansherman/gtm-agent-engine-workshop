import json
import os
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI

with patch.dict(
    os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}
):
    from gtm_agent import gtm_agent as agent


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {"LANGSMITH_TRACING": "false"})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.calls = []

    def run_with_responses(self, responses):
        def generate(model, messages, **kwargs):
            self.calls.append((messages, kwargs))
            return ChatResult(generations=[ChatGeneration(message=next(responses))])

        with patch.object(ChatOpenAI, "_generate", autospec=True, side_effect=generate):
            return agent.run_agent(
                "Email the prospect",
                user_id=agent.REP_IDS[0]["rep_id"],
                environment="test",
                thread_id="test-thread",
            )

    def test_model_receives_only_gtm_tools_and_app_context(self):
        self.run_with_responses(iter([AIMessage(content="Done")]))
        messages, kwargs = self.calls[0]
        self.assertEqual(
            {tool["function"]["name"] for tool in kwargs["tools"]},
            {
                "lookup_offering",
                "build_prospect_profile",
                "get_prospect",
                "send_prospect_email",
                "score_prospect",
                "update_prospect_info",
                "get_current_rep",
            },
        )
        self.assertEqual(len(kwargs["tools"]), 7)
        for tool in kwargs["tools"]:
            self.assertNotIn(
                "runtime", tool["function"]["parameters"].get("properties", {})
            )
        self.assertEqual(
            [
                message.content
                for message in messages
                if isinstance(message, SystemMessage)
            ],
            [agent.SYSTEM_PROMPT],
        )

    def test_email_flow_uses_signed_in_rep_metadata(self):
        prospect_id = next(iter(agent.data_service.PROSPECTS))
        prospect = agent.data_service.get_prospect_record(prospect_id)
        rep = agent.REP_IDS[0]
        responses = iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "get_current_rep", "args": {}, "id": "rep"},
                        {
                            "name": "get_prospect",
                            "args": {"prospect_id": prospect_id},
                            "id": "prospect",
                        },
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "send_prospect_email",
                            "id": "email",
                            "args": {
                                "prospect": prospect,
                                "subject": "Hello",
                                "body": "Let's connect.",
                            },
                        }
                    ],
                ),
                AIMessage(content="Email sent"),
            ]
        )
        result = self.run_with_responses(responses)
        outputs = {
            message.tool_call_id: json.loads(message.content)
            for message in result["messages"]
            if isinstance(message, ToolMessage)
        }
        self.assertEqual(outputs["rep"], {"rep": rep, "found": True})
        self.assertEqual(outputs["prospect"]["prospect"]["email"], prospect["email"])
        self.assertEqual(outputs["email"]["status"], "sent")
        self.assertEqual(outputs["email"]["from"], rep["email"])
        self.assertEqual(outputs["email"]["from_name"], rep["name"])
        self.assertEqual(outputs["email"]["to"], prospect["email"])
        self.assertEqual(result["reply"], "Email sent")
        self.assertEqual(len(self.calls), 3)
        for messages, _ in self.calls:
            self.assertEqual(
                [
                    message.content
                    for message in messages
                    if isinstance(message, SystemMessage)
                ],
                [agent.SYSTEM_PROMPT],
            )


if __name__ == "__main__":
    unittest.main()
