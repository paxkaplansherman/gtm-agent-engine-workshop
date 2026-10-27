import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

with patch("langchain_openai.ChatOpenAI"), patch("deepagents.create_deep_agent"):
    from gtm_agent import gtm_agent as agent


class ProspectPrivacyTests(unittest.TestCase):
    def setUp(self):
        self.prospect_id = "LEAD-12853"
        self.record = copy.deepcopy(agent.data_service.PROSPECTS[self.prospect_id])
        self.record["billing_qualification"] = {"tax_id": "private-tax-value"}
        self.record["unlisted_field"] = "private-unlisted-value"
        self.expected = {
            "prospect_id": self.prospect_id,
            **{key: self.record[key] for key in (
                "name", "email", "disqualified", "annual_revenue", "tech_stack",
                "engagement_history", "account_details",
            )},
        }
        self.enterContext(patch.dict(
            agent.data_service.PROSPECTS, {self.prospect_id: self.record}
        ))
        self.enterContext(patch.object(agent.data_service, "_PROFILES", {}))
        self.enterContext(patch.dict("os.environ", {"LANGSMITH_TRACING": "false"}))

    def test_contact_fields_support_email(self):
        result = agent.get_prospect.invoke({"prospect_id": self.prospect_id})
        contact = result["prospect"]
        self.assertTrue(result["found"])
        self.assertEqual(contact, {
            key: self.expected[key]
            for key in ("prospect_id", "name", "email", "disqualified")
        })
        sent = agent.send_prospect_email.func(
            contact, "Introduction", "Hello", SimpleNamespace(config={}),
            from_rep={"name": "Rep", "email": "rep@example.com"},
        )
        self.assertEqual(sent["status"], "sent")
        self.assertEqual(sent["to"], self.record["email"])
        self.assertEqual(sent["to_name"], self.record["name"])

    def test_profile_is_filtered_before_storage_and_on_cache_hit(self):
        for _ in range(2):
            result = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
            self.assertEqual(result, {"prospect_profile": self.expected, "found": True})
            self.assertEqual(agent.data_service._PROFILES[self.prospect_id], self.expected)

    def test_legacy_cached_profile_is_filtered(self):
        agent.data_service._PROFILES[self.prospect_id] = self.record
        result = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        self.assertEqual(result["prospect_profile"], {
            key: value for key, value in self.expected.items() if key != "prospect_id"
        })

    def test_scoring_filters_external_and_built_profiles(self):
        built = agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        offering = agent.lookup_offering.invoke({"offering_id": "OFFER-10001"})["offering"]
        profiles = [self.record, {**self.record, "prospect_id": self.prospect_id},
                    built["prospect_profile"]]
        for profile in profiles:
            with self.subTest(has_id="prospect_id" in profile), patch.object(
                agent, "_scoring_llm"
            ) as scorer:
                scorer.invoke.return_value.model_dump.return_value = {"score": 75}
                result = agent.score_prospect.invoke({
                    "prospect_profile": profile, "offering": offering,
                })
                self.assertEqual(result, {"score": 75})
                content = scorer.invoke.call_args.args[0][1]["content"]
                serialized = json.loads(content.split("\n\nProspect profile:\n", 1)[1])
                expected = dict(self.expected)
                if "prospect_id" not in profile:
                    expected.pop("prospect_id")
                self.assertEqual(serialized, expected)
                self.assertNotIn("billing_qualification", content)
                self.assertNotIn("private-tax-value", content)
                self.assertNotIn("private-unlisted-value", content)
        self.assertIn("billing_qualification", self.record)

    def test_missing_prospect_still_returns_not_found(self):
        self.assertEqual(agent.get_prospect.invoke({"prospect_id": "missing"}), {
            "prospect": None, "found": False,
        })
        self.assertEqual(agent.build_prospect_profile.invoke({"prospect_id": "missing"}), {
            "prospect_profile": None, "found": False,
        })


if __name__ == "__main__":
    unittest.main()
