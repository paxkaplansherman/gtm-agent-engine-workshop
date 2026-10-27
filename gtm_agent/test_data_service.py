import copy
import json
import os
import unittest
from unittest.mock import patch

from gtm_agent import data_service

with patch.dict(os.environ, {"OPENAI_API_KEY": "test", "LANGSMITH_TRACING": "false"}):
    from gtm_agent import gtm_agent


class UpdateProspectInfoTests(unittest.TestCase):
    def setUp(self):
        self.records = patch.dict(data_service.PROSPECTS, copy.deepcopy(data_service.PROSPECTS))
        self.records.start()
        self.addCleanup(self.records.stop)
        self.profiles = patch.dict(data_service._PROFILES, {}, clear=True)
        self.profiles.start()
        self.addCleanup(self.profiles.stop)
        self.tracing = patch.dict(os.environ, {"LANGSMITH_TRACING": "false"})
        self.tracing.start()
        self.addCleanup(self.tracing.stop)

    def test_update_persists_stack_and_refreshes_cached_profile(self):
        for prospect_id, technology in (("LEAD-71001", "Kafka"), ("LEAD-39002", "Terraform")):
            with self.subTest(prospect_id=prospect_id):
                old_profile = gtm_agent.build_prospect_profile.invoke({"prospect_id": prospect_id})
                old_stack = old_profile["prospect_profile"]["tech_stack"]
                self.assertNotIn(technology, old_stack)

                result = data_service.update_prospect_info(prospect_id, technology)

                expected = old_stack + [technology]
                self.assertEqual(result, {"updated": True, "found": True, "tech_stack": expected})
                self.assertEqual(data_service.fetch_tech_stack(prospect_id), expected)
                profile = gtm_agent.build_prospect_profile.invoke({"prospect_id": prospect_id})
                self.assertTrue(profile["found"])
                self.assertEqual(profile["prospect_profile"]["tech_stack"], expected)

    def test_repeated_update_does_not_duplicate_technology(self):
        data_service.update_prospect_info("LEAD-71001", "Kafka")
        result = data_service.update_prospect_info("LEAD-71001", "Kafka")
        self.assertTrue(result["updated"])
        self.assertEqual(data_service.fetch_tech_stack("LEAD-71001").count("Kafka"), 1)

    def test_missing_prospect_is_not_updated(self):
        self.assertEqual(
            data_service.update_prospect_info("missing-prospect", "Kafka"),
            {"updated": False, "found": False},
        )
        self.assertIsNone(data_service.get_prospect_record("missing-prospect"))

    def test_scoring_uses_persisted_stack_even_with_old_profile(self):
        profile = gtm_agent.build_prospect_profile.invoke({"prospect_id": "LEAD-71001"})
        offering = data_service.get_offering("OFFER-10005")
        data_service.update_prospect_info("LEAD-71001", "Kafka")

        def score_saved_stack(messages):
            saved_profile = json.loads(messages[1]["content"].split("Prospect profile:\n", 1)[1])
            stack = saved_profile["tech_stack"]
            self.assertIn("Kafka", stack)
            missing = [tech for tech in offering["required_tech_stack"] if tech not in stack]
            return gtm_agent.ProspectScore(
                score=90,
                justification=f"Has: {', '.join(stack)}. Missing: {', '.join(missing)}.",
                rubric_breakdown=gtm_agent.RubricBreakdown(
                    revenue_fit=90, tech_stack_match=90, segment_fit=90,
                ),
            )

        with patch.object(gtm_agent, "_scoring_llm") as scoring_llm:
            scoring_llm.invoke.side_effect = score_saved_stack
            result = gtm_agent.score_prospect.invoke({
                "prospect_profile": profile["prospect_profile"], "offering": offering,
            })

        self.assertIn("Kafka", result["justification"].split("Missing:")[0])
        self.assertNotIn("Kafka", result["justification"].split("Missing:")[1])


if __name__ == "__main__":
    unittest.main()
