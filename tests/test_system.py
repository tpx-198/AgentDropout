import asyncio
import os
import unittest
from unittest.mock import patch

import torch
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage

from ad_l.models import ModelAdapter, ModelSettings, create_model
from ad_l.roles import create_team
from ad_l.system import build_system
from ad_l.topology import EdgePolicy, TopologyConfig


class SystemTests(unittest.TestCase):
    def adapters(self, team):
        return {agent.node_id: create_model(mock_response=f"{agent.node_id} The answer is 5") for agent in team}

    def test_five_agents_and_scaling(self):
        team = create_team()
        self.assertEqual([agent.role for agent in team],
                         ["Math Solver", "Mathematical Analyst", "Programming Expert", "Inspector", "FinalRefer"])
        for count in (2, 5, 12):
            config = TopologyConfig(agents=count, rounds=3)
            policy = EdgePolicy(config)
            team = create_team(count)
            execution = policy.sample(torch.Generator().manual_seed(0), mode="fixed").execution
            state = build_system(team, execution, self.adapters(team)).invoke({"task": "3 + 2?"})
            self.assertEqual(state["status"], "success")
            self.assertEqual(state["model_calls"], (count - 1) * 3 + 1)
            self.assertEqual(state["real_api_calls"], 0)

    def test_visibility_round_snapshots_and_final_sources(self):
        team = create_team()
        execution = {"rounds": [
            {"spatial": [[0, 1], [1, 2]], "temporal": []},
            {"spatial": [[2, 0]], "temporal": [[0, 0], [3, 1]]},
        ], "summary": [0, 3]}
        adapters = self.adapters(team)
        graph = build_system(team, execution, adapters)
        first = graph.invoke({"task": "FIRST_TASK"})
        first_round = {call["node_id"]: call for call in first["trace"] if call["round"] == 1}
        self.assertEqual(first_round["agent_2"]["spatial"].keys(), {"agent_1"})
        self.assertEqual(first_round["agent_3"]["spatial"].keys(), {"agent_2"})
        second_round = {call["node_id"]: call for call in first["trace"] if call["round"] == 2 and call["stage"] == "discussion"}
        self.assertEqual(second_round["agent_1"]["spatial"].keys(), {"agent_3"})
        self.assertEqual(second_round["agent_1"]["temporal"]["agent_1"]["output"], first["round_outputs"][0]["agent_1"])
        self.assertEqual(second_round["agent_2"]["temporal"].keys(), {"agent_4"})
        self.assertEqual(first["trace"][-1]["spatial"].keys(), {"agent_1", "agent_4"})
        second = asyncio.run(graph.ainvoke({"task": "SECOND_TASK"}))
        self.assertEqual(second["model_calls"], 9)
        self.assertNotIn("FIRST_TASK", str(second))
        self.assertIsNone(second["usage"])

    def test_summary_edges_follow_pruning(self):
        policy = EdgePolicy(TopologyConfig())
        with torch.no_grad():
            policy.summary_logits.copy_(torch.tensor([-2.0, -1.0, 1.0, 2.0]))
        policy.prune(0.5)
        execution = policy.sample(torch.Generator().manual_seed(0), mode="fixed").execution
        team = create_team()
        state = build_system(team, execution, self.adapters(team)).invoke({"task": "3 + 2?"})
        self.assertEqual(state["trace"][-1]["spatial"].keys(), {"agent_3", "agent_4"})

    def test_model_failure_stops_and_hides_error_secrets(self):
        team = create_team()
        execution = EdgePolicy(TopologyConfig()).sample(torch.Generator(), mode="fixed").execution
        adapters = self.adapters(team)
        with patch.object(adapters["agent_1"], "invoke", side_effect=RuntimeError("private-api-key")):
            state = build_system(team, execution, adapters).invoke({"task": "3 + 2?"})
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["model_calls"], 1)
        self.assertEqual(state["answer"], "")
        self.assertNotIn("private-api-key", str(state))

    def test_real_models_require_opt_in_and_mock_does_not_read_env(self):
        with patch('dotenv.load_dotenv', side_effect=AssertionError('unexpected dotenv')):
            create_model()
        settings = ModelSettings(backend="deepseek", model="offline-stub")
        with self.assertRaisesRegex(ValueError, "allow_real_api"):
            create_model(settings)
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ValueError, "DEEPSEEK_API_KEY"):
            create_model(settings, allow_real_api=True)
        with patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'offline-placeholder'}, clear=True), \
                patch('langchain_deepseek.ChatDeepSeek', return_value=FakeListChatModel(responses=['The answer is 5'])) as client:
            adapter = create_model(settings, allow_real_api=True)
            self.assertEqual(client.call_args.kwargs['max_retries'], 0)
            team = create_team()
            execution = EdgePolicy(TopologyConfig()).sample(torch.Generator(), mode="fixed").execution
            adapters = {agent.node_id: adapter for agent in team}
            with self.assertRaisesRegex(ValueError, 'allow_real_api'):
                build_system(team, execution, adapters)
            state = build_system(team, execution, adapters, allow_real_api=True).invoke({'task': '3 + 2?'})
            self.assertEqual(state['real_api_calls'], 9)

    def test_invalid_topology_and_task_fail_before_model(self):
        team = create_team()
        adapters = self.adapters(team)
        bad = {'rounds': [{'spatial': [[0, 1], [1, 0]], 'temporal': []}], 'summary': [0]}
        with self.assertRaisesRegex(ValueError, 'DAG'):
            build_system(team, bad, adapters)
        good = {'rounds': [{'spatial': [], 'temporal': []}], 'summary': []}
        with patch.object(adapters['agent_1'], 'invoke') as invoke:
            with self.assertRaises(ValueError):
                build_system(team, good, adapters).invoke({'task': ''})
            invoke.assert_not_called()


if __name__ == "__main__":
    unittest.main()
