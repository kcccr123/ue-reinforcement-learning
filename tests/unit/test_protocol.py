"""Unit tests for MsgSerializer: encode/decode round-trips for all 6 message types."""
import struct
import pytest

from rl_platform.infra.communication import MsgSerializer


@pytest.fixture
def codec():
    return MsgSerializer()


# ---------------------------------------------------------------------------
# Framing
# ---------------------------------------------------------------------------

class TestFraming:
    def test_encode_prepends_4_byte_big_endian_length(self, codec):
        raw = codec.encode({"type": "close"})
        length_prefix = struct.unpack(">I", raw[:4])[0]
        assert length_prefix == len(raw) - 4

    def test_decode_recovers_payload_without_header(self, codec):
        msg = {"type": "close"}
        raw = codec.encode(msg)
        payload = raw[4:]
        decoded = codec.decode_data(payload)
        assert decoded == msg

    def test_empty_dict_round_trips(self, codec):
        msg = {}
        raw = codec.encode(msg)
        decoded = codec.decode_data(raw[4:])
        assert decoded == msg


# ---------------------------------------------------------------------------
# Round-trip for each message type
# ---------------------------------------------------------------------------

class TestHandshakeRoundTrip:
    def test_single_agent(self, codec):
        msg = {
            "type": "handshake",
            "env_id": "combat_arena",
            "agents": [
                {"id": "agent_0", "obs_shape": [7], "act_shape": [6], "is_scripted": False}
            ],
            "metadata": {},
        }
        raw = codec.encode(msg)
        decoded = codec.decode_data(raw[4:])
        assert decoded == msg

    def test_multi_agent_with_scripted(self, codec):
        msg = {
            "type": "handshake",
            "env_id": "team_combat",
            "agents": [
                {"id": "fighter_0", "obs_shape": [12], "act_shape": [4], "is_scripted": False},
                {"id": "fighter_1", "obs_shape": [12], "act_shape": [4], "is_scripted": False},
                {"id": "carrier_0", "obs_shape": [8], "act_shape": [2], "is_scripted": True, "team_id": "blue"},
            ],
            "metadata": {"version": 1},
        }
        raw = codec.encode(msg)
        decoded = codec.decode_data(raw[4:])
        assert decoded == msg
        assert decoded["agents"][2]["is_scripted"] is True


class TestResetRoundTrip:
    def test_reset_message(self, codec):
        msg = {"type": "reset", "task": {"name": "solo", "env_params": {"difficulty": 0.5}}}
        decoded = codec.decode_data(codec.encode(msg)[4:])
        assert decoded == msg

    def test_reset_no_task(self, codec):
        msg = {"type": "reset"}
        decoded = codec.decode_data(codec.encode(msg)[4:])
        assert decoded == msg


class TestResetResultRoundTrip:
    def test_single_agent(self, codec):
        msg = {
            "type": "reset_result",
            "agents": {"agent_0": {"obs": [1.0, 2.0, 3.0]}},
        }
        decoded = codec.decode_data(codec.encode(msg)[4:])
        assert decoded == msg

    def test_multi_agent(self, codec):
        msg = {
            "type": "reset_result",
            "agents": {
                "fighter_0": {"obs": [1.0, 2.0]},
                "fighter_1": {"obs": [3.0, 4.0]},
            },
        }
        decoded = codec.decode_data(codec.encode(msg)[4:])
        assert decoded == msg


class TestStepRoundTrip:
    def test_single_agent(self, codec):
        msg = {"type": "step", "actions": {"agent_0": [0.5, -0.3, 1.0]}}
        decoded = codec.decode_data(codec.encode(msg)[4:])
        assert decoded == msg

    def test_multi_agent(self, codec):
        msg = {
            "type": "step",
            "actions": {
                "fighter_0": [0.1, 0.2],
                "fighter_1": [-0.5, 0.8],
            },
        }
        decoded = codec.decode_data(codec.encode(msg)[4:])
        assert decoded == msg


class TestStepResultRoundTrip:
    def test_single_agent(self, codec):
        msg = {
            "type": "step_result",
            "agents": {
                "agent_0": {"obs": [1.0, 2.0], "reward": 0.5, "done": False, "info": {}},
            },
            "global": {"done": False},
        }
        decoded = codec.decode_data(codec.encode(msg)[4:])
        assert decoded == msg

    def test_episode_done(self, codec):
        msg = {
            "type": "step_result",
            "agents": {
                "agent_0": {"obs": [0.0, 0.0], "reward": -1.0, "done": True, "info": {"reason": "fell"}},
            },
            "global": {"done": True},
        }
        decoded = codec.decode_data(codec.encode(msg)[4:])
        assert decoded == msg
        assert decoded["global"]["done"] is True
        assert decoded["agents"]["agent_0"]["done"] is True


class TestCloseRoundTrip:
    def test_close(self, codec):
        msg = {"type": "close"}
        decoded = codec.decode_data(codec.encode(msg)[4:])
        assert decoded == msg


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_large_observation_preserves_precision(self, codec):
        obs = [float(i) * 0.001 for i in range(1000)]
        msg = {"type": "step_result", "agents": {"a0": {"obs": obs, "reward": 0.0, "done": False, "info": {}}}, "global": {"done": False}}
        decoded = codec.decode_data(codec.encode(msg)[4:])
        assert decoded["agents"]["a0"]["obs"] == pytest.approx(obs)

    def test_negative_reward(self, codec):
        msg = {"type": "step_result", "agents": {"a0": {"obs": [0.0], "reward": -99.5, "done": False, "info": {}}}, "global": {"done": False}}
        decoded = codec.decode_data(codec.encode(msg)[4:])
        assert decoded["agents"]["a0"]["reward"] == -99.5

    def test_nested_info_dict(self, codec):
        msg = {
            "type": "step_result",
            "agents": {"a0": {"obs": [0.0], "reward": 0.0, "done": False, "info": {"team_id": "blue", "kills": 3}}},
            "global": {"done": False},
        }
        decoded = codec.decode_data(codec.encode(msg)[4:])
        assert decoded["agents"]["a0"]["info"]["team_id"] == "blue"
        assert decoded["agents"]["a0"]["info"]["kills"] == 3
