import os
import unittest
from importlib import import_module

os.environ.setdefault("SKIP_PLATFORM_CHECK", "1")


def _make_runner(config_dict):
    """A DefaultRunner with only ``config`` populated.

    set_config touches nothing else, so skipping __init__ keeps this free of
    model weights and CUDA.
    """
    default_runner = import_module("lightx2v.models.runners.default_runner")
    lockable = import_module("lightx2v.utils.lockable_dict")
    runner = default_runner.DefaultRunner.__new__(default_runner.DefaultRunner)
    runner.config = lockable.LockableDict(config_dict)
    return runner


class TestConfigPerRequestIsolation(unittest.TestCase):
    """set_config applies ONE request's overrides, not an accumulation of them.

    The server calls set_config(task_data) per request against a runner-lifetime
    config. Before the fix it only did `config.update(...)`, so a field the current
    request omitted silently kept the PREVIOUS request's value. Measured on a
    production qwen_image deployment pinned to infer_steps=8: an identical
    step-less request took 15.5s or 73.2s depending only on what the previous
    caller had sent.
    """

    def test_omitted_field_falls_back_to_deploy_config_not_previous_request(self):
        runner = _make_runner({"infer_steps": 8, "sample_guide_scale": 4.0})

        runner.set_config({"prompt": "a", "infer_steps": 40})
        self.assertEqual(runner.config["infer_steps"], 40)

        # The whole bug: this request says nothing about steps, so the deployment
        # default must come back — not the 40 the previous caller asked for.
        runner.set_config({"prompt": "b"})
        self.assertEqual(runner.config["infer_steps"], 8)

    def test_request_only_keys_are_dropped_when_the_next_request_omits_them(self):
        runner = _make_runner({"infer_steps": 8})

        runner.set_config({"prompt": "a", "lora_name": "some.safetensors"})
        self.assertEqual(runner.config["lora_name"], "some.safetensors")

        # lora_name never existed in the deploy config, so it must disappear
        # rather than silently apply a LoRA to somebody else's request.
        runner.set_config({"prompt": "b"})
        self.assertNotIn("lora_name", runner.config)

    def test_unrelated_config_is_left_alone(self):
        """Only keys this mechanism injected are reverted.

        Other code deliberately mutates the config post-init through
        temporarily_unlocked(); a blanket reset would undo that.
        """
        runner = _make_runner({"infer_steps": 8, "cpu_offload": True})
        runner.set_config({"infer_steps": 40})

        with runner.config.temporarily_unlocked():
            runner.config["cpu_offload"] = False
            runner.config["injected_elsewhere"] = 123

        runner.set_config({"prompt": "b"})
        self.assertFalse(runner.config["cpu_offload"])
        self.assertEqual(runner.config["injected_elsewhere"], 123)
        self.assertEqual(runner.config["infer_steps"], 8)

    def test_repeated_identical_requests_are_stable(self):
        runner = _make_runner({"infer_steps": 8})
        for _ in range(3):
            runner.set_config({"prompt": "x", "infer_steps": 20})
            self.assertEqual(runner.config["infer_steps"], 20)
            runner.set_config({"prompt": "x"})
            self.assertEqual(runner.config["infer_steps"], 8)


if __name__ == "__main__":
    unittest.main()


def _make_qwen_runner(config_dict):
    """A QwenImageRunner with only ``config`` populated (disagg mode on)."""
    qwen = import_module("lightx2v.models.runners.qwen_image.qwen_image_runner")
    lockable = import_module("lightx2v.utils.lockable_dict")
    runner = qwen.QwenImageRunner.__new__(qwen.QwenImageRunner)
    runner.config = lockable.LockableDict(config_dict)
    return runner


class TestDisaggRequestFieldIsolation(unittest.TestCase):
    """Disagg rooms/ranks must not leak either.

    QwenImageRunner.set_config runs the flat rollback and then
    apply_disagg_request_overrides, which writes DERIVED top-level fields
    (data_bootstrap_room) and NESTED ones (disagg_config[...]). The flat rollback
    sees neither, so before this fix a request that omitted the rooms inherited
    the previous caller's — and that value is consumed for transfer setup.
    """

    def _runner(self):
        return _make_qwen_runner(
            {
                "disagg_mode": "encoder",
                "disagg_config": {},
                "infer_steps": 8,
            }
        )

    def test_derived_and_nested_rooms_are_reverted(self):
        runner = self._runner()

        runner.set_config({"prompt": "a", "disagg_bootstrap_room": 111})
        self.assertEqual(runner.config["disagg_config"]["bootstrap_room"], 111)
        self.assertEqual(runner.config["data_bootstrap_room"], 111)

        # Request B says nothing about rooms: neither the nested value nor the
        # derived top-level one may survive.
        runner.set_config({"prompt": "b"})
        self.assertNotIn("bootstrap_room", runner.config["disagg_config"])
        self.assertNotIn("data_bootstrap_room", runner.config)

    def test_receiver_rank_is_reverted(self):
        runner = self._runner()

        runner.set_config({"prompt": "a", "disagg_phase1_receiver_engine_rank": 3})
        self.assertEqual(runner.config["disagg_config"]["receiver_engine_rank"], 3)
        self.assertEqual(runner.config["disagg_phase1_receiver_engine_rank"], 3)

        runner.set_config({"prompt": "b"})
        self.assertNotIn("receiver_engine_rank", runner.config["disagg_config"])
        self.assertNotIn("disagg_phase1_receiver_engine_rank", runner.config)

    def test_deploy_config_baseline_rooms_are_restored_not_deleted(self):
        """A room preset in the deploy config must come back, not vanish."""
        runner = _make_qwen_runner(
            {
                "disagg_mode": "encoder",
                "disagg_config": {"bootstrap_room": 7},
                "data_bootstrap_room": 7,
                "infer_steps": 8,
            }
        )

        runner.set_config({"prompt": "a", "disagg_bootstrap_room": 111})
        self.assertEqual(runner.config["disagg_config"]["bootstrap_room"], 111)

        runner.set_config({"prompt": "b"})
        self.assertEqual(runner.config["disagg_config"]["bootstrap_room"], 7)
        self.assertEqual(runner.config["data_bootstrap_room"], 7)

    def test_non_disagg_runner_is_untouched(self):
        """apply_disagg_request_overrides early-returns without disagg_mode."""
        runner = _make_qwen_runner({"infer_steps": 8})
        runner.set_config({"prompt": "a", "disagg_bootstrap_room": 111})
        runner.set_config({"prompt": "b"})
        self.assertEqual(runner.config["infer_steps"], 8)
        self.assertNotIn("data_bootstrap_room", runner.config)
