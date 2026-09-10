"""Context search doubles from the last success and keeps narrowing after a timeout."""

import threading
import unittest
from unittest.mock import Mock, patch

from lllm2.bench import Bench
from lllm2.settings import Settings


def run_search(behaviour, seed=True, selected=32768, max_context=131072):
    """Run a context search where `behaviour(tokens_per_slot)` decides each probe."""
    bench = Bench.__new__(Bench)
    bench.engine = Mock()
    bench.engine.state.return_value = {"logs": []}
    bench.store = Mock()
    bench.lock = threading.RLock()
    bench.cancel = threading.Event()
    bench.progress = {}
    probed = []

    def measure(candidate, workload, tokens, output, timeout):
        per_slot = candidate.context // candidate.slots
        probed.append(per_slot)
        outcome = behaviour(per_slot)
        if outcome == "timeout":
            raise TimeoutError("/completion exceeded 900s wall-clock timeout.")
        if outcome == "fail":
            raise RuntimeError("Engine exited during load. See engine log.")
        return {"input_tokens": tokens}

    bench.measure = measure
    s = Settings(model="m.gguf", context=selected, slots=1)
    opts = {"max_context": max_context, "output_tokens": 1024, "context_timeout": 900}
    r = {
        "id": "r",
        "probes": [],
        "samples": [
            {"input_tokens": selected - 1024 - 32, "context_per_slot": selected}
        ]
        if seed
        else [],
    }
    with (
        patch("lllm2.bench.metadata", return_value={"context": 262144}),
        patch("lllm2.bench.hardware", return_value={"gpus": ["GPU0"], "error": None}),
    ):
        bench.context_search(s, opts, r)
    return probed, r


class ContextSearchTests(unittest.TestCase):
    def test_doubles_from_seed_to_ceiling(self):
        probed, r = run_search(lambda n: "ok")
        self.assertEqual(probed, [65536, 131072])
        self.assertEqual(r["context_search_status"], "complete")
        self.assertEqual(r["largest_observed_context"], 131072)
        self.assertTrue(r["context_ceiling_reached"])
        self.assertEqual(r["context_seed_from_speed_sample"], 32768)

    def test_timeout_bounds_search_without_ending_it(self):
        probed, r = run_search(lambda n: "timeout" if n >= 131072 else "ok")
        self.assertEqual(probed[:2], [65536, 131072])
        self.assertGreater(len(probed), 2)
        self.assertTrue(all(65536 < n < 131072 for n in probed[2:]))
        self.assertEqual(r["context_search_status"], "inconclusive_timeout")
        self.assertEqual(r["largest_observed_context"], max(probed[2:]))
        self.assertGreaterEqual(r["largest_observed_context"], 131072 * 0.9 - 12288)
        self.assertEqual(r["context_timeout_upper_bound"], 131072)
        self.assertIsNone(r["context_failed_upper_bound"])
        self.assertIn("timed out at 131,072", r["context_search_stop_reason"])
        self.assertEqual(r["probes"][1]["status"], "timed_out")

    def test_failure_is_bisected_to_resolution(self):
        limit = 100000
        probed, r = run_search(lambda n: "fail" if n > limit else "ok")
        self.assertEqual(probed[:2], [65536, 131072])
        self.assertEqual(r["context_search_status"], "complete")
        self.assertLessEqual(r["largest_observed_context"], limit)
        self.assertLessEqual(
            r["context_failed_upper_bound"] - r["largest_observed_context"],
            r["context_search_resolution"],
        )
        self.assertIsNone(r["context_timeout_upper_bound"])
        self.assertLessEqual(len(probed), 8)

    def test_real_failure_below_timeout_is_still_conclusive(self):
        def behaviour(n):
            if n >= 131072:
                return "timeout"
            return "fail" if n > 70000 else "ok"

        probed, r = run_search(behaviour)
        self.assertEqual(r["context_search_status"], "complete")
        self.assertEqual(r["context_timeout_upper_bound"], 131072)
        self.assertLessEqual(r["context_failed_upper_bound"], 98304)
        self.assertLessEqual(r["largest_observed_context"], 70000)

    def test_without_seed_starts_at_selected_context(self):
        probed, r = run_search(lambda n: "ok", seed=False)
        self.assertEqual(probed[:2], [32768, 65536])
        self.assertNotIn("context_seed_from_speed_sample", r)
        self.assertEqual(r["context_search_status"], "complete")

    def test_timeouts_narrow_down_to_the_seed(self):
        # Every probe above the seed times out; the search narrows to the seed
        # within resolution instead of stopping after the first timeout.
        probed, r = run_search(lambda n: "timeout" if n > 32768 else "ok")
        self.assertEqual(probed[0], 65536)
        self.assertLessEqual(len(probed), 8)
        self.assertEqual(r["largest_observed_context"], 32768)
        self.assertEqual(r["context_search_status"], "inconclusive_timeout")
        self.assertLessEqual(
            r["context_timeout_upper_bound"] - 32768, r["context_search_resolution"]
        )


if __name__ == "__main__":
    unittest.main()
