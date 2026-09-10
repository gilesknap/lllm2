"""Context search finds the loadable window quickly, then confirms it with a workload."""

import threading
import unittest
from unittest.mock import Mock, patch

from lllm2.bench import Bench
from lllm2.settings import Settings


def run_search(behaviour, selected=32768, max_context=131072, confirm=False):
    """Run a search where `behaviour(tokens_per_slot, stage)` decides each probe.

    `stage` is "start" for the engine load and "measure" for the full workload;
    the return value is "ok", "fail" or "timeout".
    """
    bench = Bench.__new__(Bench)
    bench.engine = Mock()
    bench.engine.state.return_value = {"logs": []}
    bench.store = Mock()
    bench.lock = threading.RLock()
    bench.cancel = threading.Event()
    bench.progress = {}
    loads, workloads = [], []

    def outcome(candidate, stage, record):
        per_slot = candidate.context // candidate.slots
        record.append(per_slot)
        result = behaviour(per_slot, stage)
        if result == "timeout":
            raise TimeoutError(f"{stage} exceeded 900s wall-clock timeout.")
        if result == "fail":
            raise RuntimeError("Engine exited during load. See engine log.")

    bench.engine.start = lambda candidate, cancel, timeout: outcome(
        candidate, "start", loads
    )

    def measure(candidate, workload, tokens, output, timeout):
        outcome(candidate, "measure", workloads)
        return {"input_tokens": tokens}

    bench.measure = measure
    s = Settings(model="m.gguf", context=selected, slots=1)
    opts = {
        "max_context": max_context,
        "output_tokens": 1024,
        "context_timeout": 900,
        "full_window": confirm,
    }
    r = {"id": "r", "probes": [], "samples": []}
    with (
        patch("lllm2.bench.metadata", return_value={"context": 262144}),
        patch("lllm2.bench.hardware", return_value={"gpus": ["GPU0"], "error": None}),
    ):
        bench.context_search(s, opts, r)
    return loads, workloads, r


def loads_up_to(limit):
    return lambda n, stage: "fail" if stage == "start" and n > limit else "ok"


class ContextSearchTests(unittest.TestCase):
    def test_loads_double_to_ceiling_without_a_prompt(self):
        loads, workloads, r = run_search(lambda n, stage: "ok")
        self.assertEqual(loads, [32768, 65536, 131072])
        self.assertEqual(workloads, [])
        self.assertEqual(r["context_search_status"], "complete")
        self.assertEqual(r["largest_observed_context"], 131072)
        self.assertEqual(r["largest_started_context"], 131072)
        self.assertFalse(r["context_confirmed"])
        self.assertTrue(r["context_ceiling_reached"])
        self.assertEqual([p["kind"] for p in r["probes"]], ["startup"] * 3)

    def test_full_window_option_confirms_with_one_workload(self):
        loads, workloads, r = run_search(lambda n, stage: "ok", confirm=True)
        self.assertEqual(loads, [32768, 65536, 131072, 131072])
        self.assertEqual(workloads, [131072])
        self.assertEqual(r["context_search_status"], "complete")
        self.assertEqual(r["largest_observed_context"], 131072)
        self.assertTrue(r["context_confirmed"])
        self.assertTrue(r["context_ceiling_reached"])
        self.assertEqual(
            [p["kind"] for p in r["probes"]], ["startup"] * 3 + ["workload"]
        )

    def test_refused_load_is_a_memory_limit_found_without_prefill(self):
        loads, workloads, r = run_search(loads_up_to(100000))
        self.assertEqual(loads[:3], [32768, 65536, 131072])
        self.assertEqual(workloads, [])
        self.assertEqual(r["context_search_status"], "complete")
        self.assertEqual(
            r["largest_observed_context"],
            max(loads_ok := [n for n in loads if n <= 100000]),
        )
        self.assertEqual(r["largest_started_context"], max(loads_ok))
        self.assertFalse(r["context_confirmed"])
        self.assertLessEqual(r["largest_observed_context"], 100000)
        self.assertLessEqual(
            r["context_failed_upper_bound"] - r["largest_observed_context"],
            r["context_search_resolution"],
        )
        self.assertLessEqual(len(loads), 13)

    def test_workload_timeout_bounds_search_without_ending_it(self):
        loads, workloads, r = run_search(
            lambda n, stage: "timeout" if stage == "measure" and n >= 131072 else "ok",
            confirm=True,
        )
        self.assertEqual(workloads[0], 131072)
        self.assertGreater(len(workloads), 1)
        self.assertTrue(all(32768 < n < 131072 for n in workloads[1:]))
        self.assertEqual(r["context_search_status"], "inconclusive_timeout")
        self.assertEqual(r["largest_observed_context"], max(workloads[1:]))
        self.assertEqual(r["context_timeout_upper_bound"], 131072)
        self.assertIsNone(r["context_failed_upper_bound"])
        self.assertIn("timed out at 131,072", r["context_search_stop_reason"])
        self.assertEqual(r["largest_started_context"], 131072)

    def test_workload_failure_below_loaded_size_is_bisected(self):
        limit = 70000
        loads, workloads, r = run_search(
            lambda n, stage: "fail" if stage == "measure" and n > limit else "ok",
            confirm=True,
        )
        self.assertEqual(workloads[0], 131072)
        self.assertEqual(r["context_search_status"], "complete")
        self.assertLessEqual(r["largest_observed_context"], limit)
        self.assertLessEqual(
            r["context_failed_upper_bound"] - r["largest_observed_context"],
            r["context_search_resolution"],
        )
        self.assertLessEqual(len(workloads), 8)

    def test_load_checks_start_at_the_experiment_window(self):
        loads, workloads, r = run_search(lambda n, stage: "ok", selected=8192)
        self.assertEqual(loads[:3], [8192, 16384, 32768])
        self.assertEqual(r["context_search_status"], "complete")

    def test_only_the_experiment_window_loads(self):
        loads, workloads, r = run_search(loads_up_to(32768))
        self.assertEqual(workloads, [])
        self.assertEqual(r["largest_observed_context"], 32768)
        self.assertEqual(r["context_search_status"], "complete")
        self.assertLessEqual(
            r["context_failed_upper_bound"] - 32768, r["context_search_resolution"]
        )

    def test_nothing_loads(self):
        loads, workloads, r = run_search(lambda n, stage: "fail")
        # Bisects down from the experiment window to the floor (1,280 tokens for
        # a 1,024-token output budget), then stops.
        self.assertEqual(loads[0], 32768)
        self.assertEqual(loads, sorted(loads, reverse=True))
        self.assertEqual(loads[-1], 1280)
        self.assertEqual(workloads, [])
        self.assertIsNone(r["largest_observed_context"])
        self.assertEqual(r["context_search_status"], "complete")
        self.assertEqual(r["context_failed_upper_bound"], 1280)


if __name__ == "__main__":
    unittest.main()
