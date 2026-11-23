import json
import signal
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts import evaluate_fpe_sampling as evaluation


class PowerWatchdogTests(unittest.TestCase):
    def test_query_uses_physical_uuid_for_logical_cuda_device(self):
        for uuid in ("physical-device", "GPU-physical-device"):
            with self.subTest(uuid=uuid):
                with patch.object(evaluation.torch.cuda, "get_device_properties",
                                  return_value=SimpleNamespace(uuid=uuid)) as properties:
                    with patch.object(evaluation.subprocess, "check_output",
                                      return_value="137.25\n") as query:
                        self.assertEqual(evaluation.query_power(0), 137.25)
                properties.assert_called_once_with(0)
                query.assert_called_once_with(
                    ["nvidia-smi", "-i", "GPU-physical-device", "--query-gpu=power.draw",
                     "--format=csv,noheader,nounits"],
                    text=True, timeout=5,
                )

    def test_monitor_failure_records_error_and_terminates_process(self):
        marker = Mock()
        failure = RuntimeError("monitor unavailable")
        with patch.object(evaluation, "query_power", side_effect=[120.0, failure]):
            watchdog = evaluation.PowerWatchdog(0, 400.0, 0.25, marker)
            watchdog._stop.wait = Mock(return_value=False)
            with patch.object(evaluation.os, "getpid", return_value=4321):
                with patch.object(evaluation.os, "kill") as kill:
                    watchdog._run()
        marker.write_text.assert_called_once()
        self.assertEqual(json.loads(marker.write_text.call_args.args[0]),
                         {"event": "power_monitor_error", "error": repr(failure)})
        self.assertEqual(marker.write_text.call_args.kwargs, {"encoding": "utf-8"})
        kill.assert_called_once_with(4321, signal.SIGTERM)


if __name__ == "__main__":
    unittest.main()
