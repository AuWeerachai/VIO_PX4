"""Offline tests: no SSH, hardware, or flight configuration changes."""
import io
from pathlib import Path
from queue import Queue
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from flight_capture import command_reader
from floor_geometry import CalibrationError
from transfer_results import offer_transfer, transfer_and_plot


class TransferTests(unittest.TestCase):
    def test_stop_releases_input_for_followup_prompt(self):
        commands = Queue()
        stream = io.StringIO('START\nSTOP\ny\nau@computer\n')
        with patch('sys.stdin', stream):
            command_reader(commands)
        self.assertEqual([commands.get_nowait(), commands.get_nowait()], ['START', 'STOP'])
        self.assertEqual(stream.readline(), 'y\n')

    def test_decline_does_not_connect(self):
        with patch('builtins.input', return_value='n'), patch('transfer_results.transfer_and_plot') as transfer:
            offer_transfer(Path('/tmp/saved'))
            transfer.assert_not_called()

    def test_yes_uses_computer_target(self):
        with patch('builtins.input', side_effect=['y', 'au@computer']), patch('transfer_results.transfer_and_plot') as transfer:
            offer_transfer(Path('/tmp/saved'))
            transfer.assert_called_once_with(Path('/tmp/saved'), 'au@computer')

    def test_copy_precedes_plot_and_failure_stops_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder/'capture.json').touch()
            for fail_index in (None, 0, 1, 2):
                commands = []
                def run(cmd, **kwargs):
                    index = len(commands)
                    commands.append(cmd)
                    return SimpleNamespace(returncode=int(index == fail_index))
                with patch('transfer_results.subprocess.run', side_effect=run):
                    if fail_index is None:
                        transfer_and_plot(folder, 'au@computer')
                    else:
                        with self.assertRaises(CalibrationError):
                            transfer_and_plot(folder, 'au@computer')
                self.assertEqual(len(commands), 3 if fail_index is None else fail_index+1)
                if len(commands) > 1:
                    self.assertEqual(commands[1][0], 'scp')
                if len(commands) > 2:
                    self.assertIn(' analyze ', commands[2][-1])
                    self.assertIn('--open', commands[2][-1])
                self.assertTrue((folder/'capture.json').exists())

    def test_invalid_target_never_runs_shell(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'capture.json').touch()
            with patch('transfer_results.subprocess.run') as run:
                with self.assertRaises(CalibrationError):
                    transfer_and_plot(tmp, '-oProxyCommand=anything')
                run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
