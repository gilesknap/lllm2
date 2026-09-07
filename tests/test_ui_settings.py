"""Settings recovery uses temporary storage; never open the workstation database."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lllm2.app import App
from lllm2.discovery import host_memory
from lllm2.settings import Settings
from lllm2.store import Store


class MemoryTests(unittest.TestCase):
    def test_reclaimable_cache_is_available(self):
        with patch('pathlib.Path.read_text', return_value='MemTotal: 8388608 kB\nMemFree: 1048576 kB\nMemAvailable: 6291456 kB\nCached: 5242880 kB\n'):
            self.assertEqual(host_memory(), dict(total_gib=8, used_gib=2, available_gib=6))

    def test_missing_invalid_or_unreadable_memory_is_unknown(self):
        for data in ('MemTotal: 8 kB\n', 'MemTotal: 8 kB\nMemAvailable: 9 kB',
                     'MemTotal: nonsense\nMemAvailable: 0 kB', ''):
            with self.subTest(data=data), patch('pathlib.Path.read_text', return_value=data):
                self.assertIsNone(host_memory()['used_gib'])
        with patch('pathlib.Path.read_text', side_effect=OSError):
            self.assertIsNone(host_memory()['total_gib'])


class SettingsRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        with patch('lllm2.config.STATE_DIR', Path(self.temp.name)):
            self.app = App.__new__(App)
            self.app.store = Store()
        self.addCleanup(self.app.store.db.close)
        self.settings = Settings(model='/models/example.gguf', engine='/engine', gpu_layers=0)
        self.result = dict(id='measured', status='complete', settings=self.settings.dict(),
                           samples=[dict(workload='generate')], probes=[], recommended_context=4096)
        self.app.store.put('result', self.result['id'], self.result)

    def test_manual_save_preserves_recoverable_experiment_and_recommendation(self):
        original = self.app.store.get('result', 'measured')
        with patch('lllm2.app.launch_args'):
            self.app.action('/api/default/save', dict(result_id='measured'))
            provenance = self.app.store.get('default-evidence', self.app.default_key(self.settings))
            self.assertEqual(provenance['kind'], 'benchmark')
            self.app.action('/api/default/save', dict(settings=Settings(**{**self.settings.dict(), 'context': 8192}).dict()))
        saved = self.app.store.get('default', self.app.default_key(self.settings))
        preview = self.app.action('/api/result/preview', dict(result_id='measured'))
        self.assertEqual(preview['settings']['gpu_layers'], 0)
        self.assertEqual(preview['evidence']['result_id'], 'measured')
        self.assertEqual(self.app.store.get('result', 'measured'), original)
        self.assertEqual(self.app.store.get('default', self.app.default_key(self.settings)), saved)
        recommended = dict(settings=Settings(gpu_layers=None).dict(), source='Estimated starting settings')
        with patch('lllm2.app.starting_defaults', return_value=recommended):
            self.assertEqual(self.app.action('/api/default/resolve', dict(settings=self.settings.dict(), source='built-in')), recommended)
        restored = self.app.action('/api/default/resolve', dict(settings=self.settings.dict(), source='saved'))
        self.assertEqual(restored['settings']['context'], 8192)
        self.assertEqual(restored['settings']['gpu_layers'], 0)
        self.assertEqual(self.app.store.get('default', self.app.default_key(self.settings)), saved)

    def test_ineligible_experiments_cannot_be_loaded_or_saved(self):
        for change in (dict(measurement_mode='warm-conversation'), dict(quality_status='failed'), dict(status='failed')):
            self.app.store.put('result', 'measured', {**self.result, **change})
            for route in ('/api/result/preview', '/api/default/save'):
                with self.subTest(change=change, route=route), self.assertRaises(ValueError):
                    self.app.action(route, dict(result_id='measured'))
