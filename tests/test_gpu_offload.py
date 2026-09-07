import unittest
from unittest.mock import patch

from lllm2 import defaults
from lllm2.settings import Settings, launch_args, launch_environment


FLAGS = ['--model', '--host', '--port', '--ctx-size', '--parallel', '--device',
         '--gpu-layers', '--jinja', '--fit', '--fit-target']
PROBE = dict(path='/llama-server', flags=FLAGS, devices=['CUDA0'], help='', error=None)
META = dict(error=None, context=262144, template='', mtp=True)


class GpuOffloadTests(unittest.TestCase):
    def test_settings_accept_auto_and_preserve_manual_counts(self):
        for value in (None, '', 0, 12, 999):
            with self.subTest(value=value):
                parsed = Settings.parse({'gpu_layers': value})
                self.assertEqual(parsed.gpu_layers, None if value == '' else value)
                self.assertEqual(Settings.parse(parsed.dict()), parsed)
        for value in (-1, 1000, True, 1.5, '12'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'GPU layers'):
                Settings.parse({'gpu_layers': value})
        self.assertIsNone(Settings().gpu_layers)

    def test_auto_fits_placement_without_changing_context(self):
        with patch('lllm2.settings.probe', return_value=PROBE), \
                patch('lllm2.settings.metadata', return_value=META), \
                patch('lllm2.settings.capabilities', return_value={}):
            for layers in (None, 0, 12, 999):
                with self.subTest(layers=layers):
                    args = launch_args(Settings(model='/model.gguf', gpu_layers=layers,
                                                context=32768, slots=2), 1920)
                    self.assertEqual(args[args.index('--ctx-size') + 1], '32768')
                    self.assertEqual(args[args.index('--parallel') + 1], '2')
                    if layers is None:
                        self.assertNotIn('--gpu-layers', args)
                        self.assertEqual(args[args.index('--fit') + 1], 'on')
                        self.assertEqual(args[args.index('--fit-target') + 1], '1024')
                    else:
                        self.assertNotIn('--fit', args)
                        self.assertEqual(args[args.index('--gpu-layers') + 1], str(layers))

    def test_old_engine_requires_manual_layers(self):
        with patch('lllm2.settings.probe', return_value={**PROBE, 'flags': FLAGS[:-2]}), \
                patch('lllm2.settings.metadata', return_value=META), \
                patch('lllm2.settings.capabilities', return_value={}):
            with self.assertRaisesRegex(ValueError, 'Automatic GPU layers require'):
                launch_args(Settings(model='/model.gguf'), 1920)
            self.assertIn('--gpu-layers', launch_args(Settings(model='/model.gguf', gpu_layers=5), 1920))

    def test_auto_clears_inherited_layer_override_only_in_child(self):
        parent = {'LLAMA_ARG_N_GPU_LAYERS': '999', 'PRESERVE': 'yes'}
        with patch('lllm2.settings.engine_environment', side_effect=lambda _: dict(parent)):
            self.assertEqual(launch_environment(Settings()), {'PRESERVE': 'yes'})
            self.assertEqual(launch_environment(Settings(gpu_layers=12)), parent)
        self.assertEqual(parent['LLAMA_ARG_N_GPU_LAYERS'], '999')

    def inherited(self, total_mib, fit=True):
        entry = dict(size_gb=18.21, max_ctx=262144, kv_kib_per_token=20,
                     full_attention_layers=10)
        caps = {name: dict(status='available') for name in ('flash', 'cache', 'draft-mtp')}
        with patch.object(defaults, 'probe', return_value=PROBE if fit else {**PROBE, 'flags': FLAGS[:-2]}), \
                patch.object(defaults, 'metadata', return_value=META), \
                patch.object(defaults, 'capabilities', return_value=caps), \
                patch.object(defaults, 'catalogue_entry', return_value=entry), \
                patch.object(defaults, 'hardware', return_value={'gpus': [dict(total_mib=total_mib, uuid='gpu')]}), \
                patch.object(defaults, 'command', return_value=(0, '512')):
            return defaults.inherited_defaults(Settings(model='/model.gguf', engine='/llama-server'))

    def test_8gb_gets_partial_offload_starting_point(self):
        result = self.inherited(8188)
        settings = Settings.parse(result['settings'])
        self.assertIsNone(settings.gpu_layers)
        self.assertEqual((settings.context, settings.slots, settings.speculation), (32768, 1, 'none'))
        self.assertIn('system RAM', ' '.join(result['notes']))

    def test_24gb_retains_inherited_context_plan(self):
        result = self.inherited(24576)
        settings = Settings.parse(result['settings'])
        self.assertIsNone(settings.gpu_layers)
        self.assertEqual(settings.speculation, 'draft-mtp')
        self.assertGreater(settings.context, 4096)
        self.assertNotIn('full-GPU context estimate does not fit', ' '.join(result['notes']))

    def test_old_engine_defaults_are_explicit_and_qualified(self):
        result = self.inherited(8188, fit=False)
        self.assertEqual(result['settings']['gpu_layers'], 999)
        self.assertIn('lacks automatic memory fitting', ' '.join(result['notes']))

    def test_measured_profile_retains_its_tested_placement(self):
        measured = dict(settings=Settings(gpu_layers=999).dict(), source='measured')
        with patch.object(defaults, 'measured_defaults', return_value=(measured, [])), \
                patch.object(defaults, 'inherited_defaults') as inherited:
            self.assertEqual(defaults.starting_defaults(Settings()), measured)
            inherited.assert_not_called()


class A400CatalogueDefaultsTests(unittest.TestCase):
    flags = FLAGS + ['--batch-size', '--ubatch-size', '--cache-ram', '--ctx-checkpoints',
                     '--flash-attn', '--cache-type-k', '--cache-type-v']

    def resolve(self, *, total_mib=4096, available_gib=28, flags=None, cache=True, ceiling=262144):
        entry = next(m for m in defaults.CATALOG if m['id'] == 'qwen3.5-35b-a3b')
        selection = Settings(model='/models/' + entry['name'] + '/' + entry['file'],
                             engine='/llama-server', context=65536, slots=4,
                             gpu_layers=999, speculation='draft-mtp', batch_size=2048)
        p = dict(PROBE, flags=self.flags if flags is None else flags)
        m = dict(META, context=ceiling)
        caps = {name: dict(status='available' if cache else 'unsupported', reason='test engine')
                for name in ('flash', 'cache', 'draft-mtp')}
        host = dict(gpus=[dict(name='NVIDIA RTX A400', total_mib=total_mib, uuid='gpu')],
                    ram_gib=32, ram=dict(available_gib=available_gib))
        with patch.object(defaults, 'probe', return_value=p), \
                patch.object(defaults, 'metadata', return_value=m), \
                patch.object(defaults, 'capabilities', return_value=caps), \
                patch.object(defaults, 'hardware', return_value=host), \
                patch.object(defaults, 'command', return_value=(0, '512')), \
                patch.object(defaults, 'measured_defaults', return_value=(None, [])):
            result = defaults.starting_defaults(selection)
        s = Settings.parse(result['settings'])
        with patch('lllm2.settings.probe', return_value=p), \
                patch('lllm2.settings.metadata', return_value=m), \
                patch('lllm2.settings.capabilities', return_value=caps):
            args = launch_args(s, 1920)
        return s, result, args

    def test_a400_launch_uses_small_context_and_automatic_offload(self):
        s, result, args = self.resolve()
        self.assertEqual((s.context, s.slots, s.speculation), (4096, 1, 'none'))
        self.assertEqual((s.batch_size, s.ubatch_size), (128, 64))
        self.assertEqual((s.cache_ram_mib, s.context_checkpoints), (0, 0))
        self.assertEqual((s.flash, s.cache), ('on', 'q8_0'))
        self.assertIsNone(s.gpu_layers)
        for flag, value in [('--ctx-size', '4096'), ('--parallel', '1'), ('--fit-target', '1024'),
                            ('--batch-size', '128'), ('--ubatch-size', '64'),
                            ('--cache-ram', '0'), ('--ctx-checkpoints', '0')]:
            self.assertEqual(args[args.index(flag) + 1], value)
        self.assertIn('estimates', result['source'])

    def test_old_engine_keeps_weights_on_cpu_and_omits_unsupported_flags(self):
        s, result, args = self.resolve(flags=FLAGS[:-2], cache=False)
        self.assertEqual((s.gpu_layers, s.context, s.slots), (0, 4096, 1))
        self.assertEqual(args[args.index('--gpu-layers') + 1], '0')
        self.assertNotIn('--batch-size', args)
        self.assertNotIn('--fit', args)
        self.assertNotIn('full offload (999)', ' '.join(result['notes']))

    def test_small_context_applies_without_quantized_cache_support(self):
        s, _, _ = self.resolve(cache=False)
        self.assertEqual((s.context, s.cache, s.speculation), (4096, 'f16', 'none'))

    def test_checkpoint_ceiling_and_available_ram_are_respected(self):
        s, result, _ = self.resolve(ceiling=2048, available_gib=16)
        self.assertEqual(s.context, 2048)
        self.assertIn('Only 16 GiB system RAM is available', ' '.join(result['notes']))

    def test_larger_gpu_retains_existing_offload_defaults(self):
        s, result, _ = self.resolve(total_mib=8192, cache=False)
        self.assertEqual(result['source'], defaults.SOURCE)
        self.assertIsNone(s.batch_size)
        self.assertIsNone(s.cache_ram_mib)
