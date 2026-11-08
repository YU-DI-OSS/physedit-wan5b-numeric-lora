import importlib.util, pathlib, tempfile, unittest

ROOT = pathlib.Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("phy", ROOT / "examples/wanvideo/model_training/phydataset_training.py")
phy = importlib.util.module_from_spec(spec); spec.loader.exec_module(phy)

class SamplingAndBoundaryTests(unittest.TestCase):
    def test_uniform_ids_include_ends(self):
        # Mirrors LoadVideo integer formula without importing optional image deps.
        ids = [round(i * 100 / 48) for i in range(49)]
        self.assertEqual((ids[0], ids[-1], len(ids)), (0, 100, 49))
        self.assertEqual(ids, sorted(ids))
    def test_short_video_may_repeat_but_includes_ends(self):
        ids = [round(i * 7 / 48) for i in range(49)]
        self.assertEqual((ids[0], ids[-1]), (0, 7))
    def test_high_low_boundaries_override_cache(self):
        cached = {"min_timestep_boundary": .7, "max_timestep_boundary": .9}
        self.assertEqual(phy.enforce_expert_boundary(cached, 0., .358, "high")["max_timestep_boundary"], .358)
        self.assertEqual(phy.enforce_expert_boundary(cached, .358, 1., "low")["min_timestep_boundary"], .358)
    def test_wrong_expert_interval_rejected(self):
        with self.assertRaises(ValueError): phy.enforce_expert_boundary({}, 0., 1., "high")
    def test_incomplete_checkpoint_rejected_before_torch_load(self):
        class FakeTorch: pass
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as d:
            with self.assertRaisesRegex(ValueError, "incomplete"):
                phy.FullStateCheckpoint.load(d, expected_fingerprint={}, optimizer=None,
                                             scheduler=None, scaler=None, torch_module=FakeTorch())
    def test_sampler_exact_cursor(self):
        class Generator:
            def manual_seed(self, seed): self.seed = seed
        class FakeTorch:
            @staticmethod
            def Generator(): return Generator()
            @staticmethod
            def randperm(n, generator):
                class P:
                    def tolist(self): return list(range(n - 1, -1, -1))
                return P()
        sampler = phy.ResumableRandomSampler(list(range(6)), FakeTorch, seed=7)
        it = iter(sampler)
        self.assertEqual([next(it), next(it)], [5, 4])
        state = sampler.state_dict()
        resumed = phy.ResumableRandomSampler(list(range(6)), FakeTorch)
        resumed.load_state_dict(state)
        self.assertEqual(list(resumed), [3, 2, 1, 0])
    def test_sampler_rejects_dataset_change(self):
        class FakeTorch: pass
        sampler = phy.ResumableRandomSampler([1, 2], FakeTorch)
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            sampler.load_state_dict({"seed": 0, "epoch": 0, "position": 0, "dataset_length": 3})
    def test_sampler_mid_epoch_does_not_advance(self):
        class FakeTorch: pass
        sampler = phy.ResumableRandomSampler([1, 2, 3], FakeTorch, epoch=4, position=2)
        self.assertFalse(sampler.advance_if_exhausted())
        self.assertEqual((sampler.epoch, sampler.position), (4, 2))
    def test_sampler_exact_tail_advances_once(self):
        class FakeTorch: pass
        sampler = phy.ResumableRandomSampler([1, 2, 3], FakeTorch, epoch=4, position=3)
        self.assertTrue(sampler.advance_if_exhausted())
        self.assertEqual((sampler.epoch, sampler.position), (5, 0))
        self.assertFalse(sampler.advance_if_exhausted())
        self.assertEqual((sampler.epoch, sampler.position), (5, 0))
    def test_sampler_consecutive_epoch_orders(self):
        class Generator:
            def manual_seed(self, seed): self.seed = seed
        class FakeTorch:
            @staticmethod
            def Generator(): return Generator()
            @staticmethod
            def randperm(n, generator):
                class P:
                    def tolist(self): return list(range(n))
                return P()
        sampler = phy.ResumableRandomSampler([0, 1], FakeTorch, seed=9)
        self.assertEqual(list(sampler), [0, 1])
        self.assertTrue(sampler.advance_if_exhausted())
        self.assertEqual(list(sampler), [0, 1])
        self.assertEqual((sampler.epoch, sampler.position), (1, 2))

if __name__ == "__main__": unittest.main()
