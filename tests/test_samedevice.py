"""Run with:  COMFYUI_PATH=/path/to/ComfyUI python -m unittest discover -s tests -v

Needs a ComfyUI checkout and its Python dependencies (CPU torch is enough).
"""

import importlib.util
import inspect
import os
import sys
import types
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
COMFYUI_PATH = os.environ.get("COMFYUI_PATH")

if not COMFYUI_PATH:
    raise unittest.SkipTest("set COMFYUI_PATH to a ComfyUI checkout")

sys.path.insert(0, COMFYUI_PATH)
sys.argv = sys.argv[:1] + ["--cpu"]
import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()  # so comfy.cli_args picks up --cpu

import torch  # noqa: E402
import nodes  # noqa: E402

ORIGINAL_ENCODE = nodes.VAEEncodeForInpaint.__dict__["encode"]


def load_samedevice():
    spec = importlib.util.spec_from_file_location("Samedevice", os.path.join(REPO, "__init__.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules["Samedevice"] = module
    spec.loader.exec_module(module)
    return module


samedevice = load_samedevice()


class StrictDevices(torch.overrides.TorchFunctionMode):
    """Make every torch op reject mixed devices, the way CUDA kernels do (meta kernels don't check)."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        devices = set()
        stack = list(args) + list(kwargs.values())
        while stack:
            a = stack.pop()
            if isinstance(a, torch.Tensor):
                devices.add(a.device)
            elif isinstance(a, (list, tuple)):
                stack.extend(a)
        if len(devices) > 1:
            raise RuntimeError(f"Expected all tensors to be on the same device, but found {sorted(map(str, devices))}")
        return func(*args, **kwargs)


class FakeVAE:
    def spacial_compression_encode(self):
        return 8

    def encode(self, pixels):
        return pixels[:, ::8, ::8, :]


def run_encode(func, device, grow=6):
    pixels = torch.rand((1, 64, 64, 3), device=device)
    mask = torch.zeros((1, 64, 64), device=device)
    if device != "meta":
        mask[:, 20:40, 20:40] = 1.0
    return func(nodes.VAEEncodeForInpaint(), FakeVAE(), pixels, mask, grow)


class RealComfyUI(unittest.TestCase):
    def test_upstream_has_bug_line_exactly_once(self):
        src = inspect.getsource(ORIGINAL_ENCODE)
        bug = samedevice.PATCHES[0].bug
        self.assertEqual(sum(line.strip() == bug for line in src.splitlines()), 1)

    def test_encode_is_patched(self):
        new = nodes.VAEEncodeForInpaint.__dict__["encode"]
        self.assertIsNot(new, ORIGINAL_ENCODE)
        self.assertTrue(getattr(new, samedevice.PATCHED_MARK))
        self.assertEqual(new.__qualname__, "VAEEncodeForInpaint.encode")
        self.assertEqual(new.__module__, "nodes")
        self.assertIs(new.__globals__, vars(nodes))

    def test_cpu_result_identical_to_upstream(self):
        new = nodes.VAEEncodeForInpaint.__dict__["encode"]
        for grow in (0, 1, 6, 13):
            torch.manual_seed(0)
            (a,) = run_encode(ORIGINAL_ENCODE, "cpu", grow)
            torch.manual_seed(0)
            (b,) = run_encode(new, "cpu", grow)
            self.assertTrue(torch.equal(a["noise_mask"], b["noise_mask"]), grow)
            self.assertTrue(torch.equal(a["samples"], b["samples"]), grow)
        self.assertGreater(b["noise_mask"].sum().item(), 0)

    def test_non_cpu_device_mismatch_fixed(self):
        # The meta device stands in for CUDA: the upstream kernel lands on CPU and conv2d refuses it.
        with StrictDevices(), self.assertRaisesRegex(RuntimeError, "same device"):
            run_encode(ORIGINAL_ENCODE, "meta")
        with StrictDevices():
            (out,) = run_encode(nodes.VAEEncodeForInpaint.__dict__["encode"], "meta")
        self.assertEqual(out["noise_mask"].device.type, "meta")
        self.assertEqual(tuple(out["noise_mask"].shape), (1, 1, 64, 64))

    def test_float64_mask_dtype_fixed(self):
        def run(func):
            pixels = torch.rand((1, 64, 64, 3), dtype=torch.float64)
            mask = torch.zeros((1, 64, 64), dtype=torch.float64)
            return func(nodes.VAEEncodeForInpaint(), FakeVAE(), pixels, mask, 6)

        with self.assertRaises(RuntimeError):
            run(ORIGINAL_ENCODE)
        (out,) = run(nodes.VAEEncodeForInpaint.__dict__["encode"])
        self.assertEqual(out["noise_mask"].dtype, torch.float64)

    def test_reload_is_idempotent(self):
        before = nodes.VAEEncodeForInpaint.__dict__["encode"]
        with self.assertLogs(level="INFO") as logs:
            load_samedevice()
        self.assertIs(nodes.VAEEncodeForInpaint.__dict__["encode"], before)
        self.assertIn("already patched", "\n".join(logs.output))


FAKE_SRC = '''
class Target:
    def method(self, x):
        y = x + 1
        return y

    @staticmethod
    def smethod(x):
        y = x + 1
        return y

class Other:
    def method(self, x):
        y = x + 1
        return y
'''


class FakeModule(unittest.TestCase):
    def setUp(self):
        self.mod = types.ModuleType("samedevice_fake")
        path = os.path.join(HERE, "_fake_src.py")
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))
        with open(path, "w") as f:
            f.write(FAKE_SRC)
        self.mod.__file__ = path
        exec(compile(FAKE_SRC, path, "exec"), vars(self.mod))
        sys.modules["samedevice_fake"] = self.mod
        self.addCleanup(sys.modules.pop, "samedevice_fake")

    def patch(self, cls="Target", method="method", bug="y = x + 1", fix="y = x + 100"):
        return samedevice.Patch("samedevice_fake", cls, method, bug, fix)

    def test_patch_applies(self):
        samedevice.apply_patches([self.patch(), self.patch(method="smethod")])
        self.assertEqual(self.mod.Target().method(1), 101)
        self.assertEqual(self.mod.Target.smethod(1), 101)
        self.assertEqual(self.mod.Other().method(1), 2)

    def test_missing_bug_line_exits(self):
        with self.assertLogs(level="CRITICAL") as logs, self.assertRaises(SystemExit) as cm:
            samedevice.apply_patches([self.patch(bug="y = x + 2")])
        self.assertEqual(cm.exception.code, 1)
        out = "\n".join(logs.output)
        self.assertIn("[Samedevice]", out)
        self.assertIn("class Target, method method", out)
        self.assertIn("found 0 times", out)

    def test_duplicate_bug_line_exits(self):
        src = "    def method(self, x):\n        y = x + 1\n        y = x + 1\n        return y\n"
        with mock.patch.object(samedevice.inspect, "getsourcelines", return_value=([src], 1)), \
                self.assertLogs(level="CRITICAL") as logs, self.assertRaises(SystemExit):
            samedevice.apply_patches([self.patch()])
        self.assertIn("found 2 times", "\n".join(logs.output))

    def test_all_or_nothing(self):
        good = self.patch(cls="Other")
        bad = self.patch(bug="no such line")
        with self.assertLogs(level="CRITICAL"), self.assertRaises(SystemExit):
            samedevice.apply_patches([good, bad])
        self.assertEqual(self.mod.Other().method(1), 2)  # good patch was NOT applied

    def test_unknown_class_or_module_exits(self):
        for p in (self.patch(cls="Nope"), self.patch(method="nope"),
                  samedevice.Patch("no_such_module_xyz", "A", "b", "c", "d")):
            with self.assertLogs(level="CRITICAL"), self.assertRaises(SystemExit):
                samedevice.apply_patches([p])

    def test_real_nodes_with_bug_line_removed_exits(self):
        lines, first = inspect.getsourcelines(ORIGINAL_ENCODE)
        bug = samedevice.PATCHES[0].bug
        fixed_upstream = [l.replace(bug, samedevice.PATCHES[0].fix) for l in lines]
        current = nodes.VAEEncodeForInpaint.__dict__["encode"]
        with mock.patch.object(samedevice.inspect, "getsourcelines", return_value=(fixed_upstream, first)), \
                mock.patch.dict(nodes.VAEEncodeForInpaint.__dict__["encode"].__dict__,
                                {samedevice.PATCHED_MARK: False}), \
                self.assertLogs(level="CRITICAL") as logs, self.assertRaises(SystemExit) as cm:
            samedevice.apply_patches(samedevice.PATCHES)
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("nodes.VAEEncodeForInpaint.encode", "\n".join(logs.output))
        self.assertIs(nodes.VAEEncodeForInpaint.__dict__["encode"], current)


if __name__ == "__main__":
    unittest.main()
