# Samedevice

[中文](#中文) | [English](#english)

---

## 中文

ComfyUI 自定义节点，在启动时用 monkeypatch 修复 ComfyUI 本体里的 device 不一致 bug，不改动 ComfyUI 源码，也不注册任何新节点。

### 解决的问题

以 `--gpu-only` 启动 ComfyUI 时，使用 **VAE Encode (for Inpainting)**（`VAEEncodeForInpaint`）且 `grow_mask_by > 0` 会报错：

```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

(depending on the torch version / 视 torch 版本也可能是: `RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same`)

原因：`nodes.py` 里生成膨胀卷积核用的是 `torch.ones((1, 1, grow_mask_by, grow_mask_by))`，没有指定 device，核在 CPU 上，而 mask 在 GPU 上，`conv2d` 失败。截至 ComfyUI 0.39.0 依然存在。

Samedevice 把这一行换成：

```python
kernel_tensor = torch.ones((1, 1, grow_mask_by, grow_mask_by), device=mask.device, dtype=mask.dtype)
```

这个写法在 CPU 上同样正确，所以无论是否 `--gpu-only` 都会打补丁。

### 原理

- **最小替换**：启动时用 `inspect.getsource` 读出上游 *当前* 的方法源码，只替换出 bug 的那一行，用原模块的 globals 重新编译，再赋回原来的类。方法里其它地方的上游改动全部原样保留。
- **匹配不上就退出**：替换前先检查 bug 原句在方法里是否恰好出现 1 次。不是 1 次（上游已修复或改了这段代码）就用 `logging.critical` 写清是哪个类、哪个方法、找到几次，然后 `sys.exit(1)`。服务在后台长期运行，宁可起不来，也不要补丁悄悄失效。
- **全部检查通过才打补丁**：补丁写在 `__init__.py` 的 `PATCHES` 表里（模块 / 类 / 方法 / bug 原句 / 修复句）。所有项都检查通过后才统一替换，不会留下打了一半的状态。

成功时日志里每个补丁一行：

```
[Samedevice] patched nodes.VAEEncodeForInpaint.encode: ...
```

### 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/sketchain/Samedevice.git
```

或者用 Docker bind mount：

```bash
docker run ... -v /path/to/Samedevice:/opt/ComfyUI/custom_nodes/Samedevice ...
```

没有额外依赖。

### 上游修复之后

上游修好这个 bug 后，特征句匹配不到，ComfyUI **会在启动时退出**（退出码 1），日志里有 `[Samedevice]` 开头的 CRITICAL 说明。届时删掉本节点，或者在 `PATCHES` 里删掉对应的那一项即可。

### 测试

```bash
COMFYUI_PATH=/path/to/ComfyUI python -m unittest discover -s tests -v
```

需要 ComfyUI 的 Python 依赖（CPU 版 torch 即可）。

---

## English

A ComfyUI custom node that fixes device-mismatch bugs in ComfyUI itself by monkeypatching at startup. It does not modify ComfyUI's source and registers no nodes.

### Problem

With ComfyUI started as `--gpu-only`, **VAE Encode (for Inpainting)** (`VAEEncodeForInpaint`) with `grow_mask_by > 0` fails with:

```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

(depending on the torch version / 视 torch 版本也可能是: `RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same`)

`nodes.py` builds the mask dilation kernel with `torch.ones((1, 1, grow_mask_by, grow_mask_by))` and no device, so the kernel is on CPU while the mask is on the GPU and `conv2d` fails. Still present in ComfyUI 0.39.0.

Samedevice replaces that one line with:

```python
kernel_tensor = torch.ones((1, 1, grow_mask_by, grow_mask_by), device=mask.device, dtype=mask.dtype)
```

This is just as correct on CPU, so the patch is applied unconditionally.

### How it works

- **Minimal replacement**: at startup it reads the upstream method's *current* source with `inspect.getsource`, replaces only the buggy line, recompiles it with the original module's globals and assigns it back to the class. Any other upstream changes to the method are kept.
- **Exit if it doesn't match**: the bug line must appear exactly once in the method. Otherwise (upstream fixed or reworked it) it logs a `logging.critical` naming the class, method and match count, then calls `sys.exit(1)`. For a long-running service, failing to start beats a patch that silently stops working.
- **All or nothing**: patches live in the `PATCHES` table in `__init__.py` (module / class / method / bug line / fix line). Every entry is checked before any is applied, so you never end up half-patched.

On success, one log line per patch:

```
[Samedevice] patched nodes.VAEEncodeForInpaint.encode: ...
```

### Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/sketchain/Samedevice.git
```

Or bind-mount it with Docker:

```bash
docker run ... -v /path/to/Samedevice:/opt/ComfyUI/custom_nodes/Samedevice ...
```

No extra dependencies.

### Once upstream fixes it

When upstream fixes the bug, the bug line no longer matches and ComfyUI **exits at startup** (exit code 1) with a CRITICAL log line starting with `[Samedevice]`. Then remove this node, or delete the matching entry from `PATCHES`.

### Tests

```bash
COMFYUI_PATH=/path/to/ComfyUI python -m unittest discover -s tests -v
```

Needs ComfyUI's Python dependencies (CPU torch is enough).
