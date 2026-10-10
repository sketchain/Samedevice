"""Samedevice: fix device-mismatch bugs in ComfyUI by patching upstream methods at startup.

Each patch re-reads the upstream method's *current* source, replaces exactly one buggy
line, recompiles it in the original module's globals and puts it back on the class.
Everything else in the method stays as upstream wrote it.

If any bug line is not found exactly once (upstream fixed or reworked that code), the
process exits with status 1 instead of running with a silently ineffective patch.
ComfyUI only catches Exception when loading custom nodes, so SystemExit really stops it.
"""

import ast
import importlib
import inspect
import logging
import sys
import textwrap
from dataclasses import dataclass

PREFIX = "[Samedevice]"
PATCHED_MARK = "__samedevice_patched__"


@dataclass(frozen=True)
class Patch:
    module: str
    cls: str
    method: str
    bug: str  # the exact buggy line, without indentation
    fix: str  # the line that replaces it, without indentation
    why: str = ""


PATCHES = [
    Patch(
        module="nodes",
        cls="VAEEncodeForInpaint",
        method="encode",
        bug="kernel_tensor = torch.ones((1, 1, grow_mask_by, grow_mask_by))",
        fix="kernel_tensor = torch.ones((1, 1, grow_mask_by, grow_mask_by), device=mask.device, dtype=mask.dtype)",
        why="dilation kernel was created on CPU; conv2d fails when mask is on GPU (--gpu-only)",
    ),
]


class PatchError(Exception):
    pass


@dataclass
class _Prepared:
    patch: Patch
    owner: type
    new: object


def _name(p):
    return f"{p.module}.{p.cls}.{p.method}"


def _prepare(p):
    """Check one patch and build the replacement function. Changes nothing. Raises PatchError."""
    if "\n" in p.bug or "\n" in p.fix:
        raise PatchError("bug/fix must be a single line")

    try:
        module = importlib.import_module(p.module)
    except Exception as e:
        raise PatchError(f"cannot import module {p.module!r}: {e!r}")
    owner = getattr(module, p.cls, None)
    if not isinstance(owner, type):
        raise PatchError(f"class {p.cls!r} not found in module {p.module!r}")
    if p.method not in owner.__dict__:
        raise PatchError(f"method {p.method!r} not defined on class {p.cls!r}")

    original = owner.__dict__[p.method]
    func = getattr(original, "__func__", original)  # unwrap staticmethod/classmethod
    if not inspect.isfunction(func):
        raise PatchError(f"{p.cls}.{p.method} is not a plain function ({type(original).__name__})")
    if getattr(func, PATCHED_MARK, False):
        return None  # module loaded twice; already patched
    if func.__code__.co_freevars:
        # e.g. zero-arg super() needs the __class__ cell, which exec at module level can't provide
        raise PatchError(f"{p.cls}.{p.method} uses closure variables {func.__code__.co_freevars}; cannot recompile")

    try:
        source, first_line = inspect.getsourcelines(func)
    except (OSError, TypeError) as e:
        raise PatchError(f"cannot read source of {p.cls}.{p.method}: {e!r}")
    source = textwrap.dedent("".join(source))

    lines = source.splitlines(keepends=True)
    hits = [i for i, line in enumerate(lines) if line.strip() == p.bug]
    if len(hits) != 1:
        raise PatchError(
            f"bug line found {len(hits)} times in {p.cls}.{p.method} (expected exactly 1); "
            f"upstream has probably fixed or changed this code. Bug line: {p.bug!r}"
        )
    i = hits[0]
    indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
    lines[i] = indent + p.fix + "\n"
    new_source = "".join(lines)

    # Keep line numbers aligned with the upstream file so tracebacks point to the right place.
    filename = f"{inspect.getsourcefile(func) or p.module} (patched by Samedevice)"
    try:
        tree = ast.parse(new_source, filename=filename)
        ast.increment_lineno(tree, first_line - 1)
        code = compile(tree, filename, "exec")
    except SyntaxError as e:
        raise PatchError(f"patched source of {p.cls}.{p.method} does not compile: {e}")

    # Build the new function now, so the install step below is just setattr and cannot fail halfway.
    namespace = {}
    try:
        exec(code, vars(module), namespace)
    except Exception as e:
        raise PatchError(f"patched {p.cls}.{p.method} failed to load: {e!r}")
    new = namespace[func.__name__]
    new_func = getattr(new, "__func__", new)
    new_func.__qualname__ = func.__qualname__
    new_func.__dict__.update(func.__dict__)
    setattr(new_func, PATCHED_MARK, True)
    new_func.__samedevice_original__ = func
    if isinstance(original, (staticmethod, classmethod)) and new is new_func:
        new = type(original)(new_func)  # decorator wasn't part of the method's source
    return _Prepared(p, owner, new)


def apply_patches(patches=PATCHES):
    """Check every patch first; only if all pass, install them all. Otherwise exit(1)."""
    prepared, failures = [], []
    for p in patches:
        try:
            prep = _prepare(p)
        except PatchError as e:
            failures.append((p, str(e)))
            continue
        except Exception as e:
            # Any surprise must also stop startup: ComfyUI would swallow an Exception and run unpatched.
            failures.append((p, f"unexpected error: {e!r}"))
            continue
        if prep is None:
            logging.info(f"{PREFIX} {_name(p)} already patched, skipping")
        else:
            prepared.append(prep)

    if failures:
        for p, reason in failures:
            logging.critical(f"{PREFIX} cannot patch {_name(p)} (class {p.cls}, method {p.method}): {reason}")
        logging.critical(
            f"{PREFIX} {len(failures)} of {len(patches)} patch(es) failed the check; nothing was patched. "
            f"If upstream fixed the bug, remove the patch entry from Samedevice/__init__.py "
            f"(or remove the Samedevice node). Exiting."
        )
        sys.exit(1)

    for prep in prepared:
        p = prep.patch
        setattr(prep.owner, p.method, prep.new)
        logging.info(f"{PREFIX} patched {_name(p)}" + (f": {p.why}" if p.why else ""))


apply_patches()

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
