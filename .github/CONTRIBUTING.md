# Contributing to LplPlugin

The flow (issues, branches, pull requests, commit titles, labels) and the C and C++ rules shared by
every Laplace repository are in the
[shared CONTRIBUTING](https://github.com/MasterLaplace/.github/blob/main/.github/CONTRIBUTING.md).
How to build and run the engine is in the [README](../README.md). This page adds what only holds
here.

## Code that runs in the kernel

[LplKernel](https://github.com/MasterLaplace/LplKernel) compiles part of this repository into its
image: the sources listed in its `libengine/arch/i386/make.config`. That code obeys the
[determinism contract](https://github.com/MasterLaplace/LplKernel/blob/main/.github/CONTRIBUTING.md#the-determinism-contract),
and a feature that reaches the kernel ships as a
[slice](https://github.com/MasterLaplace/LplKernel/blob/main/.github/CONTRIBUTING.md#a-slice-one-feature-proven-on-both-targets).
LplKernel finds this repository next to it, at `../LplPlugin`, so there is one checkout of it for
both; its `DEPENDENCIES.lock` names the commit the kernel was tested with, and a change here that the
kernel needs is merged before the kernel's.

- A module meant to be linked into the kernel uses `lpl::pmr` and the `lpl/std/` headers, never the
  `std::` containers, which need a heap the kernel does not have. Host tools (`editor/`, `bench/`,
  `apps/`, `tests/`) use `std::` freely: they never reach the kernel.
- What only exists on a host sits behind `LPL_HAS_RENDERER`, `LPL_HAS_NET` or `LPL_HAS_BCI`, which
  `xmake.lua` defines on the host and the kernel build leaves undefined.
- A `.cpp` added to a module the kernel compiles also goes into LplKernel's build lists: until it
  does, the kernel link fails with `undefined reference`.

## Fixed32

`lpl::math::Fixed32{n}` builds from the raw Q16.16 word: `Fixed32{10}` is 10/65536, not ten. Write
`Fixed32::fromInt(10)`, `Fixed32::one()` or `Fixed32::half()`; reserve the braces for a value that is
already raw.

## Tests

- A test of code the kernel links is an `LPL_TEST(name)` in `tests/<module>/`, under one
  `LPL_TEST_SUITE` per file: no list to update. `test-engine` runs it here, a debug kernel runs the
  same source in ring 0, and every `key=value` it measures must come out the same on both, so a size
  or an address is never measured. A test of what only the kernel compiles goes in
  `tests/<module>/kernel/`, and runs in ring 0 alone. `xmake run test-engine 'relief.*'` runs one
  suite.
- Any other test is a `test-*` target of `tests/xmake.lua`, and it ends on
  `ALL PASS (0 failures, N checks)`. A target that prints no verdict counts as a failure, so a test
  target is declared by the change that fills it.
- A generator or a simulation step earns three tests: it reproduces bit for bit, it changes with its
  seed, and it keeps an invariant stated as a property (the steep erodes more than the flat), not as a
  folded signature. A signature that stays stable proves nothing about what it folds.

## Traps that are not checks yet

- `xmake build` takes one target. Without one it skips the targets declared `set_default(false)`,
  the tests among them, so use `xmake build -a` for everything; `xmake build a b` refuses the second
  name and builds nothing.
- `lpl-mapview` and `lpl-worldforge` are behind `--mapview=y` and `--worldforge=y`, which the
  validation does not turn on (LplKernel #195): build them by hand after touching what they show.
