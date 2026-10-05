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
For the kernel port, change this repository through LplKernel's `LplPlugin/` submodule.

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

- A test is a `test-*` target of `tests/xmake.lua`, and it ends on `ALL PASS (0 failures, N checks)`.
  A target that prints no verdict counts as a failure, so a test target is declared by the change
  that fills it.
- A generator or a simulation step earns three tests: it reproduces bit for bit, it changes with its
  seed, and it keeps an invariant stated as a property (the steep erodes more than the flat), not as a
  folded signature. A signature that stays stable proves nothing about what it folds.

## Traps that are not checks yet

- `xmake` resolves its project from the current directory, and inside LplKernel this repository is a
  nested project: build it with `xmake -P <path to LplPlugin>`.
- `xmake build` takes one target. Without one it skips the targets declared `set_default(false)`,
  the tests among them, so use `xmake build -a` for everything; `xmake build a b` refuses the second
  name and builds nothing.
- `lpl-mapview` and `lpl-worldforge` are behind `--mapview=y` and `--worldforge=y`, which the
  validation does not turn on (LplKernel #195): build them by hand after touching what they show.
