#!/usr/bin/env python3
"""Usage: tools/bench-diff.py <before.jsonl> <after.jsonl>

Compares two files of rows written by `lpl-benchmark --json PATH`, row by row. Rows pair by
their label, and a label found in one file only is listed apart. The fields of a row are
listed by `lpl-benchmark --help`.

Each file must be one run: every line a row of schema 1, every label once, one commit, build,
compiler and machine class, and a reason beside every missing energy figure. A file that is
not is refused, with its line and what is wrong.

Prints where each file comes from, warns on stderr when the two differ in machine class, build
or compiler, since their times are then not comparable, then one line per label: the two
medians, after over before, the two coefficients of variation, and the two energies per
repetition or the reason each is missing.

Exit status: 0 when both files are read and compared, 1 when a file is refused, 2 on a usage
error.
"""

import json
import sys

SCHEMA = 1
RUN_FIELDS = ("commit", "build", "compiler", "machine_class")
ROW_FIELDS = ("schema", "label", "median_ns", "cv_percent", "min_ns", "p99_ns", "n",
              "energy_uj_per_rep", "energy_absent_reason") + RUN_FIELDS


class Refused(Exception):
    """A file that is not one run of lpl-benchmark rows, with where and why."""


def refusal_of_row(row, run, labels):
    """Says what is wrong with a row read after `labels`, within the run `run`, or None."""
    if not isinstance(row, dict):
        return "a row is a JSON object"
    missing = [field for field in ROW_FIELDS if field not in row]
    if missing:
        return "missing " + ", ".join(missing)
    if row["schema"] != SCHEMA:
        return f"schema {row['schema']!r}, and this script reads schema {SCHEMA}"
    if row["label"] in labels:
        return f"the label {row['label']!r} is already on line {labels[row['label']]}"
    if row["energy_uj_per_rep"] is None and row["energy_absent_reason"] is None:
        return "energy_uj_per_rep is null and energy_absent_reason does not say why"
    for field in RUN_FIELDS:
        if run is not None and row[field] != run[field]:
            return f"{field} is {row[field]!r}, where line 1 has {run[field]!r}: a file holds one run"
    return None


def reject_constant(name):
    """Refuses NaN and Infinity, which Python reads but JSON does not have."""
    raise ValueError(f"{name} is not JSON")


def read_run(path):
    """Reads one file of rows: the fields of its run, and its rows by label."""
    rows = {}
    labels = {}
    run = None
    try:
        with open(path, encoding="utf-8") as file:
            lines = list(file)
    except OSError as error:
        raise Refused(f"{path}: {error.strerror}") from error
    for number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line, parse_constant=reject_constant)
        except json.JSONDecodeError as error:
            raise Refused(f"{path}:{number}: not a JSON row: {error.msg}") from error
        except ValueError as error:
            raise Refused(f"{path}:{number}: not a JSON row: {error}") from error
        refusal = refusal_of_row(row, run, labels)
        if refusal is not None:
            raise Refused(f"{path}:{number}: {refusal}")
        if run is None:
            run = {field: row[field] for field in RUN_FIELDS}
        labels[row["label"]] = number
        rows[row["label"]] = row
    if run is None:
        raise Refused(f"{path}: no row")
    return run, rows


def duration(nanoseconds):
    """A duration with a unit and four significant figures, as lpl-benchmark prints it."""
    if nanoseconds is None:
        return "null"
    for limit, scale, unit in ((1e3, 1.0, "ns"), (1e6, 1e3, "us"), (1e9, 1e6, "ms")):
        if nanoseconds < limit:
            return f"{nanoseconds / scale:.4g} {unit}"
    return f"{nanoseconds / 1e9:.4g} s"


def percent(value):
    """A percentage with one decimal, or null."""
    return "null" if value is None else f"{value:.1f}%"


def energy(row):
    """The energy of one repetition, or the reason there is none."""
    if row["energy_uj_per_rep"] is None:
        return f"({row['energy_absent_reason']})"
    return f"{row['energy_uj_per_rep']:.4g} uJ"


def ratio(before, after):
    """After over before, when both are positive numbers."""
    if before is None or after is None or before <= 0:
        return "-"
    return f"{after / before:.3f}"


def compare(before_path, after_path):
    """Prints the comparison of two runs; raises Refused on a file that is not one run."""
    before_run, before_rows = read_run(before_path)
    after_run, after_rows = read_run(after_path)

    for name, path, run, rows in (("before", before_path, before_run, before_rows),
                                  ("after", after_path, after_run, after_rows)):
        print(f"{name:<6}  {run['commit']}  {run['build']}  {run['compiler']}  {run['machine_class']}"
              f"  ({len(rows)} rows, {path})")
    for field in ("machine_class", "build", "compiler"):
        if before_run[field] != after_run[field]:
            print(f"bench-diff: warning: {field} differs, {before_run[field]!r} then {after_run[field]!r}: "
                  "the times are not comparable", file=sys.stderr)

    paired = [label for label in before_rows if label in after_rows]
    width = max([len("label")] + [len(label) for label in paired])
    print()
    print(f"{'label':<{width}}  {'before':>10}  {'after':>10}  {'after/before':>12}  {'CV before':>9}  "
          f"{'CV after':>9}  {'energy before':>16}  {'energy after':>16}")
    for label in paired:
        old, new = before_rows[label], after_rows[label]
        print(f"{label:<{width}}  {duration(old['median_ns']):>10}  {duration(new['median_ns']):>10}  "
              f"{ratio(old['median_ns'], new['median_ns']):>12}  {percent(old['cv_percent']):>9}  "
              f"{percent(new['cv_percent']):>9}  {energy(old):>16}  {energy(new):>16}")
    for name, rows, other in (("before", before_rows, after_rows), ("after", after_rows, before_rows)):
        for label in rows:
            if label not in other:
                print(f"only in {name}: {label}")


def main(arguments):
    if len(arguments) == 1 and arguments[0] in ("-h", "--help"):
        print(__doc__, end="")
        return 0
    if len(arguments) != 2:
        print(f"bench-diff: expected two files, got {len(arguments)}\n{__doc__.splitlines()[0]}", file=sys.stderr)
        return 2
    try:
        compare(arguments[0], arguments[1])
    except Refused as refusal:
        print(f"bench-diff: {refusal}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
