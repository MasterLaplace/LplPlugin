#!/usr/bin/env python3
"""Usage: tools/check-conventions.py [<file>...]
       tools/check-conventions.py --self-test

Refuses C and C++ that breaks the comment and layout rules every Laplace repository shares
(the "C and C++" section of https://github.com/MasterLaplace/.github/blob/main/.github/CONTRIBUTING.md):

  line-comment           a // comment. What it says goes into the Doxygen of the function, member
                         or type, or into a name. `// TODO(#N): ...`, `} // namespace x`,
                         `#endif // GUARD` and the label of a row in a data table are the only //
                         comments allowed.
  banner                 a // rule of dashes or equal signs between sections. A group of
                         declarations is an `@name` block with `@{` and `@}`.
  triple-slash           a ///, //! or /*! comment, or a ///< on a line of its own. Doxygen is
                         written /** */ before what it documents and /**< */ or ///< after a member.
  one-line-function-doc  a function documented on one line: its block is deployed, one tag a line.
  using                  a using that shortens a namespace (`using namespace x;`, `using x::Y;`):
                         the full name is written. A type alias (`using Name = ...;`) is fine.
  header-body            a function body of more than one line in a header: it goes in the .inl
                         next to the header.
  file-block             a /** @file */ block at the top of a .c, .cpp, .inl or .cu file: only
                         headers carry one.

It reads comments and string, character and raw string literals, so a // inside a string is not a
comment, and a digit separator (1'000) is not a character literal. A row label is a // after one
element of a table (a literal, a name or a braced row) and its comma, if any. It reads layout the
way clang-format writes it: a function body is found from a line holding only `{` under a line
that ends a signature, and a one-line doc is a function's when the next line opens a parameter
list before any `=` or `{`. A body or a declaration laid out otherwise is not seen.

tools/check-conventions.declared lists the files allowed to break one rule, each with its reason:
vendored code, generated code, a shim. An entry that matches nothing in a file it names is refused
too, so the list never outlives what it explains.

Without a file it checks every C and C++ source git tracks in this repository.

  --self-test  check that a fixture of each rule is reported at its line, that the allowed forms
               pass (a TODO with its issue, the closing comments of a namespace and of an include
               guard, row labels, comments inside literals, a digit separator, a type alias,
               deployed and trailing Doxygen, an unnamed parameter's /*name*/, a one-line body in
               a header, a longer one in an .inl), that a declared entry silences its rule in its
               file only,
               that a stale or malformed entry is refused, that a file it cannot read fails
               without hiding the others, and that an empty list of files fails

Exit status: 0 when nothing is found, 1 when a rule is broken, when the declared list holds a
stale or malformed entry, or when there is no file to check; 2 on a usage error or when a file
cannot be read (the findings of the others are still printed).
"""

import bisect
import os
import pathlib
import re
import subprocess
import sys
import tempfile

SOURCE_SUFFIXES = (".c", ".cpp", ".inl", ".cu")
HEADER_SUFFIXES = (".h", ".hpp", ".cuh")
KINDS = ("line-comment", "banner", "triple-slash", "one-line-function-doc", "using", "header-body", "file-block")
MESSAGES = {
    "line-comment": "a // comment: say it in the Doxygen of what it describes, or in a name",
    "banner": "a banner between sections: group with an @name block and @{ @}",
    "triple-slash": "Doxygen is /** */ before what it documents and /**< */ or ///< after a member",
    "one-line-function-doc": "a function's doc is a deployed block, one tag a line",
    "using": "a using that shortens a namespace: write the full name",
    "header-body": "a body of more than one line in a header: move it to the .inl",
    "file-block": "only headers carry a @file block: this file starts with its includes",
}
DECLARED = pathlib.Path(__file__).with_name("check-conventions.declared")

TOKEN = re.compile(r'//|/\*|(?:u8|[uUL])?R"(?P<delimiter>[^()\\\s]{0,16})\(|"|\'')
STRING_END = re.compile(r'(?:\\.|[^"\\\n])*"')
CHARACTER_END = re.compile(r"(?:\\.|[^'\\\n])*'")
NUMBER_BEFORE_SEPARATOR = re.compile(r"(?<![\w.])\d[\w']*$")
ALLOWED_LINE_COMMENT = re.compile(r"//\s*TODO\(#\d+\)")
NAMESPACE_CLOSING = re.compile(r"//\s*namespace\b")
TABLE_ROW = re.compile(r"^(?:\{.*\}|\"[^\"]*\"|'[^']*'|[-+\w.']+)\s*,?$")
BANNER = re.compile(r"//\s*(?:[=\-─━]\s*){4,}")
SHORTENING_USING = re.compile(r"^\s*using\s+(?:namespace\s+[\w:]+|[\w:]*::\w+)\s*;")
NOT_A_FUNCTION = re.compile(r"^\s*(?:template\s*<.*>\s*)?(?:struct|class|union|enum|using|typedef|namespace|#)\b")
CONTROL = re.compile(
    r"^\s*(?:if|for|while|switch|else|do|try|catch)\b|\bnamespace\b|\b(?:struct|class|union|enum)\b|"
    r"=\s*\[|^\s*#|\breturn\b"
)
GROUP_MARKER = re.compile(r"/\*\*\s*@(?:\}|\{|name\b)")
SIGNATURE_ENDINGS = (")", "const", "noexcept", "override", "final")


def blank(text):
    """The text with every character but the line breaks turned into a space."""
    return re.sub(r"[^\n]", " ", text)


def lex(text):
    """Splits a C or C++ text into its code and its comments.

    Returns the code lines, where comments are spaces and the contents of literals are spaces
    between their quotes, and the comments as (line, column, text) with 1-based lines.
    """
    line_starts = [0] + [match.end() for match in re.finditer(r"\n", text)]
    pieces = []
    comments = []
    position = 0
    while True:
        match = TOKEN.search(text, position)
        if match is None:
            pieces.append(text[position:])
            break
        start = match.start()
        pieces.append(text[position:start])
        token = match.group(0)
        if token in ("//", "/*"):
            end = text.find("\n", start) if token == "//" else text.find("*/", start + 2)
            end = len(text) if end < 0 else end + (0 if token == "//" else 2)
            line = bisect.bisect_right(line_starts, start)
            comments.append((line, start - line_starts[line - 1], text[start:end]))
            pieces.append(blank(text[start:end]))
        elif (
            match.group("delimiter") is not None and start > 0 and (text[start - 1].isalnum() or text[start - 1] == "_")
        ):
            end = start + 1
            pieces.append(text[start])
        elif match.group("delimiter") is not None:
            closing = ")" + match.group("delimiter") + '"'
            end = text.find(closing, match.end())
            end = len(text) if end < 0 else end + len(closing)
            pieces.append(token + blank(text[match.end() : end - len(closing)]) + text[end - len(closing) : end])
        elif token == "'" and NUMBER_BEFORE_SEPARATOR.search(text, max(0, start - 64), start):
            end = start + 1
            pieces.append(token)
        else:
            closer = STRING_END if token.endswith('"') else CHARACTER_END
            literal = closer.match(text, match.end())
            end = literal.end() if literal else match.end()
            pieces.append(token + blank(text[match.end() : end - 1]) + text[end - 1 : end] if literal else token)
        position = end
    return "".join(pieces).split("\n"), comments


def comment_findings(code_lines, comments):
    """The // and /*! comments that break a rule, and the functions documented on one line."""
    findings = []
    for line, column, comment in comments:
        before = code_lines[line - 1][:column].strip()
        if comment.startswith("///<") and before:
            continue
        if comment.startswith(("///", "//!", "/*!")):
            findings.append((line, "triple-slash", comment))
        elif comment.startswith("//"):
            if ALLOWED_LINE_COMMENT.match(comment):
                continue
            if before.endswith("}") and NAMESPACE_CLOSING.match(comment):
                continue
            if re.match(r"#\s*endif\b", before) or TABLE_ROW.match(before):
                continue
            findings.append((line, "banner" if BANNER.match(comment) else "line-comment", comment))
        elif (
            comment.startswith("/**")
            and not comment.startswith("/**<")
            and "\n" not in comment
            and not before
            and not GROUP_MARKER.match(comment)
        ):
            following = next((code for code in code_lines[line:] if code.strip()), "")
            declaration = re.sub(r"\[\[.*?\]\]|\balignas\s*\([^()]*\)", "", following)
            if "(" not in declaration or NOT_A_FUNCTION.match(declaration):
                continue
            prefix = declaration[: declaration.index("(")]
            if "=" in prefix or "{" in prefix or "[" in prefix:
                continue
            findings.append((line, "one-line-function-doc", comment))
    return findings


def header_bodies(code_lines):
    """The function bodies of more than one line, as (line of the signature, signature)."""
    findings = []
    for index in range(1, len(code_lines)):
        opening = re.match(r"^(\s*)\{\s*$", code_lines[index])
        if not opening:
            continue
        signature = code_lines[index - 1].rstrip()
        if "(" not in signature and not signature.endswith(SIGNATURE_ENDINGS):
            continue
        if signature.endswith((";", ",", "{")) or CONTROL.search(signature):
            continue
        closing = re.compile("^" + re.escape(opening.group(1)) + r"\}")
        for end in range(index + 1, len(code_lines)):
            if closing.match(code_lines[end]):
                if end - index - 1 > 1:
                    findings.append((index, "header-body", signature.strip()))
                break
    return findings


def check_text(path, text):
    """Every rule the text of the file at path breaks, as (line, kind, excerpt), in line order."""
    code_lines, comments = lex(text)
    findings = comment_findings(code_lines, comments)
    suffix = pathlib.PurePath(path).suffix
    if (
        suffix in SOURCE_SUFFIXES
        and comments
        and comments[0][:2] == (1, 0)
        and comments[0][2].startswith("/**")
        and "@file" in comments[0][2]
    ):
        findings.append((1, "file-block", comments[0][2].split("\n")[0]))
    findings += [(number, "using", code) for number, code in enumerate(code_lines, 1) if SHORTENING_USING.match(code)]
    if suffix in HEADER_SUFFIXES:
        findings += header_bodies(code_lines)
    return sorted(findings)


def read_declared(path):
    """The entries of the declared list as {(file, kind): reason}, and its malformed lines."""
    entries = {}
    malformed = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(None, 2)
        if len(fields) < 3 or (fields[1] not in KINDS and fields[1] != "*") or (fields[0], fields[1]) in entries:
            malformed.append(
                f"{path.name}:{number}: expected `<file> <rule or *> <reason>`, once per file and rule: {line}"
            )
            continue
        entries[(fields[0], fields[1])] = fields[2]
    return entries, malformed


def check_files(files, declared_path, out=sys.stdout):
    """Checks the files against the rules and the declared list; returns the exit status."""
    if not files:
        print("no C or C++ file to check", file=out)
        return 1
    entries, malformed = read_declared(declared_path) if declared_path.exists() else ({}, [])
    used = set()
    status = 0
    for message in malformed:
        print(message, file=out)
        status = 1
    for path in files:
        try:
            text = pathlib.Path(path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            print(f"cannot read {path}: {error}", file=out)
            status = 2
            continue
        for line, kind, excerpt in check_text(path, text):
            entry = next((key for key in ((path, kind), (path, "*")) if key in entries), None)
            if entry is not None:
                used.add(entry)
                continue
            print(f"{path}:{line}: {kind}: {MESSAGES[kind]}: {excerpt.strip()[:100]}", file=out)
            status = max(status, 1)
    checked = set(files)
    for (path, kind), reason in sorted(entries.items()):
        if path in checked and (path, kind) not in used:
            print(
                f"{declared_path.name}: {path} is declared for {kind} ({reason}) and breaks no such rule:"
                " remove the entry",
                file=out,
            )
            status = max(status, 1)
    return status


def tracked_files():
    """Every C and C++ file git tracks in this repository, relative to its root."""
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--"] + [f"*{suffix}" for suffix in SOURCE_SUFFIXES + HEADER_SUFFIXES],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return [path for path in listed.split("\0") if path]


REFUSED_SOURCE = """\
/** @file refused.cpp */
// a line comment
int value = 1; // a trailing comment
// ======================
/// a triple-slash doc
///< a trailing doc with nothing to trail
//! a Qt doc
/*! a Qt block */
using namespace std;
using std::size_t;
/** @brief Returns one. */
int one();
"""
REFUSED_LINES = {
    1: "file-block",
    2: "line-comment",
    3: "line-comment",
    4: "banner",
    5: "triple-slash",
    6: "triple-slash",
    7: "triple-slash",
    8: "triple-slash",
    9: "using",
    10: "using",
    11: "one-line-function-doc",
}
REFUSED_HEADER = """\
#pragma once
inline int twice(int value)
{
    const int doubled = value * 2;
    return doubled;
}
"""
ACCEPTED_HEADER = """\
/**
 * @file accepted.hpp
 * @brief A header keeps its block.
 */
#ifndef ACCEPTED_HPP
#    define ACCEPTED_HPP
namespace x {
/** @brief Bytes in a block. */
inline constexpr int kBlock = 4096;
/** @brief A pair of coordinates. */
struct Point {
    int x; /**< Across. */
    int z; ///< Along.
};
/** @brief The default point. */
inline constexpr Point kOrigin = make(0, 0);
using Alias = std::vector<int>;
/**
 * @brief Returns its argument.
 */
inline int same(int value) { return value; }
void ignore(int /*value*/);
// TODO(#12): a tracked change.
} // namespace x
#endif // ACCEPTED_HPP
"""
ACCEPTED_SOURCE = """\
#include "accepted.hpp"
static const char *kUrl = "http://example.org // not a comment";
static const char *kRaw = R"json({"a": "// still not", "b": "/* nor this */"})json";
static const char kSlash = '/';
static const int kMillion = 1'000'000;
static const int kRows[][2] = {
    {1, 2}, // the first row
    {3, 4}  // the last row
};
/** @brief Storage for one int, aligned as one. */
alignas(int) static unsigned char kStorage[sizeof(int)];
static const char *kNames[] = {
    "a", // a string row
    kUrl // a named row
};
int body()
{
    const int first = 1;
    return first + kMillion;
}
"""


def self_test():
    """Runs the checks on fixtures and returns the exit status."""
    failures = []
    origin = os.getcwd()
    with tempfile.TemporaryDirectory() as directory:
        os.chdir(directory)
        empty_declared = pathlib.Path("none.declared")
        fixtures = {
            "refused.cpp": REFUSED_SOURCE,
            "refused.hpp": REFUSED_HEADER,
            "accepted.hpp": ACCEPTED_HEADER,
            "accepted.cpp": ACCEPTED_SOURCE,
            "accepted.inl": ACCEPTED_SOURCE.replace('#include "accepted.hpp"', ""),
        }
        for name, text in fixtures.items():
            pathlib.Path(name).write_text(text, encoding="utf-8")

        report = Collected()
        status = check_files(["refused.cpp"], empty_declared, report)
        reported = {int(line.split(":")[1]): line.split(":")[2].strip() for line in report.lines}
        if status != 1 or reported != REFUSED_LINES:
            failures.append(f"the refused fixture gave status {status} and {reported}, not 1 and {REFUSED_LINES}")
        report = Collected()
        if check_files(["refused.hpp"], empty_declared, report) != 1 or report.lines[:1] != [
            "refused.hpp:2: header-body: " + MESSAGES["header-body"] + ": inline int twice(int value)"
        ]:
            failures.append(f"a body of two lines in a header was not reported at its signature: {report.lines}")

        report = Collected()
        status = check_files(["accepted.hpp", "accepted.cpp", "accepted.inl"], empty_declared, report)
        if status != 0:
            failures.append("an accepted fixture was refused:\n" + "\n".join(report.lines))

        declared = pathlib.Path("fixture.declared")
        declared.write_text(
            "# file rule reason\nrefused.hpp header-body a fixture\naccepted.cpp * a stale entry\n", encoding="utf-8"
        )
        report = Collected()
        status = check_files(["refused.hpp", "accepted.cpp", "refused.cpp"], declared, report)
        stale = [line for line in report.lines if "accepted.cpp is declared for *" in line]
        if (
            status != 1
            or any(line.startswith("refused.hpp:") for line in report.lines)
            or len(stale) != 1
            or not any(line.startswith("refused.cpp:2:") for line in report.lines)
        ):
            failures.append(
                "a declared entry did not silence its rule in its file only, or a stale one passed:\n"
                + "\n".join(report.lines)
            )
        declared.write_text("refused.hpp header-body\nrefused.hpp no-such-rule a reason\n", encoding="utf-8")
        report = Collected()
        if (
            check_files(["accepted.cpp"], declared, report) != 1
            or len([l for l in report.lines if "expected" in l]) != 2
        ):
            failures.append("a malformed declared entry passed:\n" + "\n".join(report.lines))

        report = Collected()
        status = check_files(["missing.cpp", "refused.cpp"], empty_declared, report)
        if (
            status != 2
            or not any(line.startswith("cannot read missing.cpp") for line in report.lines)
            or not any(line.startswith("refused.cpp:2:") for line in report.lines)
        ):
            failures.append(
                "a file that cannot be read did not give status 2, or hid the others:\n" + "\n".join(report.lines)
            )

        if check_files([], empty_declared, Collected()) != 1:
            failures.append("an empty list of files passed")
        os.chdir(origin)
    for failure in failures:
        print(f"self-test: {failure}")
    if failures:
        return 1
    print("self-test: ok")
    return 0


class Collected:
    """A stream that keeps the lines printed to it."""

    def __init__(self):
        self.lines = []
        self._pending = ""

    def write(self, text):
        self._pending += text
        *complete, self._pending = self._pending.split("\n")
        self.lines += complete


def main(arguments):
    if arguments[:1] in (["-h"], ["--help"]):
        print(__doc__)
        return 0
    if arguments == ["--self-test"]:
        return self_test()
    if any(argument.startswith("-") for argument in arguments):
        print(__doc__, file=sys.stderr)
        return 2
    root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], check=True, capture_output=True, text=True
    ).stdout.strip()
    files = [os.path.relpath(os.path.abspath(argument), root) for argument in arguments]
    os.chdir(root)
    if not arguments:
        files = tracked_files()
    files = [path for path in files if pathlib.PurePath(path).suffix in SOURCE_SUFFIXES + HEADER_SUFFIXES]
    return check_files(files, DECLARED)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
