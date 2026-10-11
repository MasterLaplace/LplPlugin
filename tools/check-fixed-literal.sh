#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: tools/check-fixed-literal.sh [<file>...]
       tools/check-fixed-literal.sh --self-test

Refuses a Fixed32, a Fixed64 or a FixedPoint<...> built from a non-zero integer literal. Their
constructor takes the raw fixed-point word, so Fixed32{10} holds 10/65536, not 10: the literal
reads as a value and means a bit pattern. Each construction is reported with the spelled-out
alternatives: Fixed32::fromInt(10) for the integer, Fixed32::one() or Fixed32::half(), and
Fixed32::fromRaw(10) when the raw word is meant.

It reads braces and parentheses: a temporary (`Fixed32{10}`), a declaration (`Fixed32 step{10};`),
a cast (`static_cast<Fixed32>(10)`), and the same through an alias the file declares
(`using F = math::Fixed32;` then `F{10}`, followed to the end of that file). It skips comments
and string literals. Zero is the same in both readings, so `Fixed32{0}` passes.

It cannot see what the line does not spell: a template parameter (`T{1}`), a member initialised
by name (`_step{10}`), an alias declared in another file, a second declarator (in
`Fixed32 a{1}, b{2};` only `a` is reported), a C-style cast (`(Fixed32)10`), a construction split
across lines, or an argument forwarded to the constructor (`emplace_back(10)`, `std::in_place`).

Without a file it checks every C and C++ source git tracks in this repository.

  --self-test  check that each construction of a fixture is refused with the alternatives, that
               zero, a computed raw word, a comment and a string pass, that an alias is followed
               in its own file only, that a file it cannot read fails without hiding the others,
               and that an empty list of files fails

Exit status: 0 when no construction is found, 1 when one is or when there is no file to check,
2 on a usage error or when a file cannot be read (the constructions of the others are still
printed).
EOF
}

# Prints one line per fixed-point value built from a non-zero integer literal:
# `<file>:<line>: <construction> holds the raw fixed-point word...`. Comments and string literals are
# blanked first, so a sentence that names the trap is not one.
find_constructions() {
    awk '
        function blank_comments_and_strings(line,    code, position, length_of_line, character, end_of_comment) {
            code = ""
            position = 1
            length_of_line = length(line)
            while (position <= length_of_line) {
                if (inside_block_comment) {
                    end_of_comment = index(substr(line, position), "*/")
                    if (end_of_comment == 0)
                        return code
                    position += end_of_comment + 1
                    inside_block_comment = 0
                    continue
                }
                if (substr(line, position, 2) == "//")
                    return code
                if (substr(line, position, 2) == "/*") {
                    inside_block_comment = 1
                    position += 2
                    continue
                }
                character = substr(line, position, 1)
                if (character == "\"" || (character == "'"'"'" && substr(line, position - 1, 1) !~ /[A-Za-z0-9_]/)) {
                    position = end_of_quoted(line, position, character)
                    code = code " "
                    continue
                }
                code = code character
                ++position
            }
            return code
        }

        function end_of_quoted(line, opening, quote,    position) {
            position = opening + 1
            while (position <= length(line) && substr(line, position, 1) != quote)
                position += (substr(line, position, 1) == "\\") ? 2 : 1
            return position + 1
        }

        function is_zero(literal) {
            gsub(/[-+[:space:]'"'"']/, "", literal)
            sub(/[uUlLzZ]+$/, "", literal)
            sub(/^0[xXbB]/, "", literal)
            return literal ~ /^0+$/
        }

        function report(construction,    literal, type) {
            sub(/^[^A-Za-z_]/, "", construction)
            literal = construction
            sub(/.*[{(]/, "", literal)
            sub(/[[:space:]]*[})]$/, "", literal)
            gsub(/[[:space:]]/, "", literal)
            if (is_zero(literal))
                return
            match(construction, type_pattern)
            type = substr(construction, RSTART, RLENGTH)
            printf "%s:%d: %s holds the raw fixed-point word %s, not the value %s: write %s::fromInt(%s) for the integer, %s::one() or %s::half(), or %s::fromRaw(%s) when the raw word is meant\n",
                FILENAME, FNR, construction, literal, literal, type, literal, type, type, type, literal
        }

        function report_every(code, pattern,    rest, construction) {
            rest = code
            while (match(rest, pattern)) {
                construction = substr(rest, RSTART, RLENGTH)
                rest = substr(rest, RSTART + RLENGTH)
                report(construction)
            }
        }

        function build_patterns() {
            type_pattern = "(Fixed32|Fixed64|FixedPoint([[:space:]]*<[^<>]*>)?" aliases ")"
            temporary = "(^|[^A-Za-z0-9_])" type_pattern opening_pattern literal_pattern closing_pattern
            declaration = "(^|[^A-Za-z0-9_])" type_pattern "[[:space:]]+[A-Za-z_][A-Za-z0-9_]*" opening_pattern literal_pattern closing_pattern
            cast = "static_cast[[:space:]]*<[[:space:]]*[A-Za-z0-9_:]*" type_pattern "[[:space:]]*>" opening_pattern literal_pattern closing_pattern
            using_alias = "(^|[^A-Za-z0-9_])using[[:space:]]+[A-Za-z_][A-Za-z0-9_]*[[:space:]]*=[[:space:]]*[A-Za-z0-9_:]*" type_pattern "[[:space:]]*;"
            typedef_alias = "(^|[^A-Za-z0-9_])typedef[[:space:]]+[A-Za-z0-9_:]*" type_pattern "[[:space:]]+[A-Za-z_][A-Za-z0-9_]*[[:space:]]*;"
        }

        function remember_alias(name) {
            if (index(aliases "|", "|" name "|") > 0)
                return
            aliases = aliases "|" name
            build_patterns()
        }

        function remember_aliases(code,    declaration_text) {
            if (match(code, using_alias)) {
                declaration_text = substr(code, RSTART, RLENGTH)
                sub(/^[^A-Za-z_]?using[[:space:]]+/, "", declaration_text)
                sub(/[[:space:]]*=.*/, "", declaration_text)
                remember_alias(declaration_text)
            }
            if (match(code, typedef_alias)) {
                declaration_text = substr(code, RSTART, RLENGTH)
                sub(/[[:space:]]*;$/, "", declaration_text)
                match(declaration_text, /[A-Za-z_][A-Za-z0-9_]*$/)
                remember_alias(substr(declaration_text, RSTART, RLENGTH))
            }
        }

        BEGIN {
            literal_pattern = "[-+]?[[:space:]]*(0[xX][0-9a-fA-F'"'"']+|0[bB][01'"'"']+|[0-9][0-9'"'"']*)[uUlLzZ]*"
            opening_pattern = "[[:space:]]*[{(][[:space:]]*"
            closing_pattern = "[[:space:]]*[})]"
            aliases = ""
            build_patterns()
        }

        FNR == 1 {
            inside_block_comment = 0
            aliases = ""
            build_patterns()
        }

        !inside_block_comment && aliases == "" && index($0, "Fixed") == 0 && index($0, "/*") == 0 { next }

        {
            code = blank_comments_and_strings($0)
            remember_aliases(code)
            report_every(code, temporary)
            report_every(code, declaration)
            report_every(code, cast)
        }
    ' "$@"
}

check_files() {
    if [ "$#" -eq 0 ]; then
        echo "check-fixed-literal: no C or C++ source to check" >&2
        return 1
    fi
    local readable=() file
    for file in "$@"; do
        if [ -f "$file" ] && [ -r "$file" ]; then
            readable+=("$file")
        else
            echo "check-fixed-literal: cannot read $file" >&2
        fi
    done
    local constructions=""
    if [ "${#readable[@]}" -gt 0 ]; then
        constructions="$(find_constructions "${readable[@]}")"
    fi
    if [ "${#readable[@]}" -ne "$#" ]; then
        [ -z "$constructions" ] || printf '%s\n' "$constructions"
        echo "unchecked: $(($# - ${#readable[@]})) of $# files could not be read" >&2
        return 2
    fi
    if [ -z "$constructions" ]; then
        echo "ok: no Fixed32, Fixed64 or FixedPoint built from a non-zero integer literal in $# files"
        return 0
    fi
    printf '%s\n' "$constructions"
    echo "refused: $(printf '%s\n' "$constructions" | wc -l) built from a non-zero integer literal in $# files"
    return 1
}

check_repository() {
    local root
    root="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
    cd "$root"
    local files=()
    mapfile -d '' files < <(git ls-files -z -- '*.c' '*.h' '*.cpp' '*.hpp' '*.inl' '*.cu' '*.cuh')
    check_files "${files[@]}"
}

self_test() {
    local directory
    directory="$(mktemp -d)"
    trap 'rm -rf "$directory"' RETURN

    cat >"$directory/refused.cpp" <<'EOF'
auto a = Fixed32{10};
auto b = math::Fixed32{ 10 };
auto c = lpl::math::Fixed32 {-3};
auto d = Fixed32(10);
auto e = Fixed32{0x10};
auto f = Fixed32{1u};
auto g = Fixed32{1'000};
auto h = Fixed32{0b1};
auto i = Fixed64{2};
auto j = FixedPoint<core::i32, 16>{3};
return FixedPoint{4};
math::Fixed32 step{10};
constexpr math::Fixed32 kStep(10);
auto k = static_cast<math::Fixed32>(10);
world.enableSpatialPartition(math::Fixed32{10}, capacity); // a comment after the code
auto pair = Fixed32{1} + Fixed32{2};
EOF
    cat >"$directory/accepted.cpp" <<'EOF'
auto a = Fixed32{};
auto b = Fixed32{0};
auto c = math::Fixed32{ 0 };
auto d = Fixed32{0u};
auto e = Fixed32{0x0};
auto f = Fixed32{-0};
auto g = Fixed32{0'000};
math::Fixed32 r{0};
auto h = Fixed32::fromInt(10);
auto i = Fixed32::fromRaw(1);
auto j = Fixed32{raw};
auto k = Fixed32{4096 << 16};
auto l = Fixed32{base + level * step};
auto m = Vec3<Fixed32>{1, 2, 3};
auto n = MyFixed32{10};
// fromFloat, NOT Fixed32{10}: the raw-representation constructor
/** Same trap as the Fixed32{10} raw-vs-value confusion. */
/*
 * Fixed32{10} inside a block comment
 */
check("Fixed32{10} reads as ten", value);
auto quote = '"'; auto o = Fixed32{0};
using FVec3 = math::Vec3<math::Fixed32>;
auto p = FVec3{1};
EOF
    cat >"$directory/aliased.cpp" <<'EOF'
using F = math::Fixed32;
auto a = F{10};
typedef lpl::math::Fixed64 Wide;
auto b = Wide(3);
auto c = F::fromInt(10);
auto d = F{0};
EOF
    cat >"$directory/unaliased.cpp" <<'EOF'
auto a = F{10};
EOF

    local failures=0 output status lines line
    status=0
    output="$(check_files "$directory/refused.cpp")" || status=$?
    [ "$status" -eq 1 ] || { echo "self-test: a fixture of constructions passed (status $status)"; failures=$((failures + 1)); }
    lines="$(wc -l <"$directory/refused.cpp")"
    for line in $(seq 1 "$lines"); do
        grep -q "refused.cpp:$line: " <<<"$output" || { echo "self-test: line $line of the refused fixture was not reported: $(sed -n "${line}p" "$directory/refused.cpp")"; failures=$((failures + 1)); }
    done
    [ "$(grep -c 'refused.cpp:' <<<"$output")" -eq "$((lines + 1))" ] || { echo "self-test: the refused fixture did not give one report per construction"; failures=$((failures + 1)); }
    grep -q 'refused.cpp:1: Fixed32{10} holds the raw fixed-point word 10, not the value 10: write Fixed32::fromInt(10).*Fixed32::one() or Fixed32::half().*Fixed32::fromRaw(10)' <<<"$output" || { echo "self-test: the report did not name the spelled-out alternatives"; failures=$((failures + 1)); }
    grep -q 'refused.cpp:3: Fixed32 {-3} .*Fixed32::fromInt(-3)' <<<"$output" || { echo "self-test: a negative literal was not carried into the alternatives"; failures=$((failures + 1)); }

    status=0
    output="$(check_files "$directory/accepted.cpp")" || status=$?
    [ "$status" -eq 0 ] || { echo "self-test: an accepted fixture was refused:"; echo "$output"; failures=$((failures + 1)); }

    status=0
    output="$(check_files "$directory/aliased.cpp" "$directory/unaliased.cpp")" || status=$?
    [ "$status" -eq 1 ] || { echo "self-test: constructions through an alias passed (status $status)"; failures=$((failures + 1)); }
    if ! { [ "$(grep -c ': .* holds the raw' <<<"$output")" -eq 2 ] && grep -q 'aliased.cpp:2: F{10} .*F::fromInt(10)' <<<"$output" && grep -q 'aliased.cpp:4: Wide(3) .*Wide::fromInt(3)' <<<"$output"; }; then
        echo "self-test: an alias the file declares was not followed, or one it does not declare was:"
        echo "$output"
        failures=$((failures + 1))
    fi

    status=0
    output="$(check_files "$directory/missing.cpp" "$directory/refused.cpp" 2>&1)" || status=$?
    [ "$status" -eq 2 ] || { echo "self-test: a file that cannot be read did not give status 2 (status $status)"; failures=$((failures + 1)); }
    if ! { grep -q 'cannot read .*missing.cpp' <<<"$output" && grep -q 'refused.cpp:1: ' <<<"$output"; }; then
        echo "self-test: a file that cannot be read hid it, or hid the constructions of the others:"
        echo "$output"
        failures=$((failures + 1))
    fi

    status=0
    check_files >/dev/null 2>&1 || status=$?
    [ "$status" -eq 1 ] || { echo "self-test: an empty list of files passed"; failures=$((failures + 1)); }

    [ "$failures" -eq 0 ] || return 1
    echo "self-test: ok"
}

case "${1:-}" in
--self-test) self_test ;;
-h | --help) usage ;;
-*)
    usage >&2
    exit 2
    ;;
"") check_repository ;;
*) check_files "$@" ;;
esac
