"""Mini code eval: 12 easy HumanEval-style tasks x CPU models, via the ollada CLI.

Each model answers each task as a fresh one-shot CLI process (the same way
the fixed prompt set was run, at each model's recipe defaults). The returned
code is extracted, executed against small asserts, and scored. A scoreboard
prints live and results are saved as JSON.

Usage (from the ollada checkout, venv activated):
    python scripts/mini_code_eval.py
    python scripts/mini_code_eval.py --tasks reverse_string,count_vowels
    python scripts/mini_code_eval.py --models nemotron:3b

Ctrl-C keeps partial results: the scoreboard prints for everything finished.
The JSON file lands in the current directory (see --out).

Decision rule (agreed before running): nemotron:3b earns a shelf slot if it
beats sdar:4b by >= 3 tasks out of 12, or sweeps a category sdar:4b fails
entirely. A 1-2 task lead is a tie and the model stays out.
"""

import argparse
import json
import re
import subprocess
import sys
import time

# Each task: the function signature the prompt asks for, a one-sentence spec,
# and asserts that run against the returned code. The model never sees the
# asserts; the spec is deliberately natural, the way a human would ask.
TASKS = [
    {
        "id": "reverse_string",
        "sig": "reverse_string(s)",
        "spec": "returns the string reversed",
        "tests": [
            'assert reverse_string("") == ""',
            'assert reverse_string("abc") == "cba"',
            'assert reverse_string("hello") == "olleh"',
        ],
    },
    {
        "id": "count_vowels",
        "sig": "count_vowels(s)",
        "spec": "returns the number of vowels (a, e, i, o, u, case-insensitive) in the string",
        "tests": [
            'assert count_vowels("hello") == 2',
            'assert count_vowels("AEIOU") == 5',
            'assert count_vowels("xyz") == 0',
            'assert count_vowels("") == 0',
        ],
    },
    {
        "id": "sum_evens",
        "sig": "sum_evens(nums)",
        "spec": "returns the sum of the even numbers in the list",
        "tests": [
            "assert sum_evens([1, 2, 3, 4]) == 6",
            "assert sum_evens([]) == 0",
            "assert sum_evens([1, 3, 5]) == 0",
        ],
    },
    {
        "id": "is_palindrome",
        "sig": "is_palindrome(s)",
        "spec": "returns True if the string reads the same forwards and backwards, ignoring case",
        "tests": [
            'assert is_palindrome("Racecar") is True',
            'assert is_palindrome("hello") is False',
            'assert is_palindrome("") is True',
        ],
    },
    {
        "id": "factorial",
        "sig": "factorial(n)",
        "spec": "returns n! for n >= 0",
        "tests": [
            "assert factorial(0) == 1",
            "assert factorial(1) == 1",
            "assert factorial(5) == 120",
        ],
    },
    {
        "id": "find_max",
        "sig": "find_max(nums)",
        "spec": "returns the largest number in a non-empty list",
        "tests": [
            "assert find_max([3, 1, 4]) == 4",
            "assert find_max([-1, -5]) == -1",
        ],
    },
    {
        "id": "remove_duplicates",
        "sig": "remove_duplicates(nums)",
        "spec": "returns a new list with duplicates removed, keeping the original order",
        "tests": [
            "assert remove_duplicates([1, 2, 2, 3, 1]) == [1, 2, 3]",
            "assert remove_duplicates([]) == []",
        ],
    },
    {
        "id": "fizzbuzz_value",
        "sig": "fizzbuzz_value(n)",
        "spec": 'returns "Fizz" if n is divisible by 3, "Buzz" if by 5, "FizzBuzz" if by both, otherwise the number as a string',
        "tests": [
            'assert fizzbuzz_value(3) == "Fizz"',
            'assert fizzbuzz_value(5) == "Buzz"',
            'assert fizzbuzz_value(15) == "FizzBuzz"',
            'assert fizzbuzz_value(7) == "7"',
        ],
    },
    {
        "id": "is_prime",
        "sig": "is_prime(n)",
        "spec": "returns True if n is prime, False otherwise",
        "tests": [
            "assert is_prime(2) is True",
            "assert is_prime(4) is False",
            "assert is_prime(17) is True",
            "assert is_prime(1) is False",
        ],
    },
    {
        "id": "word_count",
        "sig": "word_count(s)",
        "spec": "returns the number of words in the string, where words are separated by whitespace",
        "tests": [
            'assert word_count("hello world") == 2',
            'assert word_count("") == 0',
            'assert word_count("  a  b ") == 2',
        ],
    },
    {
        "id": "merge_dicts_sum",
        "sig": "merge_dicts_sum(a, b)",
        "spec": "returns a new dictionary merging a and b, summing the values for keys present in both",
        "tests": [
            'assert merge_dicts_sum({"a": 1}, {"a": 2, "b": 3}) == {"a": 3, "b": 3}',
            "assert merge_dicts_sum({}, {}) == {}",
        ],
    },
    {
        "id": "second_largest",
        "sig": "second_largest(nums)",
        "spec": "returns the second largest distinct value in the list",
        "tests": [
            "assert second_largest([1, 2, 3]) == 2",
            "assert second_largest([5, 5, 4]) == 4",
            "assert second_largest([1, 2]) == 1",
        ],
    },
]

DEFAULT_MODELS = ["sdar:1.7b", "sdar:4b", "nemotron:3b"]


def clean_output(text):
    """Drop CLI scaffolding lines so only the model's answer remains."""
    lines = []
    for line in text.splitlines():
        if line.startswith("working"):
            continue
        if re.fullmatch(r"\(\d+(\.\d+)?s\)", line.strip()):
            continue
        lines.append(line)
    return "\n".join(lines)


def extract_code(text):
    """First fenced code block if there is one, else the whole answer."""
    text = clean_output(text)
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def grade(code, test_lines):
    """Execute the returned code, then the asserts. Returns (passed, error)."""
    ns = {}
    try:
        exec(code, ns)
    except Exception as e:  # noqa: BLE001 - the error string is the report
        return False, f"code failed: {type(e).__name__}: {e}"
    try:
        exec("\n".join(test_lines), ns)
    except Exception as e:  # noqa: BLE001 - the error string is the report
        return False, f"test failed: {type(e).__name__}: {e}"
    return True, ""


def run_one(model, prompt, timeout):
    """One fresh ollada CLI run. Returns (stdout, seconds, error)."""
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "ollada.cli", "run", model, prompt],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return "", timeout, f"timeout after {timeout}s"
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-3:]
        return "", elapsed, f"ollada exit {proc.returncode}: " + " | ".join(tail)
    return proc.stdout, elapsed, ""


def print_report(results, models, tasks):
    """Scoreboard: rows are tasks, columns are models, plus totals."""
    width = max(len(t["id"]) for t in tasks)
    header = "task".ljust(width) + "  " + "  ".join(m.ljust(12) for m in models)
    print("\n" + header)
    print("-" * len(header))
    totals = {m: 0 for m in models}
    times = {m: [] for m in models}
    for task in tasks:
        row = task["id"].ljust(width) + "  "
        cells = []
        for model in models:
            entry = results.get(task["id"], {}).get(model)
            if entry is None:
                cells.append("·".ljust(12))
            else:
                times[model].append(entry["seconds"])
                if entry["pass"]:
                    totals[model] += 1
                    cells.append("PASS".ljust(12))
                else:
                    cells.append("FAIL".ljust(12))
        print(row + "  ".join(cells))
    print("-" * len(header))
    total_row = "TOTAL".ljust(width) + "  "
    total_row += "  ".join(f"{totals[m]}/{len(tasks)}".ljust(12) for m in models)
    print(total_row)
    mean_row = "mean secs".ljust(width) + "  "
    mean_row += "  ".join(
        (f"{sum(times[m]) / len(times[m]):.1f}" if times[m] else "—").ljust(12)
        for m in models
    )
    print(mean_row)


def main():
    parser = argparse.ArgumentParser(description="Mini code eval across ollada models.")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS),
                        help="comma-separated models (default: %(default)s)")
    parser.add_argument("--tasks", default=None,
                        help="comma-separated task ids (default: all)")
    parser.add_argument("--timeout", type=int, default=600,
                        help="seconds per model run (default: %(default)s)")
    parser.add_argument("--out", default="mini_code_eval_results.json",
                        help="JSON results file (default: %(default)s)")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    by_id = {t["id"]: t for t in TASKS}
    if args.tasks:
        wanted = [t.strip() for t in args.tasks.split(",") if t.strip()]
        unknown = [t for t in wanted if t not in by_id]
        if unknown:
            print(f"unknown tasks: {', '.join(unknown)}")
            print("available: " + ", ".join(by_id))
            return 2
        tasks = [by_id[t] for t in wanted]
    else:
        tasks = TASKS

    print(f"Running {len(tasks)} tasks x {len(models)} models "
          f"= {len(tasks) * len(models)} runs.")
    results = {}
    try:
        for task in tasks:
            results[task["id"]] = {}
            prompt = (f"Write a Python function {task['sig']} that "
                      f"{task['spec']}. Only output code.")
            for model in models:
                stdout, elapsed, error = run_one(model, prompt, args.timeout)
                if error:
                    entry = {"pass": False, "seconds": round(elapsed, 1),
                             "error": error[:300]}
                else:
                    passed, grade_error = grade(extract_code(stdout),
                                                task["tests"])
                    entry = {"pass": passed, "seconds": round(elapsed, 1),
                             "error": grade_error[:300]}
                results[task["id"]][model] = entry
                status = "PASS" if entry["pass"] else "FAIL"
                line = f"{model} {task['id']}: {status} ({entry['seconds']}s)"
                if entry["error"]:
                    line += f" -- {entry['error']}"
                print(line, flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted — reporting partial results.")

    print_report(results, models, tasks)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"models": models, "results": results}, f, indent=2)
    print(f"\nSaved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
