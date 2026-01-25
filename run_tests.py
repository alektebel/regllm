#!/usr/bin/env python3
"""
Test Runner for RegLLM

Run all tests or specific test categories.

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py --unit       # Run unit tests only
    python run_tests.py --integration # Run integration tests only
    python run_tests.py --coverage   # Run with coverage report
    python run_tests.py -k "pattern" # Run tests matching pattern
"""

import subprocess
import sys
import argparse
from pathlib import Path


def run_tests(args):
    """Run pytest with specified arguments."""
    project_root = Path(__file__).parent

    # Base pytest command
    cmd = [sys.executable, "-m", "pytest"]

    # Add verbosity
    if args.verbose:
        cmd.append("-v")

    # Add coverage if requested
    if args.coverage:
        cmd.extend(["--cov=src", "--cov-report=term-missing", "--cov-report=html"])

    # Filter by test type
    if args.unit:
        cmd.extend(["-m", "unit or not (integration or slow)"])
    elif args.integration:
        cmd.extend(["-m", "integration"])
    elif args.slow:
        cmd.extend(["-m", "slow"])

    # Pattern matching
    if args.pattern:
        cmd.extend(["-k", args.pattern])

    # Specific test file
    if args.file:
        cmd.append(args.file)

    # Number of failures to stop after
    if args.maxfail:
        cmd.extend([f"--maxfail={args.maxfail}"])

    # Show local variables in tracebacks
    if args.showlocals:
        cmd.append("-l")

    # Run from project root
    print(f"Running: {' '.join(cmd)}")
    print("=" * 60)

    result = subprocess.run(cmd, cwd=project_root)

    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run RegLLM tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run with coverage report",
    )

    parser.add_argument(
        "--unit",
        action="store_true",
        help="Run unit tests only",
    )

    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run integration tests only",
    )

    parser.add_argument(
        "--slow",
        action="store_true",
        help="Run slow tests only",
    )

    parser.add_argument(
        "-k", "--pattern",
        type=str,
        help="Run tests matching pattern",
    )

    parser.add_argument(
        "-f", "--file",
        type=str,
        help="Run specific test file",
    )

    parser.add_argument(
        "--maxfail",
        type=int,
        help="Stop after N failures",
    )

    parser.add_argument(
        "-l", "--showlocals",
        action="store_true",
        help="Show local variables in tracebacks",
    )

    args = parser.parse_args()

    sys.exit(run_tests(args))


if __name__ == "__main__":
    main()
