"""Entry point for python -m rtest."""

import sys

from rtest._rtest import main_cli_with_args


def main() -> None:
    """CLI entry point for rtest."""
    arguments = sys.argv[1:]
    if "--unified-output" in arguments:
        arguments.remove("--unified-output")
        if not any(option in arguments for option in ("--collect-only", "-h", "--help", "--version", "-V")):
            from rtest.unified import run

            sys.exit(run(arguments))
    main_cli_with_args(arguments)


if __name__ == "__main__":
    main()
