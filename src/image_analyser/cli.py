"""CLI entry point for image-analyser (argparse + lens-contract, the family pattern).

Usage:
  image-analyser photo.jpg
  image-analyser photo.jpg --json
  image-analyser photo.jpg --only metadata,quality
  image-analyser serve
  image-analyser serve --port 8006 --host 0.0.0.0
  image-analyser manifest
"""

from __future__ import annotations

import json as _json
import sys
from importlib.metadata import version as _pkg_version
from pathlib import Path


def _csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def main() -> None:
    import argparse

    from lens_contract import run_contract_subcommands

    from .manifest import MANIFEST

    # `serve` and `manifest` are the family's shared subcommands (lens-contract).
    if run_contract_subcommands(
        MANIFEST,
        app_path="image_analyser.api:app",
        default_port=8006,
        env_prefix="IMAGE_ANALYSER",
    ):
        return

    parser = argparse.ArgumentParser(
        prog="image-analyser",
        description="Static image analysis (CLI + FastAPI) for the analyser family",
        epilog="subcommands: `serve` (run the HTTP API), `manifest` (print the capability manifest)",
    )
    parser.add_argument(
        "--version", action="version", version=_pkg_version("image-analyser")
    )
    parser.add_argument("file", type=Path, help="image file to analyse")
    parser.add_argument(
        "--json", action="store_true", dest="compact", help="compact JSON (default is indented)"
    )
    parser.add_argument("--skip", help="comma-separated analyses to skip")
    parser.add_argument("--only", help="comma-separated analyses to run (mutex with --skip)")
    parser.add_argument(
        "--caption-backend", dest="caption_backend", help="local|api|auto|none"
    )
    _cmd_analyse(parser.parse_args())


def _cmd_analyse(args) -> None:
    from .image_analyser import ImageAnalyser
    from .exceptions import ImageAnalyserError, UnsupportedFormatError

    if not args.file.exists():
        print(f"error: file not found: {args.file}", file=sys.stderr)
        sys.exit(2)

    try:
        analyser = ImageAnalyser(
            skip=_csv(args.skip), only=_csv(args.only), caption_backend=args.caption_backend
        )
        result = analyser.analyse(args.file)
    except (ValueError, UnsupportedFormatError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)
    except ImageAnalyserError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)

    payload = result.model_dump(mode="json")
    print(_json.dumps(payload, separators=(",", ":")) if args.compact else _json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
