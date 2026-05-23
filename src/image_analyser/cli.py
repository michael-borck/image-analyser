"""Typer-based CLI for image-analyser."""

from __future__ import annotations

import json as _json
import os
import sys
from importlib.metadata import version as _pkg_version
from pathlib import Path

import typer

from .exceptions import ImageAnalyserError, UnsupportedFormatError

cli = typer.Typer(add_completion=False, no_args_is_help=True)


def _csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


@cli.command(help="Analyse a single image and print the result as JSON.")
def analyse(
    file: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),  # noqa: B008
    json_out: bool = typer.Option(False, "--json", help="Compact JSON output."),
    skip: str | None = typer.Option(None, "--skip", help="Comma-separated analyses to skip."),
    only: str | None = typer.Option(None, "--only", help="Comma-separated analyses to run (mutex with --skip)."),
    caption_backend: str | None = typer.Option(None, "--caption-backend", help="local|api|auto|none"),
) -> None:
    from .image_analyser import ImageAnalyser
    try:
        analyser = ImageAnalyser(
            skip=_csv(skip), only=_csv(only), caption_backend=caption_backend,
        )
        result = analyser.analyse(file)
    except ValueError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(code=2) from e
    except UnsupportedFormatError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(code=2) from e
    except ImageAnalyserError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(code=1) from e
    payload = result.model_dump(mode="json")
    typer.echo(_json.dumps(payload, separators=(",", ":")) if json_out else _json.dumps(payload, indent=2))


@cli.command(help="Start the FastAPI HTTP server.")
def serve(
    port: int = typer.Option(int(os.getenv("IMAGE_ANALYSER_PORT", "8006")), "--port"),
    host: str = typer.Option(os.getenv("IMAGE_ANALYSER_HOST", "127.0.0.1"), "--host"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    import uvicorn
    uvicorn.run("image_analyser.app:app", host=host, port=port, reload=reload)


@cli.command(help="Print the capability manifest as JSON.")
def manifest() -> None:
    from .manifest import MANIFEST
    typer.echo(_json.dumps(MANIFEST, indent=2))


@cli.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", is_eager=True, help="Show version and exit."),
) -> None:
    if version:
        typer.echo(_pkg_version("image-analyser"))
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


def main() -> None:
    # Make `image-analyser FILE [--json]` work as well as `image-analyser analyse FILE`.
    # Typer doesn't support this natively, so we promote a bare positional to the analyse command.
    argv = sys.argv[1:]
    if argv and not argv[0].startswith("-") and argv[0] not in {"analyse", "serve", "manifest"}:
        sys.argv = [sys.argv[0], "analyse", *argv]
    cli(prog_name="image-analyser")


if __name__ == "__main__":
    main()
