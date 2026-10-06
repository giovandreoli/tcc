"""Entry point: ``python -m spectra`` starts the patient application."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from spectra import __version__
from spectra.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spectra", description="SPECTRA patient application")
    parser.add_argument("--version", action="version", version=f"SPECTRA {__version__}")
    parser.add_argument("--data-dir", type=Path, default=None, help="override the data directory")
    parser.add_argument("--camera", type=int, default=None, help="camera index")
    parser.add_argument(
        "--hand",
        choices=("Left", "Right"),
        default=None,
        help="handedness of the hand being treated (default: first hand detected)",
    )
    parser.add_argument(
        "--session",
        default=None,
        metavar="UUID",
        help="attach to a session created in the therapist panel and store its metrics",
    )
    parser.add_argument("--no-sound", action="store_true", help="disable audio feedback")
    parser.add_argument("--verbose", action="store_true", help="enable debug logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    config = load_config(args.data_dir)
    if args.camera is not None:
        config.camera_index = args.camera
    if args.no_sound:
        config.sound_enabled = False

    from spectra.app import SpectraApp  # imported late: pulls in MediaPipe

    try:
        SpectraApp(config, hand_label=args.hand, session_id=args.session).run()
    except RuntimeError as exc:
        logging.getLogger("spectra").error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
