"""Application configuration and filesystem layout.

All patient data lives outside the repository, under a configurable data directory
(default ``~/Documents/Spectra``). Nothing here ever writes to the current working
directory, so the app behaves the same no matter where it is launched from.
"""

from __future__ import annotations

import json
import os
import platform
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

ENV_DATA_DIR = "SPECTRA_DATA_DIR"
CONFIG_FILENAME = "config.json"
MODEL_FILENAME = "hand_landmarker.task"

#: Repository root, used only to locate the bundled MediaPipe model.
PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent


def default_data_dir() -> Path:
    """Return the default data directory, honouring :data:`ENV_DATA_DIR`."""
    override = os.environ.get(ENV_DATA_DIR)
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / "Documents" / "Spectra").resolve()


@dataclass
class AppConfig:
    """Runtime configuration, persisted as JSON inside the data directory."""

    data_dir: Path = field(default_factory=default_data_dir)
    #: Folder (e.g. a synced drive with restricted access) where reports and
    #: backups are exported. Never used for the live SQLite database.
    share_dir: Path | None = None
    locale: str = "pt_BR"

    camera_index: int = 0
    capture_width: int = 1280
    capture_height: int = 720
    capture_fps: int = 30
    detection_width: int = 640
    detection_height: int = 360

    pointer_smooth_alpha: float = 0.35
    hover_select_secs: float = 1.2
    gesture_hold_secs: float = 1.5
    trail_length: int = 18
    undo_history: int = 15

    #: Safety and ergonomics (section 4 of the specification).
    session_limit_minutes: int = 20
    rest_prompt_minutes: int = 5

    #: Masks patient names on screen and in reports, for public presentations.
    demo_mode: bool = False
    sound_enabled: bool = True

    # ------------------------------------------------------------------ paths
    @property
    def config_path(self) -> Path:
        return self.data_dir / CONFIG_FILENAME

    @property
    def db_path(self) -> Path:
        return self.data_dir / "spectra.db"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def drawings_dir(self) -> Path:
        return self.data_dir / "drawings"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def exports_dir(self) -> Path:
        return self.data_dir / "exports"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def key_path(self) -> Path:
        return self.data_dir / "secrets" / "master.key"

    def model_path(self) -> Path:
        """Return the MediaPipe model path, preferring the copy bundled in the repo."""
        bundled = REPO_ROOT / MODEL_FILENAME
        if bundled.is_file():
            return bundled
        return self.models_dir / MODEL_FILENAME

    def ensure_directories(self) -> None:
        for path in (
            self.data_dir,
            self.models_dir,
            self.drawings_dir,
            self.reports_dir,
            self.exports_dir,
            self.logs_dir,
            self.key_path.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------- serialisation
    def to_dict(self) -> dict:
        data = asdict(self)
        data["data_dir"] = str(self.data_dir)
        data["share_dir"] = str(self.share_dir) if self.share_dir else None
        return data

    @classmethod
    def from_dict(cls, data: dict) -> AppConfig:
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known}
        if kwargs.get("data_dir"):
            kwargs["data_dir"] = Path(kwargs["data_dir"]).expanduser()
        if kwargs.get("share_dir"):
            kwargs["share_dir"] = Path(kwargs["share_dir"]).expanduser()
        else:
            kwargs["share_dir"] = None
        return cls(**kwargs)

    def save(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )


def load_config(data_dir: Path | None = None) -> AppConfig:
    """Load the configuration from ``data_dir``, falling back to defaults."""
    root = Path(data_dir).expanduser() if data_dir else default_data_dir()
    config_file = root / CONFIG_FILENAME
    if config_file.is_file():
        try:
            raw = json.loads(config_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw = {}
        raw["data_dir"] = str(root)
        return AppConfig.from_dict(raw)
    return AppConfig(data_dir=root)


def is_windows() -> bool:
    return platform.system() == "Windows"
