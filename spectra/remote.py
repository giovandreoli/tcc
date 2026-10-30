"""Free internet access to the patient page through a Cloudflare quick tunnel.

``cloudflared tunnel --url http://localhost:<port>`` publishes the local panel at a
random ``https://<words>.trycloudflare.com`` address, with no account and no cost. The
panel itself stays bound to localhost; the tunnel client connects to it from this same
computer.

Requests arriving through the tunnel are recognised by Cloudflare's headers and by the
public host name, and only ever receive the **patient page** (room code + PIN). The
therapist panel answers exclusively to requests made on this computer.
"""

from __future__ import annotations

import logging
import platform
import re
import shutil
import subprocess
import threading
import urllib.request
from collections.abc import Mapping
from pathlib import Path

logger = logging.getLogger(__name__)

TUNNEL_URL = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "[::1]", "::1"})
#: Headers added by Cloudflare or any reverse proxy; their presence means "not local".
PROXY_HEADERS = ("cf-connecting-ip", "cf-ray", "x-forwarded-for", "x-forwarded-host")
DOWNLOAD_BASE = "https://github.com/cloudflare/cloudflared/releases/latest/download/"
START_TIMEOUT_SECONDS = 45.0


class TunnelError(RuntimeError):
    """Carries an i18n key."""


def _host_name(host: str) -> str:
    host = host.strip().lower()
    if host.startswith("["):
        return host.split("]")[0] + "]"
    return host.rsplit(":", 1)[0] if host.count(":") == 1 else host


def _origin_host(origin: str) -> str:
    """Host part of an ``Origin`` header such as ``https://name.trycloudflare.com``."""
    return _host_name(origin.split("://", 1)[-1].split("/", 1)[0])


def is_local_request(headers: Mapping[str, str] | None) -> bool:
    """True only for a browser on this computer talking to the panel directly.

    Checks the ``Host`` the request was addressed to, the ``Origin`` of the page (set by
    the browser on the WebSocket handshake, which no proxy rewrites) and proxy headers, so
    the decision holds even if a tunnel were configured to rewrite ``Host``.
    """
    if headers is None:
        return True
    normalized = {str(key).lower(): str(value) for key, value in headers.items()}
    if any(name in normalized for name in PROXY_HEADERS):
        return False
    origin = normalized.get("origin")
    if origin and origin != "null" and _origin_host(origin) not in LOCAL_HOSTS:
        return False
    host = normalized.get("host")
    return host is None or _host_name(host) in LOCAL_HOSTS


def binary_name() -> str:
    return "cloudflared.exe" if platform.system() == "Windows" else "cloudflared"


def download_asset() -> str:
    system = platform.system()
    machine = platform.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "amd64"
    if system == "Windows":
        return f"cloudflared-windows-{arch}.exe"
    if system == "Linux":
        return f"cloudflared-linux-{arch}"
    raise TunnelError("remote.error.platform")


def find_cloudflared(tools_dir: Path) -> Path | None:
    local = Path(tools_dir) / binary_name()
    if local.is_file():
        return local
    found = shutil.which("cloudflared")
    return Path(found) if found else None


def download_cloudflared(tools_dir: Path) -> Path:
    """Fetch the official cloudflared release from GitHub over HTTPS."""
    tools_dir = Path(tools_dir)
    tools_dir.mkdir(parents=True, exist_ok=True)
    target = tools_dir / binary_name()
    partial = target.with_suffix(target.suffix + ".part")
    try:
        with urllib.request.urlopen(DOWNLOAD_BASE + download_asset(), timeout=120) as response:
            partial.write_bytes(response.read())
    except OSError as exc:
        partial.unlink(missing_ok=True)
        raise TunnelError("remote.error.download") from exc
    partial.replace(target)
    target.chmod(0o755)
    return target


class Tunnel:
    """One cloudflared quick tunnel pointing at the local panel."""

    def __init__(self, binary: Path, port: int) -> None:
        self.binary = Path(binary)
        self.port = port
        self.url: str | None = None
        self.error_key: str | None = None
        self._process: subprocess.Popen[str] | None = None
        self._ready = threading.Event()

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def command(self) -> list[str]:
        return [
            str(self.binary),
            "tunnel",
            "--no-autoupdate",
            "--url",
            f"http://localhost:{self.port}",
        ]

    def start(self, timeout: float = START_TIMEOUT_SECONDS) -> str:
        if self.running and self.url:
            return self.url
        self.url = None
        self.error_key = None
        self._ready.clear()
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self._process = subprocess.Popen(
                self.command(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=flags,
            )
        except OSError as exc:
            raise TunnelError("remote.error.start") from exc
        threading.Thread(target=self._watch, daemon=True).start()
        if not self._ready.wait(timeout) or not self.url:
            self.stop()
            raise TunnelError(self.error_key or "remote.error.timeout")
        logger.info("internet access enabled through a Cloudflare quick tunnel")
        return self.url

    def _watch(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        for line in process.stderr:
            if self.url is None and (match := TUNNEL_URL.search(line)):
                self.url = match.group(0)
                self._ready.set()
        if self.url is None:
            self.error_key = "remote.error.start"
        self._ready.set()

    def stop(self) -> None:
        process, self._process = self._process, None
        self.url = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


def patient_link(base_url: str, code: str) -> str:
    return f"{base_url.rstrip('/')}/?sala={code}"
