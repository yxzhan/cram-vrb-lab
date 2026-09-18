"""Serve exported datasets the way the LeRobot dataset visualizer expects to find them.

Run from the repo root::

    python -m cram_vrb_lab.datasets.serve

then point the visualizer at it::

    DATASET_URL=http://localhost:8000/datasets npm run dev   # in the visualizer

**Why a shim and not ``python -m http.server``.** The visualizer is written against
the Hugging Face Hub and composes every request as
``<DATASET_URL>/<org>/<name>/resolve/<revision>/<path>``
(``src/utils/versionUtils.ts``, and ``@huggingface/lerobot`` for the parquet and the
video). A plain file server has no idea what ``resolve/main`` means, and the datasets
here have no ``<org>`` directory at all -- :mod:`cram_vrb_lab.datasets.lerobot_export`
writes them as ``<root>/<name>``, because the org half of a repo id is a Hub account
and there is no account involved. This translates between the two, which is about
thirty lines and saves rearranging the export's output to imitate a website.

Two things it has to do that ``SimpleHTTPRequestHandler`` does not:

- **Range requests.** ``hyparquet`` reads a parquet footer and then only the column
  chunks it needs, and a ``<video>`` element seeks. Both are HTTP ranges, and a
  server that ignores ``Range`` and returns 200 with the whole file makes the
  visualizer download every shard in full to read a few rows -- when it works at all,
  since the video element needs ``206`` to seek.
- **CORS.** The browser fetches these from the visualizer's own origin
  (``localhost:3000``), so without the headers below every request is blocked before
  it is sent. ``Access-Control-Expose-Headers`` is part of it: a cross-origin reader
  cannot see ``Content-Range`` unless it is named there, and a range reader that
  cannot see the range it got back is a range reader that cannot work.

Read-only and bound to the loopback interface by default: it serves a directory of
files to a browser on the same machine, and nothing about it should be reachable from
anywhere else.
"""

from __future__ import annotations

import argparse
import re
import sys
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import unquote

DEFAULT_PORT = 8000
DEFAULT_HOST = "127.0.0.1"

HUB_PATH = re.compile(
    r"^/(?:datasets/)?(?P<org>[^/]+)/(?P<name>[^/]+)/resolve/(?P<revision>[^/]+)/"
    r"(?P<path>.+)$"
)
"""The Hub URL shape the visualizer asks for.

``datasets/`` is optional because the two halves of the visualizer disagree: the
version check appends the repo id to ``DATASET_URL`` whole, while ``fetch-data.ts``
strips a trailing ``/datasets`` off it and hands the rest to ``@huggingface/lerobot``
as an *endpoint*, which puts its own ``datasets/`` back on. Accepting both means
neither has to be configured differently.

``org`` and ``revision`` are matched and then ignored. There is no Hub account here
and there are no revisions -- a directory has one version, the one on disk -- but the
visualizer's URLs carry both, so they have to be parsed out of the way.
"""

CONTENT_TYPES = {
    ".json": "application/json",
    ".parquet": "application/vnd.apache.parquet",
    ".mp4": "video/mp4",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".txt": "text/plain",
    ".jsonl": "application/x-ndjson",
}


def parse_range(header: Optional[str], size: int) -> Optional[Tuple[int, int]]:
    """``Range: bytes=start-end`` as an inclusive ``(start, end)``, or ``None``.

    Only the single-range form, which is all a browser's media element and
    ``hyparquet`` ever send. A multi-range request would need a multipart response,
    and answering it wrongly is worse than not answering it -- so anything else falls
    through to a normal 200 with the whole file, which is a valid answer to a range
    request and simply slower.

    ``bytes=-500`` (the last 500 bytes) is the form parquet readers use to find the
    footer, so it is not an edge case.
    """
    if not header or not header.startswith("bytes="):
        return None
    spec = header[len("bytes="):].strip()
    if "," in spec:
        return None
    start_text, _, end_text = spec.partition("-")
    try:
        if not start_text:  # bytes=-N, the last N bytes
            length = int(end_text)
            if length <= 0:
                return None
            return max(0, size - length), size - 1
        start = int(start_text)
        end = int(end_text) if end_text else size - 1
    except ValueError:
        return None
    end = min(end, size - 1)
    if start > end or start >= size:
        return None
    return start, end


class DatasetHandler(BaseHTTPRequestHandler):
    """Answers Hub-shaped URLs out of a local directory of exported datasets."""

    server_version = "cram-vrb-lab-dataset-server"
    protocol_version = "HTTP/1.1"  # so keep-alive and 206 work as the browser expects

    root: Path  # set by the partial in serve()

    def log_message(self, format: str, *args) -> None:  # noqa: A002 - base class name
        # One line per request, on stderr, without the default's timestamp noise. The
        # useful thing when a dataset will not load is *which* path 404s.
        sys.stderr.write(f"[dataset-server] {args[0]} -> {args[1]}\n")

    # %% resolution

    def resolve(self) -> Optional[Path]:
        """The file this request is asking for, or ``None`` if there is not one.

        Refuses any path that escapes the root once resolved, which is what makes
        ``..`` in a URL a 404 rather than a way to read the filesystem.
        """
        match = HUB_PATH.match(self.path.split("?", 1)[0])
        if not match:
            return None
        # Percent-decoded per segment, because @huggingface/lerobot builds its URLs
        # with encodeURIComponent on each one (resolveUrl in its dist/index.js).
        # Nothing this repo writes needs it today -- but a camera named with anything
        # outside [A-Za-z0-9] would 404 for a reason nobody would find.
        relative = "/".join(unquote(part) for part in match["path"].split("/"))
        candidate = (self.root / unquote(match["name"]) / relative).resolve()
        try:
            candidate.relative_to(self.root.resolve())
        except ValueError:
            return None
        return candidate if candidate.is_file() else None

    # %% verbs

    def do_OPTIONS(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's naming
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_HEAD(self) -> None:  # noqa: N802
        """Refuse, so the parquet reader sizes files with a ranged GET instead.

        **Deliberate, and the fix for a real failure.** ``jupyter-server-proxy``
        rewrites a HEAD response's ``Content-Length`` to 0 -- measured on this
        setup at 1336611 bytes direct and 0 through
        ``/user/<n>/proxy/<port>/`` -- and ``hyparquet`` sizes a parquet file from
        exactly that header. Behind the proxy every file therefore looked empty:
        no rows were read, the chart data came back as an empty array, and the
        viewer rendered no charts at all while the videos, which use ranged GETs
        and never a HEAD, played normally.

        Omitting the header was tried first and does not work -- the proxy adds
        ``Content-Length: 0`` back for the empty body, and hyparquet's fallback
        triggers on the header being *absent* (``if (!length)``), not on it being
        zero, so it believed the zero either way.

        403 is the other branch hyparquet already provides, for signed URLs that
        forbid HEAD (``byteLengthFromUrl``: "If HEAD request is forbidden (common
        with signed S3 URLs), try GET with range"). It falls back to a ranged GET
        and reads the size out of ``Content-Range``, which the proxy does forward
        intact. Nothing this server exists to serve needs a working HEAD, so
        giving up a method no client here depends on is cheaper than patching the
        viewer's parquet layer.
        """
        self.send_response(HTTPStatus.FORBIDDEN)
        self._send_cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        self._serve(body=True)

    # %% the work

    def _send_cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Range, Content-Type")
        # Without this the browser hides both from the fetch that asked for them, and
        # a range reader that cannot read Content-Range cannot find its next range.
        self.send_header("Access-Control-Expose-Headers",
                         "Content-Range, Content-Length, Accept-Ranges")

    def _not_found(self) -> None:
        body = b"not found\n"
        self.send_response(HTTPStatus.NOT_FOUND)
        self._send_cors()
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve(self, body: bool) -> None:
        path = self.resolve()
        if path is None:
            self._not_found()
            return

        size = path.stat().st_size
        span = parse_range(self.headers.get("Range"), size)
        start, end = span if span else (0, size - 1)
        length = end - start + 1

        self.send_response(
            HTTPStatus.PARTIAL_CONTENT if span else HTTPStatus.OK
        )
        self._send_cors()
        self.send_header(
            "Content-Type",
            CONTENT_TYPES.get(path.suffix, "application/octet-stream"),
        )
        # Advertised on every response, not just the 206s: a media element checks for
        # it before it will offer seeking at all.
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if span:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()

        if not body:
            return
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining > 0:
                chunk = handle.read(min(1 << 16, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    # A browser that seeks a video abandons the response it was
                    # reading. Normal, and not worth a traceback per seek.
                    return
                remaining -= len(chunk)


def serve(root: Path, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    """Serve ``root`` until interrupted.

    Threaded because the visualizer opens several of these at once -- three videos
    and a parquet -- and a single-threaded server would make them queue behind
    whichever range is slowest.
    """
    handler = partial(DatasetHandler)
    handler.root = root  # type: ignore[attr-defined]
    DatasetHandler.root = root

    datasets = sorted(
        entry.name for entry in root.iterdir()
        if entry.is_dir() and (entry / "meta" / "info.json").exists()
    )
    print(f"serving {len(datasets)} dataset(s) from {root} on http://{host}:{port}")
    for name in datasets:
        print(f"  {name}")
    if not datasets:
        print("  (none -- run cram_vrb_lab.datasets.lerobot_export first)")
    print("\nPoint the visualizer at it with:")
    print(f"  DATASET_URL=http://{host}:{port}/datasets bun dev")
    print("and open an episode at, e.g.:")
    # /<org>/<dataset>/<episode>, which is the visualizer's own route
    # (src/app/[org]/[dataset]/[episode]/page.tsx). The org is ignored here -- see
    # HUB_PATH -- so any non-empty segment works; the export's repo id is the
    # obvious one to use.
    print(f"  http://localhost:3000/cram_vrb_lab/"
          f"{datasets[0] if datasets else '<name>'}/episode_0")

    server = ThreadingHTTPServer((host, port), DatasetHandler)
    server.daemon_threads = True
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()


def main(argv=None) -> int:
    from cram_vrb_lab.sim.episode_recording import dataset_root

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root", default=None,
        help="directory holding the exported datasets, one subdirectory each "
             "(default: <episodes root>/lerobot).",
    )
    parser.add_argument("--host", default=DEFAULT_HOST,
                        help="interface to bind (default: %(default)s, loopback).")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help="port to listen on (default: %(default)s).")
    args = parser.parse_args(argv)

    root = Path(args.root) if args.root else Path(dataset_root()) / "lerobot"
    if not root.is_dir():
        raise SystemExit(
            f"no exported datasets at {root} -- run "
            "`python -m cram_vrb_lab.datasets.lerobot_export` first, or pass --root."
        )
    serve(root.resolve(), args.host, args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
