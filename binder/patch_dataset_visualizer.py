#!/usr/bin/env python3
"""Patch huggingface/lerobot-dataset-visualizer's next.config.ts for this deployment.

Run by the viewer's build stage in ``binder/Dockerfile``, against a fresh clone.
Three additions, each working around something that is not configurable upstream:

1. ``env`` -- the README presents ``DATASET_URL`` as the way to point the viewer at
   something other than the Hugging Face Hub, and it only half works. The fetch that
   actually loads an episode lives in ``fetch-data.ts``, which is imported by a
   ``"use client"`` component, and Next only inlines ``NEXT_PUBLIC_*`` into the client
   bundle -- so in the browser ``process.env.DATASET_URL`` is ``undefined`` and the
   code falls back to huggingface.co. The symptom is a local dataset 404ing against
   the Hub while the local file server logs no requests at all. ``env`` is Next's own
   documented way to inline a variable that is not ``NEXT_PUBLIC_*``.

2. ``basePath`` -- the viewer is served under ``/user/<name>/proxy/absolute/3000``.
   It has to be ``basePath`` and not ``assetPrefix``: the latter moves the static
   assets and leaves the router's navigation and the RSC payload requests
   root-relative, which under a path proxy leave the prefix behind.

3. ``output: "standalone"`` -- so the runtime image carries a self-contained server
   and its own minimal ``node_modules`` instead of the 430-package install.

Appended to the existing config object rather than replacing the file, because
upstream's own settings there are load-bearing (``transpilePackages: ["three"]``,
``optimizePackageImports``, ``generateBuildId``) and a copy of them here would go
stale without anyone noticing. If the anchor is ever gone this fails the build, which
is the point -- a silently unpatched config produces an image whose viewer quietly
talks to huggingface.co.
"""

import sys
from pathlib import Path

ANCHOR = "const nextConfig: NextConfig = {"

ADDITIONS = """
  // ---- cram-vrb-lab (binder/patch_dataset_visualizer.py) ----
  // Inline DATASET_URL into the client bundle; Next only does that for
  // NEXT_PUBLIC_* on its own, and the episode fetch runs client-side.
  env: {
    DATASET_URL: process.env.DATASET_URL ?? "https://huggingface.co/datasets",
  },
  // Serve everything under the JupyterHub proxy prefix. Both this and DATASET_URL
  // above are baked in at build time, so the image is built with placeholders and
  // binder/entrypoint.sh substitutes the real values when the pod starts.
  basePath: process.env.NEXT_BASE_PATH || undefined,
  // A self-contained server, so the runtime image needs only the build output.
  output: "standalone",
  // ---- end cram-vrb-lab ----
"""


def main(path: Path) -> int:
    source = path.read_text()
    if "cram-vrb-lab" in source:
        print(f"{path}: already patched")
        return 0
    if ANCHOR not in source:
        print(
            f"{path}: could not find {ANCHOR!r}. Upstream changed the shape of its "
            "next.config.ts -- add env/basePath/output by hand and update this "
            "script. Refusing to build an unpatched viewer.",
            file=sys.stderr,
        )
        return 1
    path.write_text(source.replace(ANCHOR, ANCHOR + ADDITIONS, 1))
    print(f"{path}: patched (env, basePath, standalone output)")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1] if len(sys.argv) > 1 else "next.config.ts")))
