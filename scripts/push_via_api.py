"""Publish a local commit through the GitHub REST API.

Needed when github.com:443 is unreachable but api.github.com is not. Replays the
local commit's changed paths as blobs/tree/commit and moves the branch ref.

    python scripts/push_via_api.py <owner/repo> <branch> <local-commit>
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys


def gh(path: str, payload: dict | None = None, method: str = "GET") -> dict:
    cmd = ["gh", "api", "-X", method, path]
    if payload is not None:
        cmd += ["--input", "-"]
    proc = subprocess.run(
        cmd,
        input=json.dumps(payload) if payload is not None else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise SystemExit(f"gh api {method} {path} failed:\n{proc.stderr}")
    return json.loads(proc.stdout)


def main() -> None:
    repo, branch, commit = sys.argv[1], sys.argv[2], sys.argv[3]
    force = "--force" in sys.argv[4:]
    parent_override = next((a.split("=", 1)[1] for a in sys.argv[4:] if a.startswith("--parent=")), None)
    message = subprocess.run(
        ["git", "log", "-1", "--format=%B", commit],
        capture_output=True, text=True, encoding="utf-8", check=True,
    ).stdout.strip()
    parent = parent_override or gh(f"repos/{repo}/git/ref/heads/{branch}")["object"]["sha"]
    base_tree = gh(f"repos/{repo}/git/commits/{parent}")["tree"]["sha"]

    changed = subprocess.run(
        ["git", "diff", "--name-status", f"{commit}^", commit],
        capture_output=True, text=True, encoding="utf-8", check=True,
    ).stdout.splitlines()

    entries = []
    for line in changed:
        status, _, path = line.partition("\t")
        path = path.strip()
        if status.startswith("D"):
            entries.append({"path": path, "mode": "100644", "type": "blob", "sha": None})
            continue
        # read from the object store, not the working tree: content filters mean
        # the checked-out bytes differ from what the commit actually records
        data = subprocess.run(
            ["git", "show", f"{commit}:{path}"],
            capture_output=True, check=True,
        ).stdout
        blob = gh(
            f"repos/{repo}/git/blobs",
            {"content": base64.b64encode(data).decode("ascii"), "encoding": "base64"},
            method="POST",
        )
        entries.append({"path": path, "mode": "100644", "type": "blob", "sha": blob["sha"]})
        print(f"blob {blob['sha'][:10]} {path}")

    tree = gh(f"repos/{repo}/git/trees", {"base_tree": base_tree, "tree": entries}, method="POST")
    new = gh(
        f"repos/{repo}/git/commits",
        {"message": message, "tree": tree["sha"], "parents": [parent]},
        method="POST",
    )
    gh(
        f"repos/{repo}/git/refs/heads/{branch}",
        {"sha": new["sha"], "force": force},
        method="PATCH",
    )
    print(f"pushed {new['sha']} to {branch}")


if __name__ == "__main__":
    main()
