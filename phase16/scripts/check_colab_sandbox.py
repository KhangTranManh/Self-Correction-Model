"""Check that Colab can isolate a future model-code verifier process."""

from __future__ import annotations

import json
import subprocess


PAYLOAD = """
import json, os, socket
try:
    connection = socket.create_connection(("1.1.1.1", 80), timeout=1)
    connection.close()
    network_connected = True
except OSError:
    network_connected = False
print(json.dumps({
    "uid": os.getuid(),
    "content_visible": os.path.exists("/content"),
    "root_visible": os.path.exists("/root"),
    "network_connected": network_connected,
}))
"""


def main() -> None:
    command = [
        "bwrap", "--die-with-parent", "--unshare-user", "--uid", "65534",
        "--gid", "65534", "--unshare-net", "--unshare-pid",
        "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib",
        "--symlink", "usr/lib64", "/lib64", "--tmpfs", "/tmp",
        "--dev", "/dev", "--chdir", "/tmp",
        "/usr/bin/python3.13", "-I", "-S", "-c", PAYLOAD,
    ]
    run = subprocess.run(command, capture_output=True, text=True, timeout=10)
    if run.returncode:
        raise RuntimeError(f"Sandbox launch failed: {run.stderr[-500:]}")
    result = json.loads(run.stdout.strip())
    if (result["uid"] != 65534 or result["content_visible"]
            or result["root_visible"] or result["network_connected"]):
        raise AssertionError(f"Sandbox isolation failed: {result}")
    print(json.dumps({"sandbox_ready": True, **result}, sort_keys=True))


if __name__ == "__main__":
    main()
