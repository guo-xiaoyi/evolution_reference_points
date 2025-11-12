# search.py
from itertools import product
import subprocess, sys
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

outcomes = [4, 6]
stakes = ["hi", "lo"]
attempts = [10000000000]


def run(cmd_parts, logname):
    # Mirror child output to both the console (to keep tqdm live) and the log file.
    process = subprocess.Popen(
        cmd_parts,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    stdout_stream = sys.stdout.buffer if hasattr(sys.stdout, "buffer") else None
    with open(logname, "wb") as logfile, process.stdout:
        for chunk in iter(lambda: process.stdout.read(1024), b""):
            if stdout_stream:
                stdout_stream.write(chunk)
                stdout_stream.flush()
            else:
                sys.stdout.write(chunk.decode(errors="replace"))
                sys.stdout.flush()
            logfile.write(chunk)
            logfile.flush()
    returncode = process.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, cmd_parts)


BASE_DIR = Path(__file__).resolve().parent


for o, s, a in product(outcomes, stakes, attempts):
    command = [
        sys.executable,
        str(BASE_DIR / "discrete_searcher.py"),
        "--outcomes",
        str(o),
        "--stake",
        s,
        "--attempts",
        str(a),
        "--cores 32"
    ]
    run(command, LOG_DIR / f"out_{o}_{s}_{a}.log")
