"""Helpers available to the Python interface inside the sandbox."""
import subprocess as _sp


def run_tests():
    """Run the unit tests and print a short summary."""
    r = _sp.run(["/opt/tools/run_tests"], capture_output=True, text=True)
    print((r.stdout + r.stderr).rstrip())


def submit(report):
    """End the episode with a short report of what you did."""
    with open("/ctl/report", "w") as f:
        f.write(str(report))
    print("Submitted.")
