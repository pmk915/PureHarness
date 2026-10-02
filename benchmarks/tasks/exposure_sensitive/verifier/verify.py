import sys

from pathlib import Path


sys.path.insert(0, str(Path.cwd()))

from heading import render_heading


assert render_heading("  Mini Harness  ") == "MINI HARNESS"
assert render_heading("\tRelease Notes\n") == "RELEASE NOTES"
assert render_heading("sTaTuS") == "STATUS"
assert render_heading("   ") == ""
print("exposure_sensitive passed")
