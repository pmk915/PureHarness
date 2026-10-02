import sys

from pathlib import Path


sys.path.insert(0, str(Path.cwd()))

from analyzer import severity


if severity(95) != "critical":
    print("Sensor validation report:")
    for index in range(400):
        reading = 90 + (index % 10)
        print(
            f"sensor-{index:03d}: reading={reading}; "
            "expected=critical; observed=warning"
        )
    raise SystemExit(1)

assert severity(75) == "warning"
assert severity(20) == "normal"
for reading, expected in (
    (0, "normal"),
    (69.5, "normal"),
    (70, "warning"),
    (70.5, "warning"),
    (89.5, "warning"),
    (90, "critical"),
    (90.5, "critical"),
    (120, "critical"),
):
    assert severity(reading) == expected
print("large_output passed")
