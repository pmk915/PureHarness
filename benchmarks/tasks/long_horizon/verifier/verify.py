import math
import sys

from pathlib import Path


sys.path.insert(0, str(Path.cwd()))

from pipeline import average, parse_numbers, text_average


assert parse_numbers("1, 2,3") == [1, 2, 3]
assert text_average("2,4,6") == 4.0
for text, numbers, expected in (
    (" -9, 3, 12, -2 ", [-9, 3, 12, -2], 1.0),
    ("5", [5], 5.0),
    ("\t1, 2\n", [1, 2], 1.5),
    ("-8,-4,-3", [-8, -4, -3], -5.0),
    ("0, 2,4, 6,8", [0, 2, 4, 6, 8], 4.0),
):
    assert parse_numbers(text) == numbers
    assert math.isclose(average(numbers), expected, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(text_average(text), expected, rel_tol=1e-9, abs_tol=1e-9)
print("long_horizon passed")
