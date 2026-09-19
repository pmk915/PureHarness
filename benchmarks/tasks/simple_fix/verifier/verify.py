import sys

from pathlib import Path


sys.path.insert(0, str(Path.cwd()))

from calculator import add


assert add(2, 3) == 5
assert add(-4, 1) == -3
print("simple_fix passed")
