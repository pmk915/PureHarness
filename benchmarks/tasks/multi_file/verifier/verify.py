import importlib
import math
import sys

from pathlib import Path


sys.path.insert(0, str(Path.cwd()))

import pricing
import settings


price_lists = ([], [10.0, 15.0], [6.5], [0.25, 2.5, 8.75])

# The visible configuration starts with an 8% surcharge.
for prices in price_lists:
    assert math.isclose(
        pricing.final_total(prices), sum(prices) * 1.08,
        rel_tol=1e-9, abs_tol=1e-9,
    )

# Vary configuration before loading the calculation, without requiring any
# particular import style or source-code implementation.
for rate in (0.0, 0.15, 0.23):
    settings.TAX_RATE = rate
    importlib.reload(pricing)
    for prices in price_lists:
        assert math.isclose(
            pricing.final_total(prices), sum(prices) * (1 + rate),
            rel_tol=1e-9, abs_tol=1e-9,
        )
print("multi_file passed")
