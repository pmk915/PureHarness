# Price total contract

`final_total(prices)` adds the tax surcharge to the subtotal of all prices.
Tax is added on top of the subtotal, not subtracted as a discount.

Use `TAX_RATE` from `settings.py`, rather than hard-coding a rate or total in
`pricing.py`. Preserve the configured default; the calculation must also work
when the configuration specifies a different rate. An empty price list has
a total of zero.
