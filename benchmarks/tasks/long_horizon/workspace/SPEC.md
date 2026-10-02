# Text-to-average pipeline contract

- A valid text input contains one or more comma-separated integers. Integers
  may be negative, and surrounding whitespace around each number is allowed.
- `parse_numbers(text)` returns those integers in their original order.
- `average(values)` computes the ordinary arithmetic mean of a nonempty list
  of integers: their sum divided by their count.
- `text_average(text)` composes parsing and averaging.

Run `python3 check.py` for local feedback. Its parsing check runs first; once
that succeeds, its mean check can expose a separate arithmetic defect. Rerun
the check after repairs. Passing it is useful feedback, not proof that all
valid inputs are handled correctly.

Empty lists, empty text and malformed input are outside this fixture's scored
contract; no particular error-handling behavior is required for them.
