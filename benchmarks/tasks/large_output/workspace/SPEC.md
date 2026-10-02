# Sensor severity contract

`severity(reading)` returns:

- `critical` for readings at or above 90;
- `warning` for readings at or above 70 but below 90;
- `normal` for readings below 70.

`diagnose.py` is an available local diagnostic. It prints a large sensor report
with expected and observed classifications, both before and after repair.
