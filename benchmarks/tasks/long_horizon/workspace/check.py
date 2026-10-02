import math

from pipeline import parse_numbers, text_average


def main() -> None:
    sample = " 7,11, 15 "
    print("Checking comma-separated parsing...", flush=True)
    assert parse_numbers(sample) == [7, 11, 15], "Parsing check failed"
    print("Parsing check passed", flush=True)

    print("Checking arithmetic mean...", flush=True)
    assert math.isclose(text_average(sample), 11.0), "Mean check failed"
    print("Local pipeline check passed")


if __name__ == "__main__":
    main()
