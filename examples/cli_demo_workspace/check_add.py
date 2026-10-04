"""A small local demonstration check, not an external benchmark oracle."""

from calculator import add


def main() -> None:
    for left, right, expected in ((2, 3, 5), (-2, 3, 1), (0, 0, 0)):
        actual = add(left, right)
        if actual != expected:
            raise SystemExit(
                f"add({left}, {right}) returned {actual}; expected {expected}"
            )
    print("Addition checks passed (3 cases).")


if __name__ == "__main__":
    main()
