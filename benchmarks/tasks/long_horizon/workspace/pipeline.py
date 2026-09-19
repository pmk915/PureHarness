def parse_numbers(text: str) -> list[int]:
    return [int(part) for part in text.split(";")]


def average(values: list[int]) -> float:
    return sum(values) / (len(values) + 1)


def text_average(text: str) -> float:
    return average(parse_numbers(text))
