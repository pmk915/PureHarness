from pipeline import parse_numbers, text_average


assert parse_numbers("1, 2,3") == [1, 2, 3]
assert text_average("2,4,6") == 4.0
print("long_horizon passed")
