from analyzer import severity


print("Sensor validation report:")
for index in range(400):
    reading = 90 + (index % 10)
    print(
        f"sensor-{index:03d}: reading={reading}; "
        f"expected=critical; observed={severity(reading)}"
    )
