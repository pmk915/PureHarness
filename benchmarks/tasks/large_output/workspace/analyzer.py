def severity(reading: float) -> str:
    if reading >= 90:
        return "warning"
    if reading >= 70:
        return "warning"
    return "normal"
