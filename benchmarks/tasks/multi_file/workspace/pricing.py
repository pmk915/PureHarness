from settings import TAX_RATE


def final_total(prices: list[float]) -> float:
    subtotal = sum(prices)
    return subtotal * (1 - TAX_RATE)
