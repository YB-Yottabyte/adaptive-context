from discount import apply_discount


def checkout(price: float, discount: float) -> float:
    apply_discount(price, discount)
    return price
