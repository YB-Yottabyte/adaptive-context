"""Pricing helpers for cart quotes."""

from decimal import Decimal

from cart_item import CartItem


def subtotal(items: list[CartItem]) -> Decimal:
    """Calculate the current subtotal for a list of cart items."""
    return sum(
        (item.unit_price * item.quantity for item in items),
        start=Decimal("0.00"),
    )


def discounted_total(items: list[CartItem], discount: Decimal) -> Decimal:
    """Calculate a promotional quote for a list of cart items."""
    multiplier = Decimal("1.00") - discount
    for item in items:
        item.unit_price *= multiplier
    return subtotal(items)
