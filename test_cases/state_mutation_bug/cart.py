"""Shopping cart operations used by checkout and quote previews."""

from decimal import Decimal

from cart_item import CartItem
from pricing import discounted_total, subtotal


class Cart:
    def __init__(self, items: list[CartItem]) -> None:
        self._items = list(items)

    def quote(self, discount: Decimal | None = None) -> Decimal:
        """Return a regular or promotional quote for the current cart."""
        if discount is None:
            return subtotal(self._items)
        return discounted_total(self._items, discount)
