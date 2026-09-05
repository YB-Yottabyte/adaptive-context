"""Domain model for items placed in a shopping cart."""

from dataclasses import dataclass
from decimal import Decimal


@dataclass
class CartItem:
    sku: str
    unit_price: Decimal
    quantity: int
