from decimal import Decimal

from cart import Cart
from cart_item import CartItem


def make_cart() -> Cart:
    return Cart([CartItem(sku="MUG", unit_price=Decimal("10.00"), quantity=2)])


def test_coupon_preview_has_expected_discounted_total() -> None:
    cart = make_cart()

    assert cart.quote(Decimal("0.25")) == Decimal("15.0000")


def test_coupon_preview_does_not_change_later_regular_quote() -> None:
    cart = make_cart()

    cart.quote(Decimal("0.25"))

    assert cart.quote() == Decimal("20.00")
