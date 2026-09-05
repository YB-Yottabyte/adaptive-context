from payment import checkout


def test_checkout_with_discount() -> None:
    assert checkout(100, 0.20) == 80.0
