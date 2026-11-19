from app.pricing import total_price


def test_string_prices_are_added_as_numbers():
    assert total_price("2.5", "3.0") == 5.5
