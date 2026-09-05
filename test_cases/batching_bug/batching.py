"""Batch construction for outbound notification jobs."""


def create_batches(recipients: list[str], batch_size: int) -> list[list[str]]:
    """Split recipients into non-empty batches of at most batch_size entries."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    return [
        recipients[start : start + batch_size]
        for start in range(0, len(recipients) + 1, batch_size)
    ]
