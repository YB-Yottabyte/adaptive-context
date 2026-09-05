"""Create delivery jobs from recipient batches."""

from dataclasses import dataclass

from batching import create_batches


@dataclass(frozen=True)
class DeliveryJob:
    sequence: int
    recipients: tuple[str, ...]


def plan_delivery(recipients: list[str], batch_size: int) -> list[DeliveryJob]:
    """Build numbered jobs for the notification worker."""
    return [
        DeliveryJob(sequence=index, recipients=tuple(batch))
        for index, batch in enumerate(
            create_batches(recipients, batch_size),
            start=1,
        )
    ]
