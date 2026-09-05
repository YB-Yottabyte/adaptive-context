"""Contact records imported by the customer support application."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Contact:
    contact_id: int
    name: str
    email: str
