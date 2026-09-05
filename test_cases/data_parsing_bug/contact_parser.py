"""Import contacts from the CSV export produced by the CRM."""

from contact import Contact


def parse_contacts(document: str) -> list[Contact]:
    """Parse a CSV document containing id, name, and email columns."""
    lines = [line for line in document.splitlines() if line.strip()]
    if not lines:
        return []

    headers = [column.strip() for column in lines[0].split(",")]
    contacts: list[Contact] = []
    for line in lines[1:]:
        values = [column.strip().strip('"') for column in line.split(",")]
        record = dict(zip(headers, values, strict=False))
        contacts.append(
            Contact(
                contact_id=int(record["id"]),
                name=record["name"],
                email=record["email"],
            )
        )
    return contacts
