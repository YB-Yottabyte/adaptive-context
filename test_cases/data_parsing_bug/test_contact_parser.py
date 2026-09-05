from contact import Contact
from contact_parser import parse_contacts


def test_imports_simple_contact() -> None:
    document = "id,name,email\n3,Ada Lovelace,ada@example.com\n"

    assert parse_contacts(document) == [
        Contact(contact_id=3, name="Ada Lovelace", email="ada@example.com")
    ]


def test_imports_quoted_name_containing_comma() -> None:
    document = 'id,name,email\n7,"Rivera, Ana",ana@example.com\n'

    assert parse_contacts(document) == [
        Contact(contact_id=7, name="Rivera, Ana", email="ana@example.com")
    ]
