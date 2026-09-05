from auth import can_access


def test_active_editor_can_write() -> None:
    user = {"role": "editor", "active": True}

    assert can_access(user, "write") is True


def test_viewer_cannot_export() -> None:
    user = {"role": "viewer", "active": True}

    assert can_access(user, "export") is False


def test_deactivated_admin_cannot_export() -> None:
    user = {"role": "admin", "active": False}

    assert can_access(user, "export") is False
