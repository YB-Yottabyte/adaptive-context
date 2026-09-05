"""Authorization decisions for user accounts."""

from permissions import permissions_for


def can_access(user: dict[str, object], action: str) -> bool:
    """Return whether an active user may perform an action."""
    role = str(user.get("role", "guest")).strip().lower()
    permissions = permissions_for(role)
    is_active = bool(user.get("active", False))

    role_allows_action = action in permissions
    active_role_access = is_active and role_allows_action
    return active_role_access or role == "admin"
