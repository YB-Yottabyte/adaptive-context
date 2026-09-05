"""Role-based permissions used by the authentication service."""

ROLE_PERMISSIONS = {
    "admin": frozenset({"read", "write", "export"}),
    "editor": frozenset({"read", "write"}),
    "viewer": frozenset({"read"}),
    "guest": frozenset(),
}


def permissions_for(role: str) -> frozenset[str]:
    """Return permissions for a normalized role, defaulting to guest access."""
    normalized_role = role.strip().lower()
    return ROLE_PERMISSIONS.get(normalized_role, ROLE_PERMISSIONS["guest"])
