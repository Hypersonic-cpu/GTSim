from .json_codec import dumps


def print_ir(value):
    """Canonical inspection view, intentionally not a second parser language."""
    return dumps(value)
