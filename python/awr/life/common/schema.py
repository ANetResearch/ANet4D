"""Schema identifiers.

Documents written by this package carry identifiers of the form ``awr.life.<name>.v<N>``. Documents written by the CHIMERA benchmark before the
transplant into ANet4D carry ``chimera.<name>.v<N>``, and older ones ``d4ale.<name>.v<N>`` (task packs: ``drone4d-ale.tasks.v1``);
``schema_id`` maps those to the current form so that checks compare one spelling.
"""

LEGACY_PREFIXES = {"d4ale.": "awr.life.", "drone4d-ale.": "awr.life.", "chimera.": "awr.life."}


def schema_id(value):
    """``value`` with a legacy namespace prefix rewritten to ``awr.life.``; other values unchanged."""
    if isinstance(value, str):
        for old, new in LEGACY_PREFIXES.items():
            if value.startswith(old):
                return new + value[len(old) :]
    return value
