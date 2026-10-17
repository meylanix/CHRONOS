"""Exception hierarchy. The CLI turns these into readable messages and exit code 1."""


class ChronosError(Exception):
    """Base class for expected, user-facing errors."""


class ConfigError(ChronosError):
    pass


class DataError(ChronosError):
    """Missing, malformed or inconsistent data."""


class SourceError(ChronosError):
    """A remote source failed, returned something unexpected, or is not reachable."""


class NotImplementedSource(ChronosError):
    """The source is catalogued but no working connector exists (reported as NOT IMPLEMENTED)."""


class LeakageError(ChronosError):
    """A model or analysis tried to use information from after the forecast origin."""


class ValidationRuleError(ChronosError):
    """A knowledge-graph or scenario rule was violated (e.g. effect precedes cause)."""
