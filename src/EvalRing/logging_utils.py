"""
Logging helpers.

The library never configures the root logger; it only emits records on the
``EvalRing`` hierarchy so that host applications keep full control of handlers
and levels. Command-line entry points call :func:`configure_logging` once to
turn those records into console output.
"""

from __future__ import annotations

import logging
import sys

LOGGER_NAME = "EvalRing"

_CONSOLE_FORMAT = "%(message)s"
_VERBOSE_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger inside the ``EvalRing`` namespace.

    Args:
        name: Dotted module name, normally ``__name__``. A name already inside
            the ``EvalRing`` namespace is used as-is; anything else is nested
            beneath it.

    Returns:
        The configured :class:`logging.Logger` instance.
    """
    if not name or name == LOGGER_NAME:
        return logging.getLogger(LOGGER_NAME)
    if name.startswith(f"{LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def configure_logging(level: int = logging.INFO, *, verbose: bool = False) -> None:
    """Attach a single stderr handler to the ``EvalRing`` logger.

    Intended for entry points (the ``evalring`` CLI, example scripts). Calling
    it twice is safe: the existing handler is reused rather than duplicated.
    Records go to stderr so that stdout stays free for machine-readable output.

    Args:
        level: Threshold for the EvalRing logger, e.g. ``logging.DEBUG``.
        verbose: When ``True``, prefix records with timestamp, level, and
            logger name instead of emitting the bare message.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    fmt = _VERBOSE_FORMAT if verbose else _CONSOLE_FORMAT
    for handler in logger.handlers:
        if getattr(handler, "_evalring_console", False):
            handler.setLevel(level)
            handler.setFormatter(logging.Formatter(fmt))
            return

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt))
    handler._evalring_console = True  # type: ignore[attr-defined]
    logger.addHandler(handler)
