"""Stable error codes for fit adapter and pipeline exceptions."""

from __future__ import annotations


# -- Error codes -------------------------------------------------------------

ADAPTER_AUTH = "FIT_ADAPTER_AUTH"
ADAPTER_RATE = "FIT_ADAPTER_RATE"
ADAPTER_MODEL = "FIT_ADAPTER_MODEL"
ADAPTER_TIMEOUT = "FIT_ADAPTER_TIMEOUT"
TRAINING_DATA = "FIT_TRAINING_DATA"
EXPORT_DEP = "FIT_EXPORT_DEP"


# -- Base exception ----------------------------------------------------------


class FitError(Exception):
    """Structured error with a stable code, optional cause, fix hint,
    and retryable flag.

    Subclasses ``Exception`` so existing ``except Exception`` blocks
    continue to catch these transparently.
    """

    def __init__(
        self,
        code: str,
        message: str,
        cause: str = "",
        fix: str = "",
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.cause = cause
        self.fix = fix
        self.retryable = retryable
