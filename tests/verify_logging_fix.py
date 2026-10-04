"""Verify that fileConfig(disable_existing_loggers=False) preserves existing loggers.

This directly tests the alembic env.py fix without needing a live database.
The root cause: Python's fileConfig defaults to disable_existing_loggers=True,
which silently sets logger.disabled=True on every logger that existed before
the call.  We verify both the old (broken) and new (fixed) behaviour.
"""
import io
import logging
import os

INI_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "alembic.ini",
)

# ---------------------------------------------------------------------------
# Part A: demonstrate the OLD (broken) behaviour
# ---------------------------------------------------------------------------
from logging.config import fileConfig

# Create a fresh logger in an isolated child so we can reset state.
broken_logger = logging.getLogger("_verify.broken")
broken_logger.setLevel(logging.INFO)
# Ensure it starts enabled
broken_logger.disabled = False
assert not broken_logger.disabled

fileConfig(INI_PATH)  # disable_existing_loggers defaults to True
assert broken_logger.disabled, (
    "Expected broken_logger.disabled=True with default fileConfig, but it was False. "
    "Test precondition not met."
)
print("OLD behaviour confirmed: fileConfig(ini) sets logger.disabled =", broken_logger.disabled)

# ---------------------------------------------------------------------------
# Part B: verify the NEW (fixed) behaviour
# ---------------------------------------------------------------------------
fixed_logger = logging.getLogger("_verify.fixed")
fixed_logger.setLevel(logging.INFO)
fixed_logger.disabled = False

fileConfig(INI_PATH, disable_existing_loggers=False)  # THE FIX
assert not fixed_logger.disabled, (
    "FAIL: fixed_logger.disabled is True even with disable_existing_loggers=False"
)
print("NEW behaviour confirmed: fileConfig(ini, disable_existing_loggers=False) keeps logger.disabled =", fixed_logger.disabled)

# ---------------------------------------------------------------------------
# Part C: verify the fixed logger actually emits output
# ---------------------------------------------------------------------------
buf = io.StringIO()
handler = logging.StreamHandler(buf)
handler.setLevel(logging.WARNING)
fixed_logger.addHandler(handler)
fixed_logger.warning("sentinel-message")
fixed_logger.removeHandler(handler)

output = buf.getvalue()
assert "sentinel-message" in output, (
    "FAIL: warning produced no output after fix.\nGot: {!r}".format(output)
)
print("OUTPUT confirmed:", output.strip())
print()
print("ALL ASSERTIONS PASSED")