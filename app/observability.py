"""
Privacy-first Rius observability and tracing for the Atlas application.

Telemetry is enabled only when RIUS_API_KEY is configured and the
RIUS_DISABLED kill switch is not enabled.

Input arguments and function outputs are never captured.
"""

import os
import logging
from functools import wraps

log = logging.getLogger(__name__)


def initialize():
    """
    Initialise Rius observability for the application.

    Call this function once during application startup, before importing
    modules containing functions decorated with @trace.

    Behaviour:
        1. Loads environment variables from the .env file.
        2. Checks whether RIUS_API_KEY is configured.
        3. Checks whether observability has been explicitly disabled.
        4. Initialises Rius with content capture disabled.

    Environment variables:
        RIUS_API_KEY:
            API key required to enable Rius observability.

        RIUS_DISABLED:
            Optional kill switch. Set to 'true', '1', or 'yes'
            to disable observability.

        RIUS_SERVICE_NAME:
            Optional service name used to identify the application
            in the Rius dashboard.

    Notes:
        - Telemetry remains disabled if no API key is configured.
        - Rius initialisation failures do not prevent the application
          from starting.
        - Function arguments and return values are not captured.
    """

    from dotenv import load_dotenv

    # Load environment variables from the project's .env file.
    load_dotenv()

    # Check whether observability is explicitly disabled.
    disabled = os.getenv("RIUS_DISABLED", "false").lower() in (
        "1", "true", "yes"
    )

    # Enable observability only when an API key exists and
    # the kill switch is not enabled.
    if not os.getenv("RIUS_API_KEY") or disabled:
        log.info("Rius observability disabled (no key or kill switch)")
        return

    try:
        import rius

        # Initialise Rius without capturing sensitive application content.
        rius.init(
            service_name=os.getenv("RIUS_SERVICE_NAME", "atlas-ai-fde"),
            capture_content=False
        )

        log.info("Rius observability enabled")

    except Exception:
        # Observability failures should never interrupt application startup.
        log.exception(
            "Rius initialization failed; application will continue "
            "without telemetry"
        )


def trace(name):
    """
    Decorator factory for adding Rius tracing to application functions.

    Supports synchronous and asynchronous functions, depending on the
    capabilities of the installed Rius SDK.

    Args:
        name (str):
            Name assigned to the trace for identification in Rius.

    Returns:
        Callable:
            A decorator that instruments the target function.

    Privacy:
        - Function arguments are not captured.
        - Function return values are not captured.
        - If the Rius package is unavailable, the original function
          is returned without instrumentation.

    Example:
        @trace("embed")
        def embed(text):
            return embedding

        @trace("answer")
        def answer(question, evidence, history=None):
            return response
    """

    def decorate(fn):
        try:
            import rius

            # Instrument the function without capturing its input or output.
            return rius.observe(
                name=name,
                capture_input=False,
                capture_output=False
            )(fn)

        except ImportError:
            # Allow the application to run even if Rius is not installed.
            return fn

    return decorate