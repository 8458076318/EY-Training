import logging
import sys
from config.settings import get_settings

settings = get_settings()


def setup_logging() -> None:
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL),
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    # Suppress noisy libs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
