"""Knowledge Graph Builder - Source Package

Central Loguru configuration for the project. Importing any module
from the `src` package will configure Loguru to output DEBUG-level
logs to stderr and to a rotating file under the project `logs/`
directory.
"""

import sys
from pathlib import Path
import logging

from loguru import logger

# Ensure logs directory exists at project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Configure Loguru
logger.remove()
logger.add(
	sys.stderr,
	level="DEBUG",
	format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
	colorize=True,
)
logger.add(
	str(LOG_DIR / "kg_builder.log"),
	level="DEBUG",
	format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
	rotation="5 MB",
	retention="7 days",
)


# Intercept standard library logging and route to Loguru
class _InterceptHandler(logging.Handler):
	def emit(self, record: logging.LogRecord) -> None:
		try:
			level = logger.level(record.levelname).name
		except Exception:
			level = record.levelno

		# Find the first frame outside logging module
		frame = logging.currentframe()
		depth = 2
		while frame and frame.f_code.co_filename == logging.__file__:
			frame = frame.f_back
			depth += 1

		logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


logging.basicConfig(handlers=[_InterceptHandler()], level=0)

# Optionally set asyncio logger level to DEBUG so warnings/errors are captured
logging.getLogger("asyncio").setLevel(logging.DEBUG)

logger.debug(f"Loguru configured — logs dir: {LOG_DIR}")

