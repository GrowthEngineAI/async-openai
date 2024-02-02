import os
from lazyops.utils.logs import get_logger, change_logger_level, null_logger

# to prevent recursive imports, we'll just use os environ here
if os.getenv('DEBUG_ENABLED') == 'True':
    logger_level = 'DEV'
else:
    logger_level: str = os.getenv('LOGGER_LEVEL', 'INFO').upper()

logger = get_logger(logger_level)