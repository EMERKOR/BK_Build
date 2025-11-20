"""
DEPRECATED: This module is deprecated.

All configuration has been moved to ball_knower.config.
Please update your imports to use:
    from ball_knower import config
    # or
    from ball_knower.config import HOME_FIELD_ADVANTAGE, ...

This stub exists for backwards compatibility only.
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "src.config is deprecated. Please use 'from ball_knower import config' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import everything from the new location for backwards compatibility
from ball_knower.config import *  # noqa: F401, F403
