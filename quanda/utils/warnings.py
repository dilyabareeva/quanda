"""Warning categories used throughout quanda.

Quanda separates its warnings into two categories so that users can
control them with a single ``warnings.filterwarnings`` call:

- :class:`QuandaAdvisoryWarning`: advisory messages about expected,
  non-critical behavior (e.g. ignored unsupported arguments, performance
  fallbacks, files being overwritten). Safe to silence::

      import warnings
      from quanda.utils.warnings import QuandaAdvisoryWarning

      warnings.filterwarnings("ignore", category=QuandaAdvisoryWarning)

- :class:`QuandaCriticalWarning`: warnings about conditions that can
  silently produce incorrect results (e.g. mismatched explanation
  caches, checkpoints being ignored). These should not be silenced;
  instead they can be escalated to errors::

      import warnings
      from quanda.utils.warnings import QuandaCriticalWarning

      warnings.filterwarnings("error", category=QuandaCriticalWarning)

Both inherit from :class:`QuandaWarning`, which can be used to filter
all quanda warnings at once.
"""


class QuandaWarning(UserWarning):
    """Base class for all warnings issued by quanda."""


class QuandaAdvisoryWarning(QuandaWarning):
    """Advisory message about expected, non-critical behavior.

    Safe to silence with
    ``warnings.filterwarnings("ignore", category=QuandaAdvisoryWarning)``.
    """


class QuandaCriticalWarning(QuandaWarning):
    """Warning about a condition that may silently corrupt results.

    Should not be silenced; can be escalated to an error with
    ``warnings.filterwarnings("error", category=QuandaCriticalWarning)``.
    """
