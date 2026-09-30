import warnings

import pytest

from quanda.explainers.global_ranking.self_influence_ranking import (
    SelfInfluenceRanking,
)
from quanda.utils.warnings import (
    QuandaAdvisoryWarning,
    QuandaCriticalWarning,
    QuandaWarning,
)


@pytest.mark.utils
def test_warning_hierarchy():
    """Both categories are QuandaWarnings and UserWarnings."""
    assert issubclass(QuandaAdvisoryWarning, QuandaWarning)
    assert issubclass(QuandaCriticalWarning, QuandaWarning)
    assert issubclass(QuandaWarning, UserWarning)
    assert not issubclass(QuandaAdvisoryWarning, QuandaCriticalWarning)
    assert not issubclass(QuandaCriticalWarning, QuandaAdvisoryWarning)


@pytest.mark.utils
def test_advisory_silenced_critical_escalated():
    """A single filter call silences advisory or escalates critical."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=QuandaAdvisoryWarning)
        warnings.warn("advisory", QuandaAdvisoryWarning)
        warnings.warn("critical", QuandaCriticalWarning)
    assert len(caught) == 1
    assert caught[0].category is QuandaCriticalWarning

    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=QuandaCriticalWarning)
        warnings.warn("advisory", QuandaAdvisoryWarning)
        with pytest.raises(QuandaCriticalWarning):
            warnings.warn("critical", QuandaCriticalWarning)


@pytest.mark.utils
def test_library_warning_uses_category():
    """Warning sites in the library emit quanda categories."""
    with pytest.warns(QuandaAdvisoryWarning):
        SelfInfluenceRanking._si_warning("reset")
