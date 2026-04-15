import pytest
import torch

from quanda.explainers.global_ranking import SelfInfluenceRanking


@pytest.mark.global_ranking
@pytest.mark.parametrize(
    "scenario",
    [
        "init_missing_both",
        "precomputed",
        "explainer_call",
        "get_si_without_source",
        "warning_methods",
    ],
)
def test_self_influence_ranking_paths(scenario, mocker):
    """Cover init error, both get_self_influence branches, and the
    unsupported-method warnings emitted by reset/state_dict/load_state_dict."""
    if scenario == "init_missing_both":
        with pytest.raises(ValueError, match="required"):
            SelfInfluenceRanking()
        return

    if scenario == "precomputed":
        si = torch.tensor([2.0, 0.0, 1.0])
        ranker = SelfInfluenceRanking(self_influence=si)
        assert torch.equal(ranker.get_self_influence(), si)
        # global rank stable-sorts by (si + tiny idx bias)
        assert torch.equal(ranker.get_global_rank(), torch.tensor([1, 2, 0]))
        return

    if scenario == "explainer_call":
        expl = mocker.MagicMock()
        expl.self_influence.return_value = torch.tensor([0.5, 0.25])
        ranker = SelfInfluenceRanking(explainer=expl)
        out = ranker.get_self_influence()
        assert torch.equal(out, torch.tensor([0.5, 0.25]))
        expl.self_influence.assert_called_once()
        return

    if scenario == "get_si_without_source":
        # Force both sources to None post-init to exercise the final
        # ValueError in get_self_influence.
        ranker = SelfInfluenceRanking(self_influence=torch.zeros(1))
        ranker._self_influence = None
        ranker.explainer = None
        with pytest.raises(ValueError, match="Cannot compute self-influence"):
            ranker.get_self_influence()
        return

    if scenario == "warning_methods":
        ranker = SelfInfluenceRanking(self_influence=torch.zeros(1))
        for name in ("reset", "load_state_dict", "state_dict"):
            with pytest.warns(UserWarning, match=name):
                getattr(ranker, name)(
                    {} if name == "load_state_dict" else None
                )
        return
