"""Wrapping explainers."""

from quanda.explainers.wrappers.captum_influence import (
    CaptumArnoldi,
    CaptumInfluence,
    CaptumSimilarity,
    CaptumTracInCP,
    CaptumTracInCPFast,
    CaptumTracInCPFastRandProj,
    captum_arnoldi_explain,
    captum_arnoldi_self_influence,
    captum_similarity_explain,
    captum_similarity_self_influence,
    captum_tracincp_explain,
    captum_tracincp_fast_explain,
    captum_tracincp_fast_rand_proj_explain,
    captum_tracincp_fast_rand_proj_self_influence,
    captum_tracincp_fast_self_influence,
    captum_tracincp_self_influence,
)
from quanda.explainers.wrappers.kronfluence import (
    Kronfluence,
    kronfluence_explain,
    kronfluence_self_influence,
)
from quanda.explainers.wrappers.representer_points import RepresenterPoints
from quanda.explainers.wrappers.trak_wrapper import (
    TRAK,
    trak_explain,
    trak_self_influence,
)

__all__ = [
    "CaptumInfluence",
    "CaptumSimilarity",
    "captum_similarity_explain",
    "captum_similarity_self_influence",
    "CaptumArnoldi",
    "captum_arnoldi_explain",
    "captum_arnoldi_self_influence",
    "CaptumTracInCP",
    "captum_tracincp_explain",
    "captum_tracincp_self_influence",
    "TRAK",
    "trak_explain",
    "trak_self_influence",
    "CaptumTracInCPFast",
    "captum_tracincp_fast_explain",
    "captum_tracincp_fast_self_influence",
    "CaptumTracInCPFastRandProj",
    "captum_tracincp_fast_rand_proj_explain",
    "captum_tracincp_fast_rand_proj_self_influence",
    "RepresenterPoints",
    "Kronfluence",
    "kronfluence_explain",
    "kronfluence_self_influence",
]
