from typing import Literal

from torchmetrics.functional.regression import (
    kendall_rank_corrcoef,
    spearman_corrcoef,
)


# torchmetrics wants the independent realizations to be the final dimension
# we transpose inputs before passing so that it is straightforward to pass explanations
# and use these funcitons in evaluation metrics
def kendall_rank_corr(tensor1, tensor2):
    return kendall_rank_corrcoef(tensor1.T, tensor2.T)


def spearman_rank_corr(tensor1, tensor2):
    return spearman_corrcoef(tensor1.T, tensor2.T)


CorrelationFnLiterals = Literal["kendall", "spearman"]

correlation_functions = {
    "kendall": kendall_rank_corr,
    "spearman": spearman_rank_corr,
}
