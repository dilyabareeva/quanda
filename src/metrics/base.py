from abc import ABC, abstractmethod


class Metric(ABC):
    def __init__(self, device, *args, **kwargs):
        self.device = device

    @abstractmethod
    def __call__(
        self,
        *args,
        **kwargs,
    ):
        """

        1) Universal assertions about the passed arguments, incl. checking that the length of train/test datset and
        explanations match.
        2) Call the _evaluate method.
        3) Format the output into a unified format for all metrics, possible using some arguments passed in kwargs.


        :param explanations:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def _evaluate_instance(
        self,
        *args,
        **kwargs,
    ):
        """
        Used to implement metric-specific logic.
        """

        raise NotImplementedError
