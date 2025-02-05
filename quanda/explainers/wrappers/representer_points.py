"""Representer Points Explainer Wrapper.

The original code is from the following repository:
https://github.com/chihkuanyeh/Representer_Point_Selection
Unlike other wrappers, this one does not wrap around a Python package. Instead,
we copied large parts of the code and adapted it to our interface. The original
code is licensed under the MIT License.

The original license is included below:

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import logging
import os
import warnings
from functools import reduce
from typing import Any, Callable, List, Optional, Union

import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from captum._utils.av import AV  # type: ignore
from torch import Tensor
from tqdm import tqdm

from quanda.explainers.base import Explainer
from quanda.utils.common import (
    default_tensor_type,
    ds_len,
    map_location_context,
    process_targets,
)
from quanda.utils.tasks import TaskLiterals

logger = logging.getLogger(__name__)


class RepresenterSoftmax(nn.Module):
    """Softmax module for the Representer Points explainer."""

    def __init__(self, W: torch.Tensor, device: Union[torch.device, str]):
        """Initialize the RepresenterSoftmax class.

        Parameters
        ----------
        W : torch.Tensor
            Final linear layer parameters, including biases.
        device : Union[torch.device, str]
            Device to use.

        """
        super(RepresenterSoftmax, self).__init__()
        self.W = nn.Parameter(W.to(device), requires_grad=True)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """Forward pass implementation of the RepresenterSoftmax class.

        Implements final linear layer and softmax.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
        y : torch.Tensor
            Classes for the input tensor.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple of the loss and the L2 regularization term.

        """
        # Compute logits and apply numerical stability trick
        D = x @ self.W
        D = D - D.max(dim=1, keepdim=True).values

        # Negative log likelihood loss (cross-entropy loss equivalent)
        Phi = torch.sum(torch.logsumexp(D, dim=1) - torch.sum(D * y, dim=1))

        # L2 regularization term
        L2 = torch.sum(self.W**2)

        return Phi, L2


def softmax_torch(temp: torch.Tensor, N: int):
    """Torch implementation of the softmax function.

    Parameters
    ----------
    temp : torch.Tensor
        The input tensor.
    N : int
        The number of samples.

    Returns
    -------
    torch.Tensor
        The softmax output.

    """
    max_value, _ = torch.max(temp, 1, keepdim=True)
    temp = temp - max_value
    D_exp = torch.exp(temp)
    D_exp_sum = torch.sum(D_exp, dim=1).view(N, 1)
    return D_exp.div(D_exp_sum.expand_as(D_exp))


def av_samples(av_dataset: AV.AVDataset) -> Tensor:
    """Concatenates the samples of an captum AV dataset.

    Parameters
    ----------
    av_dataset : AV.AVDataset
        The captum AV dataset.

    Returns
    -------
    Tensor
        The concatenated samples.

    """
    warnings.warn(
        "This method is only a good idea for small datasets and small "
        "architectures. Otherwise, this will consume "
        "a lot of memory."
    )
    samples = []

    for i in range(len(av_dataset)):  # type: ignore
        samples.append(av_dataset[i])

    return torch.cat(samples)


class RepresenterPoints(Explainer):
    """Representer Point Explainer, using the official code release [2].

    The method decomposes the pre-activation prediction of a neural network
    into a linear combination of activations from the training points. The
    weights, or representer values, indicate the influence of each training
    point: positive values correspond to excitatory points, while negative
    values correspond to inhibitory points.

    References
    ----------
        (1) Yeh, Chih-Kuan, Kim, Joon, Yen, Ian En-Hsu, Ravikumar, Pradeep K.
        (2018). "Representer Point Selection for Explaining Deep Neural
        Networks." Advances in Neural Information Processing Systems, vol. 31.

        (2) https://github.com/chihkuanyeh/Representer_Point_Selection

    """

    accepted_tasks: List[TaskLiterals] = ["image_classification"]

    def __init__(
        self,
        model: Union[torch.nn.Module, L.LightningModule],
        train_dataset: torch.utils.data.Dataset,
        model_id: str,
        features_layer: str,
        classifier_layer: str,
        task: TaskLiterals = "image_classification",
        checkpoints: Optional[Union[str, List[str]]] = None,
        checkpoints_load_func: Optional[Callable[..., Any]] = None,
        cache_dir: str = "./cache",
        features_postprocess: Optional[Callable] = None,
        lmbd: float = 0.003,
        epoch: int = 3000,
        lr: float = 3e-4,
        min_loss: float = 10000.0,
        epsilon: float = 1e-10,
        normalize: bool = False,
        batch_size: int = 32,
        load_from_disk: bool = True,
        show_progress: bool = True,
    ):
        """Initialize the RepresenterPoints class.

        Parameters
        ----------
        model : Union[torch.nn.Module, L.LightningModule]
            The model to be explained.
        model_id : str
            The model identifier.
        train_dataset : torch.utils.data.Dataset
            The training dataset used to train the model.
        features_layer : str
            The name of the penuultimate layer of the model.
        classifier_layer : str
            The name of the final classifier layer of the model.
        task: TaskLiterals, optional
            Task type of the model. Defaults to "image_classification".
            Possible options: "image_classification", "text_classification",
            "causal_lm".
        checkpoints : Optional[Union[str, List[str]]], optional
            Path to the model checkpoint file(s), defaults to None.
        checkpoints_load_func : Optional[Callable[..., Any]], optional
            Function to load the model from the checkpoint file, takes
            (model, checkpoint path) as two arguments, by default None.
        cache_dir : str, optional
            The directory to save the cache, defaults to "./cache".
        features_postprocess : Optional[Callable], optional
            A postprocessing function for the features, defaults to None.
        lmbd : float, optional
            Regularization constant, defaults to 0.003.
        epoch : int, optional
            Number of epochs, defaults to 3000.
        lr : float, optional
            Learning rate, defaults to 3e-4.
        min_loss : float, optional
            Initial minimum loss value to start training loop, defaults to
            10000.0.
        epsilon : float, optional
            Epsilon value for backtracking line search, defaults to 1e-10.
        normalize : bool, optional
            Whether to normalize the features, defaults to False.
        batch_size : int, optional
            Batch size for training, defaults to 32.
        load_from_disk : bool, optional
            Whether to load the activations from disk, defaults to True.
        show_progress : bool, optional
            Whether to show the training progress, defaults to True.

        """
        logger.info("Initializing Representer Point Selection explainer...")
        super(RepresenterPoints, self).__init__(
            model=model,
            train_dataset=train_dataset,
            task=task,
            checkpoints=checkpoints,
            checkpoints_load_func=checkpoints_load_func,
        )

        self.model_id = model_id
        self.cache_dir = cache_dir
        self.normalize = normalize
        self.features_layer = features_layer
        self.classifier_layer = classifier_layer
        self.lmbd = lmbd
        self.epoch = epoch
        self.lr = lr
        self.min_loss: Any = min_loss
        self.epsilon = epsilon
        self.features_postprocess = features_postprocess
        self.show_progress = show_progress

        self.dataloader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=False
        )

        with default_tensor_type(self.device):
            act_dataset = AV.generate_dataset_activations(
                path=cache_dir,
                model=model,
                model_id=model_id,
                layers=[features_layer],
                dataloader=self.dataloader,
                load_from_disk=load_from_disk,
                return_activations=True,
            )[0]

        self.current_acts: torch.Tensor
        self.learned_weights: torch.Tensor
        self.coefficients: torch.Tensor

        with (
            map_location_context(self.device),
            default_tensor_type(self.device),
        ):
            self.samples = av_samples(act_dataset)

        if self.features_postprocess is not None:
            self.samples = self.features_postprocess(self.samples)

        self.labels = torch.tensor(
            [train_dataset[i][1] for i in range(ds_len(self.train_dataset))],
            device=self.device,
        ).type(torch.int)

        self.mean = self.samples.mean(dim=0)
        self.std_dev = torch.sqrt(
            torch.sum((self.samples - self.mean) ** 2, dim=0)
            / self.samples.shape[0]
        )
        self.samples = (
            self._normalize_features(self.samples)
            if normalize
            else self.samples
        )

        if load_from_disk:
            try:
                self.coefficients = torch.load(
                    os.path.join(
                        self.cache_dir, f"{self.model_id}_repr_weights.pt"
                    ),
                    weights_only=True,
                )
            except FileNotFoundError:
                self.train()
        else:
            self.train()

    def _normalize_features(self, features: torch.Tensor):
        """Normalize the features.

        Parameters
        ----------
        features : torch.Tensor
            The input features.

        Returns
        -------
        torch.Tensor
            The normalized features

        """
        return (features - self.mean) / self.std_dev

    def _get_activations(self, x: torch.Tensor, layer: str) -> torch.Tensor:
        """Returnthe activations of a specific layer for a given input batch.

        Parameters
        ----------
        x : torch.Tensor
            The input batch of data.
        layer : str
            The layer for which the activations are returned.

        Returns
        -------
        torch.Tensor
            The activations of the specified layer.

        """

        # Define a hook function to store the activations
        def hook_fn(module, input, output):
            self.current_acts = output

        # Register the hook to the specified layer
        target_layer = dict([*self.model.named_modules()])[layer]
        hook_handle = target_layer.register_forward_hook(hook_fn)

        # Forward pass
        self.model(x)

        # Remove the hook
        hook_handle.remove()

        return self.current_acts

    def explain(
        self,
        test_data: torch.Tensor,
        targets: Union[List[int], torch.Tensor],
    ) -> torch.Tensor:
        """Explain the predictions of the model for a given test batch.

        Parameters
        ----------
        test_data : torch.Tensor
            The test batch for which explanations are generated.
        targets : Union[List[int], torch.Tensor]
            The target values for the explanations.

        Returns
        -------
        torch.Tensor
            The explanations for the test batch.

        """
        test_data = test_data.to(self.device)
        targets = process_targets(targets, self.device)

        f = self._get_activations(test_data, self.features_layer)

        if self.features_postprocess is not None:
            f = self.features_postprocess(f)

        if self.normalize:
            f = self._normalize_features(f)

        cross_corr = torch.einsum("ik,jk->ij", f, self.samples).unsqueeze(-1)

        explanations = self.coefficients * cross_corr

        indices = targets[:, None, None].expand(-1, self.samples.shape[0], 1)
        explanations = torch.gather(explanations, dim=-1, index=indices)
        return torch.squeeze(explanations)

    def train(self):
        """Train the model to obtain the representer point coefficients.

        Raises
        ------
        ValueError
            If the gradient of the final layer parameters can not be obtained.

        """
        samples_with_bias = torch.cat(
            [
                self.samples,
                torch.ones((self.samples.shape[0], 1), device=self.device),
            ],
            dim=1,
        )
        linear_classifier = reduce(
            getattr, self.classifier_layer.split("."), self.model
        )
        logits = linear_classifier(self.samples)
        labels = softmax_torch(logits, self.samples.shape[0])

        weight_linear, bias_linear = (
            linear_classifier.weight.data,
            linear_classifier.bias.data,
        )
        w_and_b = torch.concatenate(
            [weight_linear.T, bias_linear.unsqueeze(0)]
        )
        model = RepresenterSoftmax(w_and_b, self.device)

        x = nn.Parameter(samples_with_bias.to(self.device))
        y = nn.Parameter(labels.to(self.device))

        N = len(labels)
        min_loss = self.min_loss
        optimizer = optim.SGD([model.W], lr=self.lr)
        if self.show_progress:
            pbar = tqdm(
                range(self.epoch),
                desc="Representer Training | Epoch: 0 | Loss: 0 | Phi Loss: "
                "0 | Grad: 0",
            )

        for epoch in range(self.epoch):
            phi_loss = 0
            optimizer.zero_grad()
            (Phi, L2) = model(x, y)
            loss = L2 * self.lmbd + Phi / N
            phi_loss += (Phi / N).detach().cpu().numpy()
            loss.backward()
            temp_W = model.W.data

            if model.W.grad is None:
                raise ValueError("Gradient is None")

            grad_loss = (
                torch.mean(torch.abs(model.W.grad)).detach().cpu().numpy()
            )

            # save the W with lowest loss
            if grad_loss < min_loss:
                if epoch == 0:
                    init_grad = grad_loss
                min_loss = grad_loss
                best_W = temp_W
                if min_loss < init_grad / 200:
                    logger.info(
                        "Stopping criteria reached in epoch :{}".format(epoch)
                    )
                    break
            self.backtracking_line_search(model, model.W.grad, x, y, loss, N)
            if self.show_progress:
                pbar.set_description(
                    f"Representer Training | Epoch: {epoch:4d} | Loss: "
                    f"{loss.detach().cpu().numpy():.4f} |"
                    f" Phi Loss: {phi_loss:.4f} | Grad: {grad_loss:.4f}"
                )

                pbar.update(1)

        # calculate w based on the representer theorem's decomposition
        temp = torch.matmul(
            x, nn.Parameter(best_W.to(self.device), requires_grad=True)
        )
        self.learned_weight = best_W.T
        softmax_value = softmax_torch(temp, N)
        # derivative of softmax cross entropy
        weight_matrix = softmax_value - y
        weight_matrix = torch.div(weight_matrix, (-2.0 * self.lmbd * N))

        self.coefficients = weight_matrix

        # save weight matrix to cache
        torch.save(
            weight_matrix,
            os.path.join(self.cache_dir, f"{self.model_id}_repr_weights.pt"),
        )

    def backtracking_line_search(
        self,
        model: torch.nn.Module,
        grad: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        val: torch.Tensor,
        N: int,
    ):
        """Perform the backtracking line search algorithm.

        Parameters
        ----------
        model : torch.nn.Module
            The model to be trained.
        grad : torch.Tensor
            The gradient of the model.
        x : torch.Tensor
            The input tensor.
        y : torch.Tensor
            The target tensor.
        val : torch.Tensor
            The loss value.
        N : int
            The number of samples.

        """
        t = 10.0
        beta = 0.5
        W_O = model.W.detach().cpu().numpy()
        grad_np = grad.detach().cpu().numpy()

        while True:
            model.W = nn.Parameter(
                torch.from_numpy(W_O - t * grad_np).to(self.device),
                requires_grad=True,
            )
            (Phi, L2) = model(x, y)
            val_n = Phi / N + L2 * self.lmbd
            if t < self.epsilon:
                break
            if (
                val_n - val + t * torch.norm(grad) ** 2 / 2
            ).detach().cpu().numpy() >= 0:
                t = beta * t
            else:
                break

    def self_influence(self, batch_size: int = 32) -> torch.Tensor:
        """Calculate self-influence, i.e. the global attribution values.

        For representer points, we define the self-influence as the
        coefficients of the representer points, as per Sec. 4.1 of the original
        paper (Yeh et al., 2018).

        Returns
        -------
        torch.Tensor
            The self-influence (global attribution) values for the representer
            points. Used in metrics that require a global ranking, e.g.
            MislabelingDetecion.

        """
        # coefficients for each training label
        return self.coefficients[
            torch.arange(self.coefficients.shape[0]), self.labels
        ]
