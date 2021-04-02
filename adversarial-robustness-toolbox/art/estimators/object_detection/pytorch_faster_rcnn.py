# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module implements the task specific estimator for Faster R-CNN v3 in PyTorch.
"""
import logging
from typing import List, Dict, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np

from art.estimators.object_detection.object_detector import ObjectDetectorMixin
from art.estimators.pytorch import PyTorchEstimator

if TYPE_CHECKING:
    # pylint: disable=C0412
    import torch
    import torchvision

    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor.preprocessor import Preprocessor
    from art.defences.postprocessor.postprocessor import Postprocessor

logger = logging.getLogger(__name__)


class PyTorchFasterRCNN(ObjectDetectorMixin, PyTorchEstimator):
    """
    This class implements a model-specific object detector using Faster-RCNN and PyTorch.
    """

    estimator_params = PyTorchEstimator.estimator_params + ["attack_losses"]

    def __init__(
        self,
        model: Optional["torchvision.models.detection.fasterrcnn_resnet50_fpn"] = None,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        channels_first: Optional[bool] = None,
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing: "PREPROCESSING_TYPE" = None,
        attack_losses: Tuple[str, ...] = ("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg",),
        device_type: str = "gpu",
    ):
        """
        Initialization.

        :param model: Faster-RCNN model. The output of the model is `List[Dict[Tensor]]`, one for each input image. The
                      fields of the Dict are as follows:

                      - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values \
                        between 0 and H and 0 and W
                      - labels (Int64Tensor[N]): the predicted labels for each image
                      - scores (Tensor[N]): the scores or each prediction
        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and
               maximum values allowed for features. If floats are provided, these will be used as the range of all
               features. If arrays are provided, each value will be considered the bound for a feature, thus
               the shape of clip values needs to match the total number of features.
        :param channels_first: Set channels first or last.
        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.
        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
        :param preprocessing: Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :param attack_losses: Tuple of any combination of strings of loss components: 'loss_classifier', 'loss_box_reg',
                              'loss_objectness', and 'loss_rpn_box_reg'.
        :param device_type: Type of device to be used for model and tensors, if `cpu` run on CPU, if `gpu` run on GPU
                            if available otherwise run on CPU.
        """
        import torch  # lgtm [py/repeated-import]

        super().__init__(
            model=model,
            clip_values=clip_values,
            channels_first=channels_first,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )

        self._input_shape = None

        if self.clip_values is not None:
            if self.clip_values[0] != 0:
                raise ValueError("This classifier requires un-normalized input images with clip_vales=(0, max_value).")
            if self.clip_values[1] <= 0:
                raise ValueError("This classifier requires un-normalized input images with clip_vales=(0, max_value).")

        if preprocessing is not None:
            raise ValueError("This estimator does not support `preprocessing`.")
        if self.postprocessing_defences is not None:
            raise ValueError("This estimator does not support `postprocessing_defences`.")

        if model is None:
            import torchvision  # lgtm [py/repeated-import]

            self._model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                pretrained=True, progress=True, num_classes=91, pretrained_backbone=True
            )
        else:
            self._model = model

        # Set device
        self._device: torch.device
        if device_type == "cpu" or not torch.cuda.is_available():
            self._device = torch.device("cpu")
        else:
            cuda_idx = torch.cuda.current_device()
            self._device = torch.device("cuda:{}".format(cuda_idx))

        self._model.to(self._device)
        self._model.eval()
        self.attack_losses: Tuple[str, ...] = attack_losses

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """
        Return the shape of one input sample.

        :return: Shape of one input sample.
        """
        return self._input_shape  # type: ignore

    @property
    def device(self) -> "torch.device":
        """
        Get current used device.

        :return: Current used device.
        """
        return self._device

    def loss_gradient(
        self, x: np.ndarray, y: Union[List[Dict[str, np.ndarray]], List[Dict[str, "torch.Tensor"]]], **kwargs
    ) -> np.ndarray:
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Samples of shape (nb_samples, height, width, nb_channels).
        :param y: Target values of format `List[Dict[Tensor]]`, one for each input image. The
                  fields of the Dict are as follows:

                  - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values \
                    between 0 and H and 0 and W
                  - labels (Int64Tensor[N]): the predicted labels for each image
                  - scores (Tensor[N]): the scores or each prediction.
        :return: Loss gradients of the same shape as `x`.
        """
        import torch  # lgtm [py/repeated-import]
        import torchvision  # lgtm [py/repeated-import]

        self._model.train()

        # Apply preprocessing
        if self.all_framework_preprocessing:
            if isinstance(x, torch.Tensor):
                raise NotImplementedError

            if y is not None and isinstance(y[0]["boxes"], np.ndarray):
                y_tensor = list()
                for i, y_i in enumerate(y):
                    y_t = dict()
                    y_t["boxes"] = torch.from_numpy(y_i["boxes"]).type(torch.float).to(self._device)
                    y_t["labels"] = torch.from_numpy(y_i["labels"]).type(torch.int64).to(self._device)
                    y_t["scores"] = torch.from_numpy(y_i["scores"]).to(self._device)
                    y_tensor.append(y_t)
            else:
                y_tensor = y

            transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
            image_tensor_list_grad = list()
            y_preprocessed = list()
            inputs_t = list()

            for i in range(x.shape[0]):
                if self.clip_values is not None:
                    x_grad = transform(x[i] / self.clip_values[1]).to(self._device)
                else:
                    x_grad = transform(x[i]).to(self._device)
                x_grad.requires_grad = True
                image_tensor_list_grad.append(x_grad)
                x_grad_1 = torch.unsqueeze(x_grad, dim=0)
                x_preprocessed_i, y_preprocessed_i = self._apply_preprocessing(
                    x_grad_1, y=[y_tensor[i]], fit=False, no_grad=False
                )
                x_preprocessed_i = torch.squeeze(x_preprocessed_i)
                y_preprocessed.append(y_preprocessed_i[0])
                inputs_t.append(x_preprocessed_i)

        elif isinstance(x, np.ndarray):
            x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y=y, fit=False, no_grad=True)

            if y_preprocessed is not None and isinstance(y_preprocessed[0]["boxes"], np.ndarray):
                y_preprocessed_tensor = list()
                for i, y_i in enumerate(y_preprocessed):
                    y_preprocessed_t = dict()
                    y_preprocessed_t["boxes"] = torch.from_numpy(y_i["boxes"]).type(torch.float).to(self._device)
                    y_preprocessed_t["labels"] = torch.from_numpy(y_i["labels"]).type(torch.int64).to(self._device)
                    y_preprocessed_t["scores"] = torch.from_numpy(y_i["scores"]).to(self._device)
                    y_preprocessed_tensor.append(y_preprocessed_t)
                y_preprocessed = y_preprocessed_tensor

            transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
            image_tensor_list_grad = list()

            for i in range(x_preprocessed.shape[0]):
                if self.clip_values is not None:
                    x_grad = transform(x_preprocessed[i] / self.clip_values[1]).to(self._device)
                else:
                    x_grad = transform(x_preprocessed[i]).to(self._device)
                x_grad.requires_grad = True
                image_tensor_list_grad.append(x_grad)

            inputs_t = image_tensor_list_grad

        else:
            raise NotImplementedError("Combination of inputs and preprocessing not supported.")

        if isinstance(y_preprocessed, np.ndarray):
            labels_t = torch.from_numpy(y_preprocessed).to(self._device)
        else:
            labels_t = y_preprocessed

        output = self._model(inputs_t, labels_t)

        # Compute the gradient and return
        loss = None
        for loss_name in self.attack_losses:
            if loss is None:
                loss = output[loss_name]
            else:
                loss = loss + output[loss_name]

        # Clean gradients
        self._model.zero_grad()

        # Compute gradients
        loss.backward(retain_graph=True)  # type: ignore

        grad_list = list()
        if isinstance(x, np.ndarray):
            for img in image_tensor_list_grad:
                gradients = img.grad.cpu().numpy().copy()
                grad_list.append(gradients)
            grads = np.stack(grad_list, axis=0)
        else:
            for img in inputs_t:
                gradients = img.grad.copy()
                grad_list.append(gradients)
            grads = torch.stack(grad_list, dim=0)

        grads = np.transpose(grads, (0, 2, 3, 1))

        if self.clip_values is not None:
            grads = grads / self.clip_values[1]

        if not self.all_framework_preprocessing:
            grads = self._apply_preprocessing_gradient(x, grads)

        assert grads.shape == x.shape

        return grads

    def predict(self, x: np.ndarray, batch_size: int = 128, **kwargs) -> List[Dict[str, np.ndarray]]:
        """
        Perform prediction for a batch of inputs.

        :param x: Samples of shape (nb_samples, height, width, nb_channels).
        :param batch_size: Batch size.
        :return: Predictions of format `List[Dict[str, np.ndarray]]`, one for each input image. The
                 fields of the Dict are as follows:

                 - boxes [N, 4]: the predicted boxes in [x1, y1, x2, y2] format, with values \
                   between 0 and H and 0 and W
                 - labels [N]: the predicted labels for each image
                 - scores [N]: the scores or each prediction.
        """
        import torchvision  # lgtm [py/repeated-import]

        self._model.eval()

        # Apply preprocessing
        x, _ = self._apply_preprocessing(x, y=None, fit=False)

        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        image_tensor_list: List[np.ndarray] = list()

        if self.clip_values is not None:
            norm_factor = self.clip_values[1]
        else:
            norm_factor = 1.0
        for i in range(x.shape[0]):
            image_tensor_list.append(transform(x[i] / norm_factor).to(self._device))
        predictions = self._model(image_tensor_list)

        for i_prediction, _ in enumerate(predictions):
            predictions[i_prediction]["boxes"] = predictions[i_prediction]["boxes"].detach().cpu().numpy()
            predictions[i_prediction]["labels"] = predictions[i_prediction]["labels"].detach().cpu().numpy()
            predictions[i_prediction]["scores"] = predictions[i_prediction]["scores"].detach().cpu().numpy()

        return predictions

    def fit(self, x: np.ndarray, y, batch_size: int = 128, nb_epochs: int = 20, **kwargs) -> None:
        raise NotImplementedError

    def get_activations(
        self, x: np.ndarray, layer: Union[int, str], batch_size: int, framework: bool = False
    ) -> np.ndarray:
        raise NotImplementedError

    def compute_loss(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute the loss of the neural network for samples `x`.

        :param x: Samples of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,
                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices
                  of shape `(nb_samples,)`.
        :return: Loss values.
        :rtype: Format as expected by the `model`
        """
        raise NotImplementedError
