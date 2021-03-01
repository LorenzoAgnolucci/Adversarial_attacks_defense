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
This module implements a variation of the adversarial patch attack `DPatch` for object detectors.
It follows Lee & Kolter (2019) in using sign gradients with expectations over transformations.
The particular transformations supported in this implementation are cropping, rotations by multiples of 90 degrees,
and changes in the brightness of the image.

| Paper link (original DPatch): https://arxiv.org/abs/1806.02299v4
| Paper link (physical-world patch from Lee & Kolter): https://arxiv.org/abs/1906.11897
"""
import logging
import math
import random
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
from tqdm.auto import trange

from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.estimators.object_detection.object_detector import ObjectDetectorMixin
from art import config

if TYPE_CHECKING:
    from art.utils import OBJECT_DETECTOR_TYPE

logger = logging.getLogger(__name__)


class RobustDPatch(EvasionAttack):
    """
    Implementation of a particular variation of the DPatch attack.
    It follows Lee & Kolter (2019) in using sign gradients with expectations over transformations.
    The particular transformations supported in this implementation are cropping, rotations by multiples of 90 degrees,
    and changes in the brightness of the image.

    | Paper link (original DPatch): https://arxiv.org/abs/1806.02299v4
    | Paper link (physical-world patch from Lee & Kolter): https://arxiv.org/abs/1906.11897
    """

    attack_params = EvasionAttack.attack_params + [
        "patch_shape",
        "learning_rate",
        "max_iter",
        "batch_size",
        "verbose",
        "patch_location",
        "crop_range",
        "brightness_range",
        "rotation_weights",
        "sample_size",
    ]

    _estimator_requirements = (BaseEstimator, LossGradientsMixin, ObjectDetectorMixin)

    def __init__(
        self,
        estimator: "OBJECT_DETECTOR_TYPE",
        patch_shape: Tuple[int, int, int] = (40, 40, 3),
        patch_location: Tuple[int, int] = (0, 0),
        crop_range: Tuple[int, int] = (0, 0),
        brightness_range: Tuple[float, float] = (1.0, 1.0),
        rotation_weights: Union[Tuple[float, float, float, float], Tuple[int, int, int, int]] = (1, 0, 0, 0),
        sample_size: int = 1,
        learning_rate: float = 5.0,
        max_iter: int = 500,
        batch_size: int = 16,
        verbose: bool = True,
    ):
        """
        Create an instance of the :class:`.RobustDPatch`.

        :param estimator: A trained object detector.
        :param patch_shape: The shape of the adversarial patch as a tuple of shape (height, width, nb_channels).
        :param patch_location: The location of the adversarial patch as a tuple of shape (upper left x, upper left y).
        :param crop_range: By how much the images may be cropped as a tuple of shape (height, width).
        :param brightness_range: Range for randomly adjusting the brightness of the image.
        :param rotation_weights: Sampling weights for random image rotations by (0, 90, 180, 270) degrees clockwise.
        :param sample_size: Number of samples to be used in expectations over transformation.
        :param learning_rate: The learning rate of the optimization.
        :param max_iter: The number of optimization steps.
        :param batch_size: The size of the training batch.
        :param verbose: Show progress bars.
        """

        super().__init__(estimator=estimator)

        self.patch_shape = patch_shape
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        if self.estimator.clip_values is None:
            self._patch = np.zeros(shape=patch_shape, dtype=config.ART_NUMPY_DTYPE)
        else:
            self._patch = (
                np.random.randint(0, 255, size=patch_shape)
                / 255
                * (self.estimator.clip_values[1] - self.estimator.clip_values[0])
                + self.estimator.clip_values[0]
            ).astype(config.ART_NUMPY_DTYPE)
        self.verbose = verbose
        self.patch_location = patch_location
        self.crop_range = crop_range
        self.brightness_range = brightness_range
        self.rotation_weights = rotation_weights
        self.sample_size = sample_size
        self._check_params()

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate RobustDPatch.

        :param x: Sample images.
        :param y: Target labels for object detector.
        :return: Adversarial patch.
        """
        channel_index = 1 if self.estimator.channels_first else x.ndim - 1
        if x.shape[channel_index] != self.patch_shape[channel_index - 1]:
            raise ValueError("The color channel index of the images and the patch have to be identical.")
        if y is not None:
            raise ValueError("The RobustDPatch attack does not use target labels.")
        if x.ndim != 4:
            raise ValueError("The adversarial patch can only be applied to images.")

        # Check whether patch fits into the cropped images:
        if self.estimator.channels_first:
            image_height, image_width = x.shape[2:4]
        else:
            image_height, image_width = x.shape[1:3]

        if (
            self.patch_location[0] + self.patch_shape[0] > image_height - self.crop_range[0]
            or self.patch_location[1] + self.patch_shape[1] > image_width - self.crop_range[1]
        ):
            raise ValueError("The patch (partially) lies outside the cropped image.")

        for i_step in trange(self.max_iter, desc="RobustDPatch iteration", disable=not self.verbose):
            if i_step == 0 or (i_step + 1) % 100 == 0:
                logger.info("Training Step: %i", i_step + 1)

            num_batches = math.ceil(x.shape[0] / self.batch_size)
            patch_gradients_old = np.zeros_like(self._patch)

            for e_step in range(self.sample_size):
                if e_step == 0 or (e_step + 1) % 100 == 0:
                    logger.info("EOT Step: %i", e_step + 1)

                for i_batch in range(num_batches):
                    i_batch_start = i_batch * self.batch_size
                    i_batch_end = min((i_batch + 1) * self.batch_size, x.shape[0])

                    # Sample and apply the random transformations:
                    patched_images, patch_target, transforms = self._augment_images_with_patch(
                        x[i_batch_start:i_batch_end], self._patch, channels_first=self.estimator.channels_first
                    )

                    gradients = self.estimator.loss_gradient(x=patched_images, y=patch_target,)

                    gradients = self._untransform_gradients(
                        gradients, transforms, channels_first=self.estimator.channels_first
                    )

                    patch_gradients = patch_gradients_old + np.sum(gradients, axis=0)
                    logger.debug(
                        "Gradient percentage diff: %f)",
                        np.mean(np.sign(patch_gradients) != np.sign(patch_gradients_old)),
                    )

                    patch_gradients_old = patch_gradients

            self._patch = self._patch + np.sign(patch_gradients) * self.learning_rate

            if self.estimator.clip_values is not None:
                self._patch = np.clip(
                    self._patch, a_min=self.estimator.clip_values[0], a_max=self.estimator.clip_values[1],
                )

        return self._patch

    def _augment_images_with_patch(
        self, x: np.ndarray, patch: np.ndarray, channels_first: bool
    ) -> Tuple[np.ndarray, List[Dict[str, np.ndarray]], Dict[str, Union[int, float]]]:
        """
        Augment images with patch.

        :param x: Sample images.
        :param patch: The patch to be applied.
        :param channels_first: Set channels first or last.
        """

        transformations: Dict[str, Union[float, int]] = dict()
        x_copy = x.copy()
        patch_copy = patch.copy()
        x_patch = x.copy()

        if channels_first:
            x_copy = np.transpose(x_copy, (0, 2, 3, 1))
            x_patch = np.transpose(x_patch, (0, 2, 3, 1))
            patch_copy = np.transpose(patch_copy, (1, 2, 0))

        # Apply patch:
        x_1, y_1 = self.patch_location
        x_2, y_2 = x_1 + patch_copy.shape[0], y_1 + patch_copy.shape[1]
        x_patch[:, x_1:x_2, y_1:y_2, :] = patch_copy

        # 1) crop images:
        crop_x = random.randint(0, self.crop_range[0])
        crop_y = random.randint(0, self.crop_range[1])
        x_1, y_1 = crop_x, crop_y
        x_2, y_2 = x_copy.shape[1] - crop_x + 1, x_copy.shape[2] - crop_y + 1
        x_copy = x_copy[:, x_1:x_2, y_1:y_2, :]
        x_patch = x_patch[:, x_1:x_2, y_1:y_2, :]

        transformations.update({"crop_x": crop_x, "crop_y": crop_y})

        # 2) rotate images:
        rot90 = random.choices([0, 1, 2, 3], weights=self.rotation_weights)[0]

        x_copy = np.rot90(x_copy, rot90, (1, 2))
        x_patch = np.rot90(x_patch, rot90, (1, 2))

        transformations.update({"rot90": rot90})

        # 3) adjust brightness:
        brightness = random.uniform(*self.brightness_range)
        x_copy = np.round(brightness * x_copy)
        x_patch = np.round(brightness * x_patch)

        transformations.update({"brightness": brightness})

        logger.debug("Transformations: %s", str(transformations))

        patch_target: List[Dict[str, np.ndarray]] = list()
        predictions = self.estimator.predict(x=x_copy)
        for i_image in range(x_copy.shape[0]):
            target_dict = dict()
            target_dict["boxes"] = predictions[i_image]["boxes"]
            target_dict["labels"] = predictions[i_image]["labels"]
            target_dict["scores"] = predictions[i_image]["scores"]

            patch_target.append(target_dict)

        if channels_first:
            x_patch = np.transpose(x_patch, (0, 3, 1, 2))

        return x_patch, patch_target, transformations

    def _untransform_gradients(
        self, gradients: np.ndarray, transforms: Dict[str, Union[int, float]], channels_first: bool,
    ) -> np.ndarray:
        """
        Revert transformation on gradients.

        :param gradients: The gradients to be reverse transformed.
        :param transforms: The transformations in forward direction.
        :param channels_first: Set channels first or last.
        """

        if channels_first:
            gradients = np.transpose(gradients, (0, 2, 3, 1))

        # Account for brightness adjustment:
        gradients = transforms["brightness"] * gradients

        # Undo rotations:
        rot90 = (4 - transforms["rot90"]) % 4
        gradients = np.rot90(gradients, rot90, (1, 2))

        # Account for cropping when considering the upper left point of the patch:
        x_1 = self.patch_location[0] - int(transforms["crop_x"])
        y_1 = self.patch_location[1] - int(transforms["crop_y"])
        x_2 = x_1 + self.patch_shape[0]
        y_2 = y_1 + self.patch_shape[1]
        gradients = gradients[:, x_1:x_2, y_1:y_2, :]

        if channels_first:
            gradients = np.transpose(gradients, (0, 3, 1, 2))

        return gradients

    def apply_patch(self, x: np.ndarray, patch_external: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply the adversarial patch to images.

        :param x: Images to be patched.
        :param patch_external: External patch to apply to images `x`. If None the attacks patch will be applied.
        :return: The patched images.
        """

        x_patch = x.copy()

        if patch_external is not None:
            patch_local = patch_external.copy()
        else:
            patch_local = self._patch.copy()

        if self.estimator.channels_first:
            x_patch = np.transpose(x_patch, (0, 2, 3, 1))
            patch_local = np.transpose(patch_local, (1, 2, 0))

        # Apply patch:
        x_1, y_1 = self.patch_location
        x_2, y_2 = x_1 + patch_local.shape[0], y_1 + patch_local.shape[1]

        if x_2 > x_patch.shape[1] or y_2 > x_patch.shape[2]:
            raise ValueError("The patch (partially) lies outside the image.")

        x_patch[:, x_1:x_2, y_1:y_2, :] = patch_local

        if self.estimator.channels_first:
            x_patch = np.transpose(x_patch, (0, 3, 1, 2))

        return x_patch

    def _check_params(self) -> None:
        if not isinstance(self.patch_shape, (tuple, list)) or not all(isinstance(s, int) for s in self.patch_shape):
            raise ValueError("The patch shape must be either a tuple or list of integers.")
        if len(self.patch_shape) != 3:
            raise ValueError("The length of patch shape must be 3.")

        if not isinstance(self.learning_rate, float):
            raise ValueError("The learning rate must be of type float.")
        if self.learning_rate <= 0.0:
            raise ValueError("The learning rate must be greater than 0.0.")

        if not isinstance(self.max_iter, int):
            raise ValueError("The number of optimization steps must be of type int.")
        if self.max_iter <= 0:
            raise ValueError("The number of optimization steps must be greater than 0.")

        if not isinstance(self.batch_size, int):
            raise ValueError("The batch size must be of type int.")
        if self.batch_size <= 0:
            raise ValueError("The batch size must be greater than 0.")

        if not isinstance(self.verbose, bool):
            raise ValueError("The argument `verbose` has to be of type bool.")

        if not isinstance(self.patch_location, (tuple, list)) or not all(
            isinstance(s, int) for s in self.patch_location
        ):
            raise ValueError("The patch location must be either a tuple or list of integers.")
        if len(self.patch_location) != 2:
            raise ValueError("The length of patch location must be 2.")

        if not isinstance(self.crop_range, (tuple, list)) or not all(isinstance(s, int) for s in self.crop_range):
            raise ValueError("The crop range must be either a tuple or list of integers.")
        if len(self.crop_range) != 2:
            raise ValueError("The length of crop range must be 2.")

        if self.crop_range[0] > self.crop_range[1]:
            raise ValueError("The first element of the crop range must be less or equal to the second one.")

        if self.patch_location[0] < self.crop_range[0] or self.patch_location[1] < self.crop_range[1]:
            raise ValueError("The patch location must be outside the crop range.")

        if not isinstance(self.brightness_range, (tuple, list)) or not all(
            isinstance(s, float) for s in self.brightness_range
        ):
            raise ValueError("The brightness range must be either a tuple or list of floats.")
        if len(self.brightness_range) != 2:
            raise ValueError("The length of brightness range must be 2.")

        if self.brightness_range[0] < 0.0 or self.brightness_range[1] > 1.0:
            raise ValueError("The brightness range must be between 0.0 and 1.0.")

        if self.brightness_range[0] > self.brightness_range[1]:
            raise ValueError("The first element of the brightness range must be less or equal to the second one.")

        if not isinstance(self.rotation_weights, (tuple, list)) or not all(
            isinstance(s, float) or isinstance(s, int) for s in self.rotation_weights
        ):
            raise ValueError("The rotation sampling weights must be provided as tuple or list of float or int values.")
        if len(self.rotation_weights) != 4:
            raise ValueError("The number of rotation sampling weights must be 4.")

        if not all(s >= 0.0 for s in self.rotation_weights):
            raise ValueError("The rotation sampling weights must be non-negative.")

        if all(s == 0.0 for s in self.rotation_weights):
            raise ValueError("At least one of the rotation sampling weights must be strictly greater than zero.")

        if not isinstance(self.sample_size, int):
            raise ValueError("The EOT sample size must be of type int.")
        if self.sample_size <= 0:
            raise ValueError("The EOT sample size must be greater than 0.")
