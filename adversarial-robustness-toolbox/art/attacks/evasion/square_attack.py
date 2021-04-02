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
This module implements the `SquareAttack` attack.

| Paper link: https://arxiv.org/abs/1912.00049
"""
import bisect
import logging
import math
import random
from typing import Optional, Union, TYPE_CHECKING

import numpy as np
from tqdm.auto import trange

from art.config import ART_NUMPY_DTYPE
from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator, NeuralNetworkMixin
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import check_and_transform_label_format, get_labels_np_array

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE

logger = logging.getLogger(__name__)


class SquareAttack(EvasionAttack):
    attack_params = EvasionAttack.attack_params + [
        "norm",
        "max_iter",
        "eps",
        "p_init",
        "nb_restarts",
        "batch_size",
        "verbose",
    ]

    _estimator_requirements = (BaseEstimator, ClassifierMixin, NeuralNetworkMixin)

    def __init__(
            self,
            estimator: "CLASSIFIER_TYPE",
            norm: Union[int, float, str] = np.inf,
            max_iter: int = 100,
            eps: float = 0.3,
            p_init: float = 0.8,
            nb_restarts: int = 1,
            batch_size: int = 128,
            verbose: bool = True,
            max_queries: int = 10000,
    ):
        """
        Create a :class:`.SquareAttack` instance.

        :param estimator: An trained estimator.
        :param norm: The norm of the adversarial perturbation. Possible values: "inf", np.inf, 1 or 2.
        :param max_iter: Maximum number of iterations.
        :param eps: Maximum perturbation that the attacker can introduce.
        :param p_init: Initial fraction of elements.
        :param nb_restarts: Number of restarts.
        :param batch_size: Batch size for estimator evaluations.
        :param verbose: Show progress bars.
        """
        super().__init__(estimator=estimator)

        self.norm = norm
        self.max_iter = max_iter
        self.eps = eps
        self.p_init = p_init
        self.nb_restarts = nb_restarts
        self.batch_size = batch_size
        self.verbose = verbose
        self._check_params()
        self.max_queries = max_queries
        self.num_queries = 0
        self.old_y_pred = []
        self.new_y_pred = []

    def _get_norm(self, image, norm):
        if norm == 2:
            return np.linalg.norm(np.reshape(image, -1), norm)
        if norm == np.inf:
            return np.max(np.abs(image))

    def _get_logits_diff(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.new_y_pred = self.estimator.predict(x, batch_size=self.batch_size)
        self.num_queries += 1

        logit_correct = np.take_along_axis(self.new_y_pred, np.expand_dims(np.argmax(y, axis=1), axis=1), axis=1)

        correct_index = np.argmax(y, axis=1)[0]
        copy_y_pred = self.new_y_pred.copy()
        copy_y_pred[0][correct_index] = 0
        logit_highest_incorrect = np.take_along_axis(copy_y_pred,
                                                     np.expand_dims(np.argsort(copy_y_pred, axis=1)[:, -1], axis=1),
                                                     axis=1)

        return (logit_correct - logit_highest_incorrect)[:, 0]

    def _get_percentage_of_elements(self, i_iter: int) -> float:
        i_p = i_iter / self.max_iter
        intervals = [0.001, 0.005, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]
        p_ratio = [1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64, 1 / 128, 1 / 256, 1 / 512]
        i_ratio = bisect.bisect_left(intervals, i_p)

        return self.p_init * p_ratio[i_ratio]

    def generate(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate adversarial samples and return them in an array.

        :param x: An array with the original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :return: An array holding the adversarial examples.
        """
        normalization_term = self._get_norm(x, norm=self.norm)
        self.eps *= normalization_term
        self.num_queries = 0

        if x.ndim != 4:
            raise ValueError("Unrecognized input dimension. Attack can only be applied to image data.")

        x_adv = x.astype(ART_NUMPY_DTYPE)

        y = check_and_transform_label_format(y, self.estimator.nb_classes)

        if y is None:
            # Use model predictions as true labels
            logger.info("Using model predictions as true labels.")
            y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))
            self.num_queries += 1

        if self.estimator.channels_first:
            channels = x.shape[1]
            height = x.shape[2]
            width = x.shape[3]
        else:
            height = x.shape[1]
            width = x.shape[2]
            channels = x.shape[3]

        for _ in trange(self.nb_restarts, desc="SquareAttack - restarts", disable=not self.verbose):

            # Determine correctly predicted samples
            y_pred = self.estimator.predict(x_adv, batch_size=self.batch_size)
            self.num_queries += 1
            sample_is_robust = np.argmax(y_pred, axis=1) == np.argmax(y, axis=1)

            if np.sum(sample_is_robust) == 0:
                break

            x_robust = x[sample_is_robust]
            y_robust = y[sample_is_robust]
            y_pred = self.estimator.predict(x_robust, batch_size=self.batch_size)
            self.num_queries += 1
            logit_correct = np.take_along_axis(y_pred, np.expand_dims(np.argmax(y_robust, axis=1), axis=1), axis=1)

            correct_index = np.argmax(y_robust, axis=1)[0]
            copy_y_pred = y_pred.copy()
            copy_y_pred[0][correct_index] = 0
            logit_highest_incorrect = np.take_along_axis(copy_y_pred,
                                                         np.expand_dims(
                                                             np.argsort(copy_y_pred, axis=1)[:, -1],
                                                             axis=1),
                                                         axis=1)

            sample_logits_diff_init = (logit_correct - logit_highest_incorrect)[:, 0]

            if self.norm in [np.inf, "inf"]:

                if self.estimator.channels_first:
                    size = (x_robust.shape[0], channels, 1, width)
                else:
                    size = (x_robust.shape[0], 1, width, channels)

                # Add vertical stripe perturbations
                x_robust_new = np.clip(
                    x_robust + self.eps * np.random.choice([-1, 1], size=size),
                    a_min=self.estimator.clip_values[0],
                    a_max=self.estimator.clip_values[1],
                ).astype(ART_NUMPY_DTYPE)

                sample_logits_diff_new = self._get_logits_diff(x_robust_new, y_robust)
                logits_diff_improved = (sample_logits_diff_new - sample_logits_diff_init) < 0.0

                x_robust[logits_diff_improved] = x_robust_new[logits_diff_improved]

                x_adv[sample_is_robust] = x_robust

                for i_iter in trange(
                        self.max_iter, desc="SquareAttack - iterations", leave=False, disable=not self.verbose
                ):

                    percentage_of_elements = self._get_percentage_of_elements(i_iter)

                    # Determine correctly predicted samples
                    y_pred = self.estimator.predict(x_adv, batch_size=self.batch_size)
                    self.num_queries += 1
                    sample_is_robust = np.argmax(y_pred, axis=1) == np.argmax(y, axis=1)

                    if np.sum(sample_is_robust) == 0:
                        break

                    x_robust = x_adv[sample_is_robust]
                    x_init = x[sample_is_robust]
                    y_robust = y[sample_is_robust]

                    sample_logits_diff_init = self._get_logits_diff(x_robust, y_robust)

                    height_tile = max(int(round(math.sqrt(percentage_of_elements * height * width))), 1)

                    height_mid = np.random.randint(0, height - height_tile)
                    width_start = np.random.randint(0, width - height_tile)

                    delta_new = np.zeros(self.estimator.input_shape)

                    if self.estimator.channels_first:
                        delta_new[
                        :, height_mid: height_mid + height_tile, width_start: width_start + height_tile
                        ] = np.random.choice([-2 * self.eps, 2 * self.eps], size=[channels, 1, 1])
                    else:
                        delta_new[
                        height_mid: height_mid + height_tile, width_start: width_start + height_tile, :
                        ] = np.random.choice([-2 * self.eps, 2 * self.eps], size=[1, 1, channels])

                    x_robust_new = x_robust + delta_new

                    x_robust_new = np.minimum(np.maximum(x_robust_new, x_init - self.eps), x_init + self.eps)

                    x_robust_new = np.clip(
                        x_robust_new, a_min=self.estimator.clip_values[0], a_max=self.estimator.clip_values[1]
                    ).astype(ART_NUMPY_DTYPE)

                    sample_logits_diff_new = self._get_logits_diff(x_robust_new, y_robust)
                    logits_diff_improved = (sample_logits_diff_new - sample_logits_diff_init) < 0.0

                    x_robust[logits_diff_improved] = x_robust_new[logits_diff_improved]

                    x_adv[sample_is_robust] = x_robust

                    if self.num_queries >= self.max_queries:
                        print("\nReached max number of queries\n")
                        break

            elif self.norm == 2:

                n_tiles = 5

                height_tile = height // n_tiles

                def _get_perturbation(h):
                    delta = np.zeros([h, h])
                    gaussian_perturbation = np.zeros([h // 2, h])

                    x_c = h // 4
                    y_c = h // 2

                    for i_y in range(y_c):
                        gaussian_perturbation[
                        max(x_c, 0): min(x_c + (2 * i_y + 1), h // 2), max(0, y_c): min(y_c + (2 * i_y + 1), h)
                        ] += 1.0 / ((i_y + 1) ** 2)
                        x_c -= 1
                        y_c -= 1

                    gaussian_perturbation /= np.sqrt(np.sum(gaussian_perturbation ** 2))

                    delta[: h // 2] = gaussian_perturbation
                    delta[h // 2: h // 2 + gaussian_perturbation.shape[0]] = -gaussian_perturbation

                    delta /= np.sqrt(np.sum(delta ** 2))

                    if random.random() > 0.5:
                        delta = np.transpose(delta)

                    if random.random() > 0.5:
                        delta = -delta

                    return delta

                delta_init = np.zeros(x_robust.shape, dtype=ART_NUMPY_DTYPE)

                height_start = 0
                for _ in range(n_tiles):
                    width_start = 0
                    for _ in range(n_tiles):
                        if self.estimator.channels_first:
                            perturbation_size = (1, 1, height_tile, height_tile)
                            random_size = (x_robust.shape[0], channels, 1, 1)
                        else:
                            perturbation_size = (1, height_tile, height_tile, 1)
                            random_size = (x_robust.shape[0], 1, 1, channels)

                        perturbation = _get_perturbation(height_tile).reshape(perturbation_size) * np.random.choice(
                            [-1, 1], size=random_size
                        )

                        if self.estimator.channels_first:
                            delta_init[
                            :, :, height_start: height_start + height_tile, width_start: width_start + height_tile
                            ] += perturbation
                        else:
                            delta_init[
                            :, height_start: height_start + height_tile, width_start: width_start + height_tile, :
                            ] += perturbation
                        width_start += height_tile
                    height_start += height_tile

                x_robust_new = np.clip(
                    x_robust + delta_init / np.sqrt(np.sum(delta_init ** 2, axis=(1, 2, 3), keepdims=True)) * self.eps,
                    self.estimator.clip_values[0],
                    self.estimator.clip_values[1],
                )

                sample_logits_diff_new = self._get_logits_diff(x_robust_new, y_robust)
                logits_diff_improved = (sample_logits_diff_new - sample_logits_diff_init) < 0.0

                x_robust[logits_diff_improved] = x_robust_new[logits_diff_improved]

                x_adv[sample_is_robust] = x_robust

                self.old_y_pred = self.estimator.predict(x_adv, batch_size=self.batch_size)
                self.num_queries += 1

                for i_iter in trange(
                        self.max_iter, desc="SquareAttack - iterations", leave=False, disable=not self.verbose
                ):

                    percentage_of_elements = self._get_percentage_of_elements(i_iter)

                    # Determine correctly predicted samples
                    sample_is_robust = np.argmax(self.old_y_pred, axis=1) == np.argmax(y, axis=1)

                    if np.sum(sample_is_robust) == 0:
                        for i in range(5):
                            self.old_y_pred = self.estimator.predict(x_adv, batch_size=self.batch_size)
                            self.num_queries += 1
                            sample_is_robust = np.argmax(self.old_y_pred, axis=1) == np.argmax(y, axis=1)
                            if np.sum(sample_is_robust) != 0:
                                break
                        if np.sum(sample_is_robust) == 0:
                            break

                    x_robust = x_adv[sample_is_robust]
                    x_init = x[sample_is_robust]
                    y_robust = y[sample_is_robust]

                    logit_correct = np.take_along_axis(self.old_y_pred,
                                                       np.expand_dims(np.argmax(y_robust, axis=1), axis=1), axis=1)
                    correct_index = np.argmax(y_robust, axis=1)[0]
                    copy_y_pred = self.old_y_pred.copy()
                    copy_y_pred[0][correct_index] = 0
                    logit_highest_incorrect = np.take_along_axis(copy_y_pred,
                                                                 np.expand_dims(
                                                                     np.argsort(copy_y_pred, axis=1)[:, -1],
                                                                     axis=1),
                                                                 axis=1)
                    sample_logits_diff_init = (logit_correct - logit_highest_incorrect)[:, 0]

                    delta_x_robust_init = x_robust - x_init

                    height_tile = max(int(round(math.sqrt(percentage_of_elements * height * width))), 3)

                    if height_tile % 2 == 0:
                        height_tile += 1
                    height_tile_2 = height_tile

                    height_start = np.random.randint(0, height - height_tile)
                    width_start = np.random.randint(0, width - height_tile)

                    new_deltas_mask = np.zeros(x_init.shape)
                    if self.estimator.channels_first:
                        new_deltas_mask[
                        :, :, height_start: height_start + height_tile, width_start: width_start + height_tile
                        ] = 1.0
                        w_1_norm = np.sqrt(
                            np.sum(
                                delta_x_robust_init[
                                :,
                                :,
                                height_start: height_start + height_tile,
                                width_start: width_start + height_tile,
                                ]
                                ** 2,
                                axis=(2, 3),
                                keepdims=True,
                            )
                        )
                    else:
                        new_deltas_mask[
                        :, height_start: height_start + height_tile, width_start: width_start + height_tile, :
                        ] = 1.0
                        w_1_norm = np.sqrt(
                            np.sum(
                                delta_x_robust_init[
                                :,
                                height_start: height_start + height_tile,
                                width_start: width_start + height_tile,
                                :,
                                ]
                                ** 2,
                                axis=(1, 2),
                                keepdims=True,
                            )
                        )

                    height_2_start = np.random.randint(0, height - height_tile_2)
                    width_2_start = np.random.randint(0, width - height_tile_2)

                    new_deltas_mask_2 = np.zeros(x_init.shape)
                    if self.estimator.channels_first:
                        new_deltas_mask_2[
                        :,
                        :,
                        height_2_start: height_2_start + height_tile_2,
                        width_2_start: width_2_start + height_tile_2,
                        ] = 1.0
                    else:
                        new_deltas_mask_2[
                        :,
                        height_2_start: height_2_start + height_tile_2,
                        width_2_start: width_2_start + height_tile_2,
                        :,
                        ] = 1.0

                    norms_x_robust = np.sqrt(np.sum((x_robust - x_init) ** 2, axis=(1, 2, 3), keepdims=True))
                    w_norm = np.sqrt(
                        np.sum(
                            (delta_x_robust_init * np.maximum(new_deltas_mask, new_deltas_mask_2)) ** 2,
                            axis=(1, 2, 3),
                            keepdims=True,
                        )
                    )

                    if self.estimator.channels_first:
                        new_deltas_size = [x_init.shape[0], channels, height_tile, height_tile]
                        random_choice_size = [x_init.shape[0], channels, 1, 1]
                        perturbation_size = [1, 1, height_tile, height_tile]
                    else:
                        new_deltas_size = [x_init.shape[0], height_tile, height_tile, channels]
                        random_choice_size = [x_init.shape[0], 1, 1, channels]
                        perturbation_size = [1, height_tile, height_tile, 1]

                    delta_new = (
                            np.ones(new_deltas_size)
                            * _get_perturbation(height_tile).reshape(perturbation_size)
                            * np.random.choice([-1, 1], size=random_choice_size)
                    )

                    if self.estimator.channels_first:
                        delta_new += delta_x_robust_init[
                                     :, :, height_start: height_start + height_tile,
                                     width_start: width_start + height_tile
                                     ] / (np.maximum(1e-9, w_1_norm))
                    else:
                        delta_new += delta_x_robust_init[
                                     :, height_start: height_start + height_tile,
                                     width_start: width_start + height_tile, :
                                     ] / (np.maximum(1e-9, w_1_norm))

                    diff_norm = (self.eps * np.ones(delta_new.shape)) ** 2 - norms_x_robust ** 2
                    diff_norm[diff_norm < 0.0] = 0.0

                    if self.estimator.channels_first:
                        np.seterr(divide='ignore', invalid='ignore')
                        delta_new /= np.sqrt(np.sum(delta_new ** 2, axis=(2, 3), keepdims=True)) * np.sqrt(
                            diff_norm / channels + w_norm ** 2
                        )
                        delta_x_robust_init[
                        :,
                        :,
                        height_2_start: height_2_start + height_tile_2,
                        width_2_start: width_2_start + height_tile_2,
                        ] = 0.0
                        delta_x_robust_init[
                        :, :, height_start: height_start + height_tile, width_start: width_start + height_tile
                        ] = delta_new
                    else:
                        delta_new /= np.sqrt(np.sum(delta_new ** 2, axis=(1, 2), keepdims=True)) * np.sqrt(
                            diff_norm / channels + w_norm ** 2
                        )
                        delta_x_robust_init[
                        :,
                        height_2_start: height_2_start + height_tile_2,
                        width_2_start: width_2_start + height_tile_2,
                        :,
                        ] = 0.0
                        delta_x_robust_init[
                        :, height_start: height_start + height_tile, width_start: width_start + height_tile, :
                        ] = delta_new

                    x_robust_new = np.clip(
                        x_init
                        + self.eps
                        * delta_x_robust_init
                        / np.sqrt(np.sum(delta_x_robust_init ** 2, axis=(1, 2, 3), keepdims=True)),
                        self.estimator.clip_values[0],
                        self.estimator.clip_values[1],
                    )

                    sample_logits_diff_new = self._get_logits_diff(x_robust_new, y_robust)
                    logits_diff_improved = (sample_logits_diff_new - sample_logits_diff_init) < 0.0

                    if logits_diff_improved:
                        self.old_y_pred = self.new_y_pred

                    x_robust[logits_diff_improved] = x_robust_new[logits_diff_improved]

                    x_adv[sample_is_robust] = x_robust

                    if self.num_queries >= self.max_queries:
                        print("\nReached max number of queries\n")
                        self.old_y_pred = self.estimator.predict(x, batch_size=self.batch_size)
                        break

        self.eps /= normalization_term
        return x_adv

    def _check_params(self) -> None:
        if self.norm not in [1, 2, np.inf, "inf"]:
            raise ValueError('The argument NORM has to be either 1, 2, np.inf, or "inf".')

        if not isinstance(self.max_iter, int) or self.max_iter <= 0:
            raise ValueError("The argument max_iter has to be of type int and larger than zero.")

        if not isinstance(self.eps, (int, float)) or self.eps <= 0.0:
            raise ValueError("The argument eps has to be either of type int or float and larger than zero.")

        if not isinstance(self.p_init, (int, float)) or self.p_init <= 0.0 or self.p_init >= 1.0:
            raise ValueError("The argument p_init has to be either of type int or float and in range [0, 1].")

        if not isinstance(self.nb_restarts, int) or self.nb_restarts <= 0:
            raise ValueError("The argument nb_restarts has to be of type int and larger than zero.")

        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("The argument batch_size has to be of type int and larger than zero.")

        if not isinstance(self.verbose, bool):
            raise ValueError("The argument `verbose` has to be of type bool.")