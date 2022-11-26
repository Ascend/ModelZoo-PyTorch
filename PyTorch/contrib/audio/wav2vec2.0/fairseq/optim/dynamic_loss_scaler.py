# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch


class DynamicLossScaler(object):
    def __init__(
        self,
        init_scale=2.0**15,
        scale_factor=2.0,
        scale_window=2000,
        tolerance=0.0,
        threshold=None,
        min_loss_scale=1e-4,
    ):
        self.loss_scale = init_scale
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self.tolerance = tolerance
        self.threshold = threshold
        self._iter = 0
        self._last_overflow_iter = -1
        self._last_rescale_iter = -1
        self._overflows_since_rescale = 0
        self.min_loss_scale = min_loss_scale
        self.found_inf = torch.npu.FloatTensor([0.0])

    def scale(self, outputs):
        return self.loss_scale * outputs

    def update(self):
        if (self._iter - self._last_overflow_iter) % self.scale_window == 0:
            self.loss_scale *= self.scale_factor
            self._last_rescale_iter = self._iter
        self._iter += 1

    def _decrease_loss_scale(self):
        self.loss_scale /= self.scale_factor
        if self.threshold is not None:
            self.loss_scale = max(self.loss_scale, self.threshold)

    def get_npu_overflow_flag(self):
        float_status = torch.zeros(8).npu()
        result = torch.npu_get_float_status(float_status)
        if float_status.cpu()[0] != 0:
            return True
        else:
            return False

    def check_overflow(self, grad_norm):
        # detect inf and nan
        self.found_inf.fill_(0.0)
        has_overflow = self.get_npu_overflow_flag()
        if has_overflow:
            self.found_inf.fill_(1)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(self.found_inf,
                                        op=torch.distributed.ReduceOp.MAX)
        found_inf_flag = (self.found_inf.item()) > 0
        if found_inf_flag:
            # overflow has occured
            prev_scale = self.loss_scale
            iter_since_rescale = self._iter - self._last_rescale_iter

            self._last_overflow_iter = self._iter
            self._overflows_since_rescale += 1
            pct_overflow = self._overflows_since_rescale / float(iter_since_rescale)
            if pct_overflow >= self.tolerance:
                self._decrease_loss_scale()
                self._last_rescale_iter = self._iter
                self._overflows_since_rescale = 0

            if self.loss_scale <= self.min_loss_scale:
                # Use FloatingPointError as an uncommon error that parent
                # functions can safely catch to stop training.
                self.loss_scale = prev_scale
                raise FloatingPointError(
                    (
                        "Minimum loss scale reached ({}). Your loss is probably exploding. "
                        "Try lowering the learning rate, using gradient clipping or "
                        "increasing the batch size."
                    ).format(self.min_loss_scale)
                )

            self._iter += 1
            raise OverflowError("setting loss scale to: " + str(self.loss_scale))