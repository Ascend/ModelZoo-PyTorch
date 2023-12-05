# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from cldm.cldm import ControlLDM


class AscendControlNet(ControlLDM):
    def apply_model(self, x_noisy, t, cond, sd_session, control_session):
        assert isinstance(cond, dict)
        cond_txt = torch.cat(cond["c_crossattn"], 1)
        mode = "static"

        control = control_session.infer(
                        [
                            x_noisy.numpy(),
                            torch.cat(cond["c_concat"], 1).numpy(),                            
                            t.numpy(),
                            cond_txt.numpy()
                        ], mode
                    )

        control = [c * scale for c, scale in zip(control, self.control_scales)]

        eps = torch.from_numpy(
                sd_session.infer(
                [
                    x_noisy.numpy(),
                    t.numpy(),
                    cond_txt .numpy()
                ]               
                + control, mode
                )[0]
        )

        return eps
