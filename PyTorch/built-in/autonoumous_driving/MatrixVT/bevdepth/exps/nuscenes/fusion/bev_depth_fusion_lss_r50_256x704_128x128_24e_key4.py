# Copyright (c) Megvii Inc. All rights reserved.
from bevdepth.exps.base_cli import run_cli
from bevdepth.exps.nuscenes.fusion.bev_depth_fusion_lss_r50_256x704_128x128_24e import \
    BEVDepthLightningModel as BaseBEVDepthLightningModel  # noqa
from bevdepth.models.fusion_bev_depth import FusionBEVDepth


class BEVDepthLightningModel(BaseBEVDepthLightningModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sweep_idxes = [4]
        self.head_conf['bev_backbone_conf']['in_channels'] = 80 * (
            len(self.sweep_idxes) + 1)
        self.head_conf['bev_neck_conf']['in_channels'] = [
            80 * (len(self.sweep_idxes) + 1), 160, 320, 640
        ]
        self.head_conf['train_cfg']['code_weight'] = [
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
        ]
        self.model = FusionBEVDepth(self.backbone_conf,
                                    self.head_conf,
                                    is_train_depth=False)


if __name__ == '__main__':
    run_cli(BEVDepthLightningModel,
            'bev_depth_fusion_lss_r50_256x704_128x128_24e_key4')
