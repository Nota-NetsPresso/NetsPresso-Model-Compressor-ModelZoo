#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 1.0
        self.width = 1.0
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        
    def get_model(self):
        from yolox.models.yolo_pafpn import YOLOPAFPN
        from yolox.models.yolo_head_woLoss import YOLOXHead
        from yolox.models.yolox_woLoss import YOLOX
        in_channels = [256, 512, 1024]
        backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act)
        head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act)
        self.model = YOLOX(backbone, head)
        return self.model