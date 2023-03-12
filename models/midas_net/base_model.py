#!/user/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from collections import OrderedDict


class BaseModel(torch.nn.Module):
    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        parameters = torch.load(path)

        if "optimizer" in parameters:
            parameters = parameters["net"]

        self.load_state_dict(parameters)

    def get_current_visuals(self):
        visual_ret=OrderedDict()
        for name in self.visual_names:
            if isinstance (name,str):
                visual_ret[name]=getattr(self,name)
        return visual_ret
