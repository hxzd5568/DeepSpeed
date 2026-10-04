# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# Apache-2.0 License Copyright (c) UChicago Argonne LLC, operator of Argonne National Laboratory.

# DeepSpeed Team

from deepspeed.runtime.config_utils import DeepSpeedConfigObject
import copy

DATASTATES_CHECKPOINTING = "datastates_ckpt"
CASYNC_CHECKPOINTING = "casync_ckpt"
DATASTATES_CHECKPOINTING_ENABLED = False
ASYNC_LOAD = "async_load"
ASYNC_LOAD_DEFAULT = False


class DeepSpeedDataStatesConfig(DeepSpeedConfigObject):

    def __init__(self, param_dict):
        super(DeepSpeedDataStatesConfig, self).__init__()

        self.enabled = param_dict.get(DATASTATES_CHECKPOINTING, DATASTATES_CHECKPOINTING_ENABLED) is not False
        self.enabled_casync = param_dict.get(CASYNC_CHECKPOINTING, DATASTATES_CHECKPOINTING_ENABLED) is not False
        if self.enabled:
            self.config = copy.deepcopy(param_dict.get(DATASTATES_CHECKPOINTING, None))
        else:
            self.config = copy.deepcopy(param_dict.get(CASYNC_CHECKPOINTING, None))

        # Controls whether optimizer states may be loaded asynchronously in the
        # background while the model weights are already usable for forward/backward.
        # Can be given as a top-level "async_load" key or inside the
        # "datastates_ckpt"/"casync_ckpt" engine config.
        self.async_load = param_dict.get(ASYNC_LOAD, ASYNC_LOAD_DEFAULT)
        if isinstance(self.config, dict) and ASYNC_LOAD in self.config:
            self.async_load = self.config.get(ASYNC_LOAD, self.async_load)
