# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from deepspeed.utils import log_dist
from deepspeed.runtime.checkpoint_engine.checkpoint_engine import \
    CheckpointEngine, CheckpointCommitInfo
from deepspeed.runtime.model_checkpointing import CHECKPOINT_SERIALIZATION
import os 
import time

ENGINE_NAME = "TorchCheckpointEngine"


def drop_file_cache(filepath):
    t0 = time.perf_counter()
    fd = os.open(filepath, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        print('drop successfully')
    finally:
        os.close(fd)
    time.sleep(0.1)
    t1 = time.perf_counter() - t0
    print(f'drop using time: {t1}')

class TorchCheckpointEngine(CheckpointEngine):

    def __init__(self, config_params=None):
        super().__init__(config_params)
        self.name = ENGINE_NAME
        if config_params is None:
            self.zipfile_serialization = False
        else:
            self.zipfile_serialization = config_params.checkpoint_config[CHECKPOINT_SERIALIZATION]
        log_dist(f'[{ENGINE_NAME}] Initialized with serialization = {self.zipfile_serialization}', ranks=[0])

    def create(self, info: CheckpointCommitInfo):
        log_dist(f"[Torch] Checkpoint {info.tag} is about to be saved!", ranks=[0])
        pass

    def save(self, state_dict, path: str):
        # log_dist(f"[Torch] Saving [begin] {path}... {self.zipfile_serialization=}", ranks=[0])
        torch.save(state_dict, path, _use_new_zipfile_serialization=self.zipfile_serialization)
        # log_dist(f"[Torch] Saving [end] {path}... {self.zipfile_serialization=}", ranks=[0])

    def load(self, path: str, map_location=None):
        # drop_file_cache(path)
        log_dist(f"[Torch] Begin Load checkpoint from {path}...", ranks=[0])
        # raise ValueError("TRACE++++++++++++++++++++++++++++++")
        partition = torch.load(path, map_location=map_location, weights_only=False)
        print(map_location)
        log_dist(f"[Torch] End Load checkpoint from {path}...", ranks=[0])
        return partition

    def commit(self, info: CheckpointCommitInfo):
        #logger.info(f"[Torch] Checkpoint {tag} is ready now!")
        return True
    def wait(self, persist=True):

        return True