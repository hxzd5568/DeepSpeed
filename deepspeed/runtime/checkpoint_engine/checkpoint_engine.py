# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os

import abc
from abc import ABC

from dataclasses import dataclass


@dataclass
class CheckpointCommitInfo(object):
    tag: str
    save_dir: str
    save_latest: bool


class CheckpointEngine(ABC):
    # init checkpoint engine for save/load
    def __init__(self, config_params=None):
        self.name = None

    @abc.abstractmethod
    def create(self, info: CheckpointCommitInfo):
        # create checkpoint on give tag for save/load.
        ...

    @abc.abstractmethod
    def save(self, state_dict, path: str):
        ...

    def makedirs(self, path, exist_ok=False):
        os.makedirs(path, exist_ok=exist_ok)

    @abc.abstractmethod
    def load(self, path: str, map_location=None):
        ...

    @abc.abstractmethod
    def commit(self, info: CheckpointCommitInfo):
        # to tell checkpoint services if all files are ready.
        ...

    def is_data_parallel_writer(self, dp_rank):
        return dp_rank == 0

    def is_decoupled(self):
        return False

    def supports_async_load(self):
        """Whether this engine supports asynchronous optimizer-state loading.

        When True and the ``async_load`` config option is enabled, the staged
        load API (``engine.load_checkpoint_stage(..., stage=0)``) may start
        loading the optimizer states in a background thread right after the
        weights are loaded; ``engine.wait_for_optimizer_states()`` blocks until
        they are ready. Engines returning False fall back to the synchronous
        two-stage load.
        """
        return False

    def supports_hot_backup(self):
        """Whether this engine keeps an in-memory hot backup of the model
        weights after save (created on the engine side, see ``hot_weights``).
        """
        return False

    def set_commit_info(self, info: CheckpointCommitInfo):
        pass

    def get_commit_info(self):
        return None

    def cleanup(self):
        pass

    def preserves_storage_sharing(self):
        return True
