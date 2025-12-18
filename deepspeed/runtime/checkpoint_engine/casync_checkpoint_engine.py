# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# Apache-2.0 License Copyright (c) UChicago Argonne LLC, operator of Argonne National Laboratory.

# DeepSpeed Team

from deepspeed.runtime.checkpoint_engine.checkpoint_engine import \
    CheckpointEngine, CheckpointCommitInfo
import time
ENGINE_NAME = "CasyncEngine"


class CasyncEngine(CheckpointEngine):

    def __init__(self, deepspeed_config, rank):
        super().__init__(deepspeed_config)
        self.commit_info = None
        self.rank = rank
        self.ckpt_engine = None
        try:
            from datastates import CheckpointEngine as DataStatesEngine
            self.ckpt_engine = DataStatesEngine(deepspeed_config, rank)
        except ImportError:
            raise RuntimeError("Please install DataStates from https://github.com/DataStates/datastates-llm.")
        except Exception as e:
            raise RuntimeError(f"An error occurred while initializing DataStates Checkpoint Engine: {e}")

    def __del__(self):
        self.cleanup()

    def create(self, info: CheckpointCommitInfo):
        self.commit_info = info
        return None

    def save(self, state_dict, path: str):
        # return self.ckpt_engine.coalition_save(state_dict, path)
        print(f"rank [{self.rank}] at {time.time()} save {path}")
        self.ckpt_engine.coalition_save(state_dict, path)


    def load(self, path: str, map_location=None):
        if 'ds_checkpoints/global_step15/layer_01' in path:
            print(path, map_location)
            # exit(0)
        return self.ckpt_engine.split_load_(path, map_location)
    
    def wait(self, persist=True):
        self.ckpt_engine.wait(persist=persist)

    def commit(self, info: CheckpointCommitInfo):
        if info is None:
            return
        # assert info == self.commit_info
        self.ckpt_engine.wait(persist=True)
        self.commit_info = None
        return True

    def cleanup(self):
        self.commit(self.commit_info)
        if self.ckpt_engine:
            self.ckpt_engine.wait(persist=True)
            del self.ckpt_engine

    def is_decoupled(self):
        return True

    def preserves_storage_sharing(self):
        return False
