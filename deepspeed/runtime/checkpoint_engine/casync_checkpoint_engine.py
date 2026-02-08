# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# Apache-2.0 License Copyright (c) UChicago Argonne LLC, operator of Argonne National Laboratory.

# DeepSpeed Team

from deepspeed.runtime.checkpoint_engine.checkpoint_engine import \
    CheckpointEngine, CheckpointCommitInfo
import time
# import torch

ENGINE_NAME = "CasyncEngine"


# def profile_tensor(state_dict):
#     all_max, all_p_min, all_min, all_abs_mean = [], [], [], []
    
#     def collect_stats(obj):
#         if isinstance(obj, dict):
#             for v in obj.values():
#                 collect_stats(v)
#         elif isinstance(obj, list):
#             for item in obj:
#                 collect_stats(item)
#         elif isinstance(obj, torch.Tensor) and obj.numel() > 1200:
#             t = obj.detach().float()
#             all_max.append(t.max().item())
#             all_min.append(t.min().item())
#             all_p_min.append(t.abs().min().item())
#             all_abs_mean.append(t.abs().mean().item())
#             if t.abs().mean().item() > 0.01:
#                 print(obj)
#         else:
#             pass
    
#     collect_stats(state_dict)
#     if all_max:
#         print(f"[PROFILE] max: {sum(all_max)/len(all_max):.6f}, "
#               f"[PROFILE] positive min: {sum(all_p_min)/len(all_min)}, "
#               f"[PROFILE] min: {sum(all_min)/len(all_min):.6f}, "
#               f"[PROFILE] abs_mean: {sum(all_abs_mean)/len(all_abs_mean):.6f}")
#         look_up_sign = sum(all_abs_mean)/len(all_abs_mean)
#     else:
#         print("[PROFILE] No tensors found in optimizer state.")

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
        self.ckpt_engine.coalition_save(state_dict, path)

    def load(self, path: str, map_location=None):
        return self.ckpt_engine.split_load(path, map_location)
    
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
