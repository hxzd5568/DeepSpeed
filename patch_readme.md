# DeepSeek Casync Checkpoint Patch

This directory contains patches for integrating async checkpoint engine (casync) support into DeepSpeed, adapted from @hxzd5568's work on the `develop` branch.

## Changes Included (6 commits)

| Commit | Description |
|--------|-------------|
| `cde7b599` | [Apply] casync engine |
| `d1545006` | minor fix |
| `fcf0c560` | minor fix |
| `a99fc6ff` | [Modify] cnn evaluation |
| `1c84cdc0` | [Integrate] long-short term |
| `632f16c6` | [fit into zero3] |

## Files Modified

- `deepspeed/datastates/config.py` — add `casync_ckpt` config key
- `deepspeed/runtime/checkpoint_engine/casync_checkpoint_engine.py` — **new** CasyncEngine
- `deepspeed/runtime/checkpoint_engine/datastates_checkpoint_engine.py` — add `wait()`, relax assertion
- `deepspeed/runtime/checkpoint_engine/torch_checkpoint_engine.py` — add `wait()` stub
- `deepspeed/runtime/checkpoint_engine/utils.py` — wire casync into engine selection

## Apply to Installed DeepSpeed

```bash
bash scripts/apply_hxzd5568_patches.sh
```

To specify a custom site-packages path:

```bash
bash scripts/apply_hxzd5568_patches.sh /path/to/site-packages
```

## Dependencies

The casync engine requires [datastates-llm](https://github.com/DataStates/datastates-llm). If not installed, it falls back to `torch.save`.
