# Model checkpoints

- `TrackNet_official_best.pt` is the official TrackNetV3 TrackNet checkpoint
  (`seq_len=8`, 30 training epochs, `bg_mode=concat`). It is the default used
  by `run_all_mac.sh` when present.
- `TrackNet_best.pt` is the legacy repository checkpoint (`seq_len=4`, 3
  epochs) and remains available as a fallback; it is not overwritten.
- `yolov8s-pose.pt` is used for player pose/body validation.

Official TrackNetV3 checkpoint bundle: Google Drive file
`1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA`.

SHA-256 for `TrackNet_official_best.pt`:

```text
df867641a02712b021f04548ff4b1208ddfdb47f629ab2094ceb978667e83b1a
```

The `*.pt` files are Git LFS objects. Model licensing follows the respective
upstream project and is intended for research/educational use.
