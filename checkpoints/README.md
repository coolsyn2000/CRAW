# Released Checkpoints

Git ignores model binaries. The public release provides Ours-S as the default
quick-inference model and Ours-M as an optional higher-rate model. Both are
stripped of optimizer, scheduler, and other training state:

```text
checkpoints/
  ours_s_lambda6e4_176k/inference_fp16.pth
  ours_m_lambda0300_100k/inference_fp16.pth
```

Floating-point tensors are stored on an FP16 value grid and are loaded into the
model's runtime dtype. The two model variants use independent decoder weights;
the `.craw` decoder ID and checkpoint must therefore match. Ours-S remains the
default used by `quick_inference.py`.

Use `tools/export_inference_checkpoint.py` to reproduce this format from a
training checkpoint. Download and verify the released file with:

```bash
bash scripts/download_ours_s_checkpoint.sh
bash scripts/download_ours_m_checkpoint.sh
```

| Model | Google Drive | Size | SHA-256 |
|:---|:---|---:|:---|
| Ours-S (`lambda=6e-4`, step 176k) | [Download](https://drive.google.com/file/d/1_e8Oi28LN9lqUwekWumcBm2Enlu7yIeh/view?usp=sharing) | `44,983,782` bytes | `31d5966026cedf7773bafe62e67d47cdcd00968b805e1840a2040e00c8acb55c` |
| Ours-M (`lambda=3e-4`, step 100k) | [Download](https://drive.google.com/file/d/1opwk4OF1jhN2cFblA17XmkFczdeJLh1O/view?usp=sharing) | `62,754,620` bytes | `4d7c03371dbe19a9cdb74b832373d8f9eabd18c1d98329c788549ad550613a6a` |
