# FNO and LNO training

These four files retain the original training loops, model widths, Fourier/Laplace
modes, interpolation, optimizer and learning-rate schedule. The command line
selects the dataset, downsampling factors, device and a new output directory.
Each selected factor trains its own model. Runs are sequential on the chosen
device; `--batch-size` can reduce memory use.

| Script | Input scaling | Original batch size |
| --- | --- | --- |
| `train_FNO_23456.py` | Raw field values | 350 at 2x/6x, 768 at 3x--5x |
| `train_LNO_23456.py` | Raw field values | 128 at 2x/6x, 256 at 3x--5x |
| `train_FNO_channel_nor_23456.py` | Per-channel training-set mean and standard deviation | 128 |
| `train_LNO_channel_nor_23456.py` | Per-channel training-set mean and standard deviation | 32 |

All four use 10-frame windows on a 64-by-64 grid, 16 modes, an Adam learning
rate of 0.001, weight decay of 0.0001 and 200 epochs. The learning rate halves
after 100 epochs. FNO uses width 64; LNO uses width 8. `--case free-field`
selects pressure and two particle-velocity channels; `--case kol` selects the
two velocity channels. Arrays have shape `[sample,time,x,y,channel]`.

From the repository root:

```bash
python scripts/baselines/train_FNO_channel_nor_23456.py --case free-field \
  --output-dir outputs/fno_free_field --device cuda:0 --batch-size 32
python scripts/baselines/train_LNO_channel_nor_23456.py --case free-field \
  --output-dir outputs/lno_free_field --device cuda:0 --batch-size 32
python scripts/baselines/train_FNO_23456.py --case kol \
  --output-dir outputs/fno_kol --device cuda:0 --batch-size 32
python scripts/baselines/train_LNO_23456.py --case kol \
  --output-dir outputs/lno_kol --device cuda:0 --batch-size 32
```

Use `--scales 6` to train one factor, and `--train-data`/`--test-data` to override
the published paths. The output folder must be new. Released checkpoints stay
under `results_FNO*`/`results_LNO*`; new runs write under `outputs/`.

The free-field comparison uses the channel-normalized models and their saved
normalization parameters. The original Kolmogorov training files use raw
velocity values. The paper evaluation notebook applies the score-model
`scalar.npy` before baseline inference. The released Kolmogorov
weights and the evaluation notebook specify the reported comparison. The training scripts preserve the original preprocessing.
