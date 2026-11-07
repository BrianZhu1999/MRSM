"""Command-line paths and case settings for the baseline trainers."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
CASE_FILES = {
    "free-field": ("data/anechoic_high_prec_dataset_last700_2.npy",
                   "data/anechoic_high_prec_dataset_last700_test_2.npy", 3),
    "kol": ("data/kolmogorov_flow_train.npy", "data/kolmogorov/kolmogorov_flow_test.npy", 2),
}


def path_from_root(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def run(config_type, load_data, train, model, normalizer, batches):
    parser = argparse.ArgumentParser(description="Train independent 2x--6x baseline models on 10-frame windows.")
    parser.add_argument("--case", choices=CASE_FILES, default="free-field")
    parser.add_argument("--train-data", type=path_from_root)
    parser.add_argument("--test-data", type=path_from_root)
    parser.add_argument("--output-dir", type=path_from_root, required=True)
    parser.add_argument("--scales", type=int, nargs="+", choices=range(2,7), default=[2,3,4,5,6])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, help="override the per-scale batch sizes")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or (args.batch_size is not None and args.batch_size < 1) or args.workers < 0:
        parser.error("epochs and batch-size must be positive; workers must be nonnegative")
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        parser.error("device must be cpu or cuda:N")
    if device.type == "cuda" and (device.index is None or not torch.cuda.is_available()):
        parser.error("select an available explicit CUDA device, such as cuda:0")
    config = config_type()
    train_file, test_file, channels = CASE_FILES[args.case]
    config.TRAIN_PATH = str(args.train_data or ROOT / train_file)
    config.TEST_PATH = str(args.test_data or ROOT / test_file)
    config.NUM_CHANNELS = channels
    config.SAVE_DIR = str(args.output_dir)
    config.EPOCHS = args.epochs
    config.NUM_WORKERS = args.workers
    config.FILE_PREFIX = model + ("_2dSound" if model == "fno" else "_2dsound") if args.case == "free-field" else model
    for filename in (config.TRAIN_PATH, config.TEST_PATH):
        if not Path(filename).is_file():
            raise FileNotFoundError(filename)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    x_train = load_data(config.TRAIN_PATH, config, "Training data")
    x_test = load_data(config.TEST_PATH, config, "Test data")
    if normalizer is not None:
        scale = normalizer(x_train, num_channels=channels)
        x_train, x_test = scale.encode(x_train), scale.encode(x_test)
        name = "normalizer_params.pth" if model == "fno" else "lno_normalizer_params.pth"
        torch.save(scale.state_dict(), args.output_dir / name)
    settings = {key:getattr(config,key) for key in dir(config) if key.isupper()}
    settings.update(case=args.case, scales=args.scales, seed=args.seed, device=args.device,
                    normalization="per-channel" if normalizer else "raw",
                    batch_sizes={s:args.batch_size or batches[s] for s in args.scales})
    (args.output_dir / "training_settings.json").write_text(json.dumps(settings, indent=2))
    for scale in dict.fromkeys(args.scales):
        train(scale, args.batch_size or batches[scale], config, x_train, x_test,
              device.index if device.type == "cuda" else -1)
