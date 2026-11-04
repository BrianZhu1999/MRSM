"""Check downloaded paper arrays and strictly load every released score model on CPU.

Run after extracting the data and weight archives into the repository root:
    python tests/check_release_assets.py --output verification/release_assets.json
"""
import argparse
import ast
import gc
import importlib.metadata
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from models import get_model, get_ema
from trainer.utils import restore_checkpoint

CHECKPOINTS = {
    "2DSound_v22": "checkpoint_200.pth",
    "kol_v1": "checkpoint_517.pth",
    "kol_v1_finetune_FPE": "checkpoint_517.pth",
    "era5_v0": "checkpoint.pth",
    "1dtube_v31": "checkpoint.pth",
    "1dtube_v41": "checkpoint.pth",
    "exp_p_v21": "checkpoint.pth",
    "line_array_v33": "checkpoint_600.pth",
    "line_array_v3": "checkpoint.pth",
}
ARRAYS = {
    "data/anechoic_high_prec_dataset_last700_2.npy": (130, 700, 64, 64, 3),
    "results/2DSound_v22/scalar.npy": (2, 1, 1, 1, 1, 3),
    "results/kol_v1/scalar.npy": (2,),
    "results/kol_v1_finetune_FPE/scalar.npy": (2,),
    "results/era5_v0/scalars_era5.npy": (4, 2),
    "results/1dtube_v31/scalar.npy": (2, 1, 1, 1, 2),
    "results/1dtube_v41/scalar.npy": (2, 1, 1, 1, 2),
    "results/exp_p_v21/scalar.npy": (2,),
    "results/line_array_v33/scalar.npy": (2,),
    "results/line_array_v3/scalar.npy": (2,),
    "data/scalar.npy": (),
    "data/anechoic_high_prec_dataset_last700_test_2.npy": (9, 700, 64, 64, 3),
    "data/kolmogorov_flow_train.npy": (700, 400, 64, 64, 2),
    "data/kolmogorov/kolmogorov_flow_test.npy": (20, 400, 64, 64, 2),
    "data/era5/era5_train.npy": (1, 16071, 64, 64, 4),
    "data/era5/era5_test.npy": (1, 273, 64, 64, 4),
    "data/era5/scalars_era5.npy": (4, 2),
    "data/tube_pressure_velocity_train.npy": (78, 2001, 64, 2),
    "data/tube_pressure_train.npy": (78, 2001, 64, 1),
    "data/tube_highfre.npy": (205, 2001, 64, 2),
    "data/tube_highfre_test.npy": (20, 2001, 64, 2),
    "data/pu385.npy": (1, 50000, 2, 15),
    "data/pu1139.npy": (1, 50000, 2, 16),
    "data/tube175_resampled_calibrated.npy": (1, 100000, 1, 16),
    "data/tube345_resampled_calibrated.npy": (1, 100000, 1, 16),
    "data/tube395_resampled_calibrated.npy": (1, 100000, 1, 16),
    "data/fdtd_targeted_400_0.032.npy": (400, 1000, 32, 4),
    "data/Aligned_2900_3000_pos5.npy": (7, 4, 5000),
    "data/fdtd_3d_data_an_100_test.npy": (100, 2000, 32, 4),
    "data/fdtd_3d_data_norm.npy": (1500, 2000, 32, 4),
    "data/fdtd_3d_data_an_norm.npy": (500, 2000, 32, 4),
}


def npy_header(path):
    with path.open("rb") as stream:
        if stream.read(6) != b"\x93NUMPY":
            raise ValueError(f"Not a NumPy array: {path}")
        major, minor = stream.read(2)
        if major not in (1, 2, 3):
            raise ValueError(f"Unsupported NumPy header version: {(major, minor)}")
        count = 2 if major == 1 else 4
        length = struct.unpack("<H" if count == 2 else "<I", stream.read(count))[0]
        if length > 65536:
            raise ValueError(f"Unexpectedly large NumPy header: {path}")
        return ast.literal_eval(stream.read(length).decode("utf8" if major == 3 else "latin1").strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    results = {"python": sys.version, "packages": {}, "arrays": [], "models": []}
    for package in ("torch", "numpy", "scipy", "matplotlib", "einops", "tqdm", "pandas"):
        results["packages"][package] = importlib.metadata.version(package)
    for relative, shape in ARRAYS.items():
        header = npy_header(ROOT / relative)
        if tuple(header["shape"]) != shape:
            raise ValueError(f"Unexpected array shape for {relative}: {header['shape']}, expected {shape}")
        results["arrays"].append({"path": relative, **header})
    for name, filename in CHECKPOINTS.items():
        directory = ROOT / "results" / name
        config = json.loads((directory / "config.json").read_text())
        if not (ROOT / config["data_location"]).is_file():
            raise FileNotFoundError(f"Training data for {name}: {config['data_location']}")
        config["num_channels"] = config["num_components"] + config["num_conditions"]
        model = get_model(SimpleNamespace(**config)).cpu()
        ema = get_ema(model.parameters(), decay=config["ema_rate"])
        state = restore_checkpoint(directory / filename, {"model": model, "ema": ema}, "cpu")
        for key, parameter in model.named_parameters():
            if not torch.isfinite(parameter).all().item():
                raise ValueError(f"Nonfinite model parameter in {name}: {key}")
        for index, shadow in enumerate(ema.shadow_params):
            if not torch.isfinite(shadow).all().item():
                raise ValueError(f"Nonfinite EMA parameter in {name}: {index}")
        ema.copy_to(model.parameters())
        row = {"case": name, "checkpoint": filename, "epoch": state.get("epoch"),
               "iteration": state.get("iteration"), "parameters": sum(p.numel() for p in model.parameters()),
               "ema_tensors": len(ema.shadow_params), "strict_load": True, "finite": True}
        results["models"].append(row)
        print(json.dumps(row), flush=True)
        del state, model, ema
        gc.collect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Checked {len(results['models'])} score models and {len(results['arrays'])} arrays.")


if __name__ == "__main__":
    main()
