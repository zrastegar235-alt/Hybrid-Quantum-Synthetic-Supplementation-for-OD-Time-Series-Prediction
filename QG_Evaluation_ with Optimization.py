"""Quantum Generator + Classical Discriminator: OD hyperparameter optimization.

Preserves the supplied QGAN's Hadamards + efficient_su2 / SamplerQNN /
TorchConnector generator, weighted adversarial loss and alternating Adam steps.

Run on your Windows computer:
    python -m pip install numpy pandas matplotlib scipy torch qiskit qiskit-machine-learning openpyxl
    python QG_Optimized.py

Your real data1.csv (with an OD column) is required. This file does not
contain fabricated training results. Qiskit simulations can take considerable
computing time, especially with high SHOTS, REPS and N_EPOCHS.
"""
from __future__ import annotations

# =============================================================================
# 1. USER SETTINGS: EDIT ONLY THIS SECTION IN NORMAL USE
# =============================================================================
from pathlib import Path

FILE_PATH = Path(r"C:\Users\zrast\Desktop\Python Code\MOHAMMAD\GENERAL\SC1\EXCEL\data1.csv")
SHEET_NAME = 0                         # Used only for XLSX data input.
OD_COLUMN = "OD"

# Separate QG output so that the earlier CG workbook is not overwritten.
OUTPUT_DIR = Path(r"C:\Users\zrast\Desktop\Python Code\MOHAMMAD\GENERAL\SC1\EXCEL\Optimization\QG")
EXCEL_FILENAME = "Optimization_Results.xlsx"

RUN_MODE = "optimize"                 # "optimize" (search) or "single" (one configuration).
OPTIMIZATION_METHOD = "random"        # "random" recommended, or "grid" for a small search space.
NUM_TRIALS = 15                       # Includes the original baseline run.
RESUME_PREVIOUS_RUN = True            # Restart interrupted search if settings/data match.
SEED = 123456                         # Common seed for fairer comparison across trials.

# All eleven parameters supplied in the question. Change the lists to edit
# candidate values. N_EPOCHS is searched independently for every trial.
PARAMETER_VALUES = {
    "LR":                          [0.0001, 0.0003, 0.001, 0.003, 0.01],
    "NUM_BINS":                    [8, 16, 32, 64],
    "REPS":                        [1, 2, 3, 4, 6],
    "N_EPOCHS":                    [100, 200, 400, 600, 1000],
    "SHOTS":                       [1024, 2048, 4096, 8192, 16384],
    "G_WEIGHT_DECAY":              [0.0, 0.00001, 0.0001, 0.001, 0.005],
    "D_WEIGHT_DECAY":              [0.0, 0.0001, 0.001, 0.005, 0.01],
    "DISCRIMINATOR_HIDDEN_WIDTH":  [8, 16, 20, 32, 64],
    "ADAM_BETA1":                  [0.0, 0.5, 0.7, 0.9],
    "ADAM_BETA2":                  [0.9, 0.99, 0.999],
    "LEAKY_RELU_SLOPE":             [0.01, 0.05, 0.1, 0.2, 0.3],
}

# Values in your supplied original QG code, tested as trial 1 when included
# in the lists above. Edit this to run your chosen best parameters in single mode.
BASELINE_PARAMETERS = {
    "LR": 0.01,
    "NUM_BINS": 32,
    "REPS": 6,
    "N_EPOCHS": 200,
    "SHOTS": 4096,
    "G_WEIGHT_DECAY": 0.005,
    "D_WEIGHT_DECAY": 0.005,
    "DISCRIMINATOR_HIDDEN_WIDTH": 20,
    "ADAM_BETA1": 0.7,
    "ADAM_BETA2": 0.999,
    "LEAKY_RELU_SLOPE": 0.2,
}
SINGLE_RUN_PARAMETERS = dict(BASELINE_PARAMETERS)  # Replace with winning values.

# Evaluation options. Keep EVAL_NUM_BINS unchanged across all trials,
# even though NUM_BINS changes quantum qubit count / training distribution.
EVAL_NUM_BINS = 64                    # Shared evaluation grid for FAIR KL ranking.
KL_SMOOTHING = 1e-8                  # Probability added to each evaluation bin for KL.
N_SYNTHETIC = 1000                   # Independent synthetic OD samples per trial.
SAMPLE_WITHIN_BINS = True           # Original QG behavior: uniform inside picked bin.
SHOW_BEST_PLOT = False               # Save plots automatically; display only if True.
LOG_EVERY_N_EPOCHS = 50             # Console printing frequency (not a hyperparameter).

# =============================================================================
# 2. IMPORTS AND WORKBOOK DEFINITIONS -- normally no edits below this line
# =============================================================================
import csv
import hashlib
import itertools
import json
import math
import os
import random
import shutil
import time
import traceback
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from scipy.stats import entropy, ks_2samp, wasserstein_distance
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

PARAM_NAMES = list(PARAMETER_VALUES)
METRIC_NAMES = [
    "Generator_loss", "Discriminator_loss",
    "KL_generated_to_real",             # PRIMARY: on the same fixed evaluation grid.
    "CDF_Distance_FixedGrid",            # Tie break: difference in cumulative bin masses.
    "CDF_Distance_KS_Empirical",         # Two-sample empirical CDF distance.
    "Wasserstein_1_OD",
    "KL_model_bins_original",           # Diagnostic ONLY; NOT comparable across NUM_BINS.
]
REPORT_COLUMNS = (
    ["Run", "Status", "Best_Run", "Seed"] + PARAM_NAMES +
    ["n_qubits", "EVAL_NUM_BINS", "N_SYNTHETIC"] + METRIC_NAMES +
    ["Duration_seconds", "Error"]
)
CODE_VERSION = "QG_optimizer_fixed_grid_1"


def load_od_data(path: Path, sheet_name: int, column: str) -> np.ndarray:
    """Same OD import/cleaning, repeated samples preserved, negative OD rejected."""
    if not path.is_file():
        raise FileNotFoundError(f"Input data not found: {path}\nEdit FILE_PATH at top of script.")
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
    else:
        raise ValueError("FILE_PATH must end with .csv or .xlsx.")
    df.columns = [str(v).strip() for v in df.columns]
    if column not in df.columns:
        raise ValueError(f"Expected column {column!r}, found {list(df.columns)}")
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(values)
    print(f"OD rows: {len(values)}; excluded missing/invalid: {int((~mask).sum())}")
    values = values[mask]
    if len(values) < 2 or values.max() == values.min():
        raise ValueError("Need at least two distinct finite OD measurements.")
    if np.any(values < 0):
        raise ValueError("Negative OD measurements were found; check input data.")
    return values


def validate_settings() -> None:
    if RUN_MODE not in {"optimize", "single"}:
        raise ValueError("RUN_MODE must be 'optimize' or 'single'.")
    if OPTIMIZATION_METHOD not in {"random", "grid"}:
        raise ValueError("OPTIMIZATION_METHOD must be 'random' or 'grid'.")
    if NUM_TRIALS < 1 or EVAL_NUM_BINS < 2 or N_SYNTHETIC < 2:
        raise ValueError("NUM_TRIALS >= 1; EVAL_NUM_BINS and N_SYNTHETIC >= 2 are needed.")
    if KL_SMOOTHING <= 0 or LOG_EVERY_N_EPOCHS < 1:
        raise ValueError("KL_SMOOTHING and LOG_EVERY_N_EPOCHS must be positive.")
    if set(BASELINE_PARAMETERS) != set(PARAMETER_VALUES):
        raise ValueError("BASELINE_PARAMETERS keys do not match PARAMETER_VALUES.")
    if set(SINGLE_RUN_PARAMETERS) != set(PARAMETER_VALUES):
        raise ValueError("SINGLE_RUN_PARAMETERS keys do not match PARAMETER_VALUES.")
    for name, values in PARAMETER_VALUES.items():
        if not values or len(set(values)) != len(values):
            raise ValueError(f"{name}: enter a nonempty list of unique values.")
        for x in values:
            if not isinstance(x, (int, float)) or not math.isfinite(x):
                raise ValueError(f"{name}: invalid candidate {x!r}")
            check_param(name, x)
    for name, value in BASELINE_PARAMETERS.items():
        check_param(name, value)
    for name, value in SINGLE_RUN_PARAMETERS.items():
        check_param(name, value)


def check_param(name: str, value: float) -> None:
    if name in {"NUM_BINS", "REPS", "N_EPOCHS", "SHOTS", "DISCRIMINATOR_HIDDEN_WIDTH"}:
        if int(value) != value or value < 1:
            raise ValueError(f"{name} must be a positive integer; got {value}")
    if name == "NUM_BINS" and (value < 2 or (int(value) & (int(value)-1))):
        raise ValueError("NUM_BINS must be a power of two: e.g. 8, 16, 32, 64.")
    if name in {"LR", "LEAKY_RELU_SLOPE"} and value <= 0:
        raise ValueError(f"{name} must be positive.")
    if name in {"G_WEIGHT_DECAY", "D_WEIGHT_DECAY"} and value < 0:
        raise ValueError(f"{name} cannot be negative.")
    if name in {"ADAM_BETA1", "ADAM_BETA2"} and not 0 <= value < 1:
        raise ValueError(f"{name} must be in [0, 1).")


def planned_trials() -> list[dict]:
    if RUN_MODE == "single":
        return [dict(SINGLE_RUN_PARAMETERS)]
    total = math.prod(len(PARAMETER_VALUES[p]) for p in PARAM_NAMES)
    if OPTIMIZATION_METHOD == "grid":
        if total > NUM_TRIALS:
            raise ValueError(
                f"Full grid has {total:,} combinations but NUM_TRIALS={NUM_TRIALS}. "
                "Choose random search or reduce candidate lists."
            )
        return [dict(zip(PARAM_NAMES, x)) for x in
                itertools.product(*(PARAMETER_VALUES[k] for k in PARAM_NAMES))]
    n = min(total, NUM_TRIALS)
    rng = random.Random(SEED)
    used = set()
    trials: list[dict] = []
    if all(BASELINE_PARAMETERS[k] in PARAMETER_VALUES[k] for k in PARAM_NAMES):
        key = tuple(BASELINE_PARAMETERS[k] for k in PARAM_NAMES)
        used.add(key)
        trials.append(dict(BASELINE_PARAMETERS))
    while len(trials) < n:
        key = tuple(rng.choice(PARAMETER_VALUES[k]) for k in PARAM_NAMES)
        if key in used:
            continue
        used.add(key)
        trials.append(dict(zip(PARAM_NAMES, key)))
    return trials


def prepare_od_distribution(od: np.ndarray, num_bins: int):
    """Original QG histogram, bin centers and [-2, 2] discriminator grid."""
    counts, edges = np.histogram(od, bins=num_bins)
    target_prob = counts / counts.sum()
    centers = 0.5 * (edges[1:] + edges[:-1])
    scaled = 4 * (centers - edges[0]) / np.ptp(edges) - 2
    return target_prob, edges, centers, scaled.reshape(-1, 1)


class Discriminator(nn.Module):
    """Same single hidden layer and sigmoid as original QG model."""
    def __init__(self, hidden_width: int, slope: float):
        super().__init__()
        self.linear_input = nn.Linear(1, hidden_width)
        self.leaky_relu = nn.LeakyReLU(slope)
        self.linear20 = nn.Linear(hidden_width, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.linear20(self.leaky_relu(self.linear_input(x))))


def adversarial_loss(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Original weighted binary cross-entropy without changing its meaning."""
    pred = torch.clamp(pred, min=1e-7, max=1-1e-7)
    return -torch.sum(weights * (target * torch.log(pred) + (1-target) * torch.log(1-pred)))


def normalize_probabilities(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float).reshape(-1)
    if p.size == 0 or not np.isfinite(p).all() or p.min() < -1e-10 or p.sum() <= 0:
        raise ValueError("Generator returned invalid output probabilities.")
    p = np.maximum(p, 0)
    return p / p.sum()


def sample_synthetic_od(probs: np.ndarray, edges: np.ndarray, n: int,
                        rng: np.random.Generator, within_bins: bool) -> np.ndarray:
    probs = normalize_probabilities(probs)
    if len(probs) != len(edges)-1:
        raise ValueError("Generated probability count does not equal the OD bins.")
    choices = rng.choice(len(probs), size=n, p=probs)
    if within_bins:
        return rng.uniform(edges[choices], edges[choices+1])
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers[choices]


def rebin_probabilities(probs: np.ndarray, model_edges: np.ndarray,
                        evaluation_edges: np.ndarray, within_bins: bool) -> np.ndarray:
    """Project model histogram to a SHARED reference grid, without sampling noise.

    Inside-bin uniformity is exactly the same extra assumption as the original
    SAMPLE_WITHIN_BINS=True QG synthetic-data generation method.
    """
    probs = normalize_probabilities(probs)
    out = np.zeros(len(evaluation_edges)-1, dtype=float)
    if not within_bins:
        centers = 0.5*(model_edges[:-1]+model_edges[1:])
        out, _ = np.histogram(centers, bins=evaluation_edges, weights=probs)
        return normalize_probabilities(out)
    for mass, left, right in zip(probs, model_edges[:-1], model_edges[1:]):
        # Proportional contribution from every overlap of model and reference bins.
        overlap = np.maximum(0.0, np.minimum(evaluation_edges[1:], right)
                                 - np.maximum(evaluation_edges[:-1], left))
        out += mass * overlap / (right-left)
    return normalize_probabilities(out)


def evaluate_trial(od: np.ndarray, synthetic: np.ndarray,
                   model_p: np.ndarray, model_edges: np.ndarray,
                   model_real_prob: np.ndarray, evaluation_edges: np.ndarray) -> dict:
    """Fixed-bin KL is comparable even if different runs use 8 vs 64 model bins."""
    real_fixed, _ = np.histogram(od, bins=evaluation_edges)
    real_fixed = real_fixed / real_fixed.sum()
    generated_fixed = rebin_probabilities(model_p, model_edges, evaluation_edges,
                                          SAMPLE_WITHIN_BINS)
    eps = KL_SMOOTHING
    kl_fixed = float(entropy(generated_fixed + eps, real_fixed + eps))
    original_model_kl = float(entropy(model_p + 1e-12, model_real_prob + 1e-12))
    cdf_fixed = float(np.max(np.abs(np.cumsum(real_fixed) - np.cumsum(generated_fixed))))
    empirical_ks = float(ks_2samp(od, synthetic).statistic)
    return {
        "KL_generated_to_real": kl_fixed,
        "CDF_Distance_FixedGrid": cdf_fixed,
        "CDF_Distance_KS_Empirical": empirical_ks,
        "Wasserstein_1_OD": float(wasserstein_distance(od, synthetic)),
        "KL_model_bins_original": original_model_kl,
    }


def train_one_trial(od: np.ndarray, params: dict, trial_dir: Path, seed: int) -> dict:
    """Rebuild a new QGAN for EACH trial; preserve original loss and optimizer order."""
    # Delay Qiskit imports so config/reporting helpers are testable independently.
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import efficient_su2
    from qiskit.primitives import StatevectorSampler
    from qiskit.quantum_info import Statevector
    from qiskit_machine_learning.connectors import TorchConnector
    from qiskit_machine_learning.neural_networks import SamplerQNN
    from qiskit_machine_learning.utils import algorithm_globals

    start = time.perf_counter()
    torch.manual_seed(seed)
    np.random.seed(seed)
    algorithm_globals.random_seed = seed
    rng = np.random.default_rng(seed)
    n_bins = int(params["NUM_BINS"])
    n_qubits = n_bins.bit_length()-1
    target, model_edges, centers, grid = prepare_od_distribution(od, n_bins)
    fixed_edges = np.linspace(float(od.min()), float(od.max()), EVAL_NUM_BINS+1)

    # EXACT original QGAN generator design: H gates, efficient_su2, SamplerQNN.
    qc = QuantumCircuit(n_qubits)
    qc.h(qc.qubits)
    qc.compose(efficient_su2(n_qubits, reps=int(params["REPS"])), inplace=True)
    sampler = StatevectorSampler(default_shots=int(params["SHOTS"]), seed=seed)
    qnn = SamplerQNN(circuit=qc, sampler=sampler, input_params=[],
                     weight_params=qc.parameters, sparse=False)
    initial = algorithm_globals.random.random(qc.num_parameters)
    generator = TorchConnector(qnn, initial)
    discriminator = Discriminator(int(params["DISCRIMINATOR_HIDDEN_WIDTH"]),
                                  float(params["LEAKY_RELU_SLOPE"]))
    lr = float(params["LR"])
    betas = (float(params["ADAM_BETA1"]), float(params["ADAM_BETA2"]))
    generator_optimizer = Adam(generator.parameters(), lr=lr, betas=betas,
                               weight_decay=float(params["G_WEIGHT_DECAY"]))
    discriminator_optimizer = Adam(discriminator.parameters(), lr=lr, betas=betas,
                                   weight_decay=float(params["D_WEIGHT_DECAY"]))

    real_dist = torch.tensor(target, dtype=torch.float32).reshape(-1, 1)
    positions = torch.tensor(grid, dtype=torch.float32)
    valid = torch.ones(n_bins, 1, dtype=torch.float32)
    fake = torch.zeros(n_bins, 1, dtype=torch.float32)
    empty_input = torch.empty(0, dtype=torch.float32)
    history = []
    epochs = int(params["N_EPOCHS"])
    for epoch in range(epochs):
        disc_value = discriminator(positions)
        gen_dist = generator(empty_input).reshape(-1, 1)

        # Same generator weighted BCE, but disc_value is detached to prevent
        # unused discriminator gradients (mathematically identical for G).
        generator_optimizer.zero_grad()
        generator_loss = adversarial_loss(disc_value.detach(), valid, gen_dist)
        generator_loss.backward()
        generator_optimizer.step()

        # Same discriminator weighted real/fake BCE and original step order.
        # As in the uploaded code, D uses the generator distribution computed
        # BEFORE this epoch's generator update.
        discriminator_optimizer.zero_grad()
        real_loss = adversarial_loss(disc_value, valid, real_dist)
        fake_loss = adversarial_loss(disc_value, fake, gen_dist.detach())
        discriminator_loss = (real_loss + fake_loss) / 2.0
        discriminator_loss.backward()
        discriminator_optimizer.step()

        g_loss, d_loss = float(generator_loss.item()), float(discriminator_loss.item())
        model_p = normalize_probabilities(gen_dist.detach().cpu().numpy())
        epoch_kl = float(entropy(model_p+1e-12, target+1e-12))
        if not np.isfinite([g_loss, d_loss, epoch_kl]).all():
            raise RuntimeError("Non-finite training loss/KL; check learning rate.")
        history.append([epoch+1, g_loss, d_loss, epoch_kl])
        if epoch == 0 or (epoch+1) % LOG_EVERY_N_EPOCHS == 0 or epoch+1 == epochs:
            print(f"   Epoch {epoch+1}/{epochs}: G={g_loss:.5g}, "
                  f"D={d_loss:.5g}, model-bin KL={epoch_kl:.5g}", flush=True)

    # Evaluate the FINAL trained circuit exactly (NO extra finite-shot noise).
    # SHOTS remains a training hyperparameter; this makes comparison fairer.
    generator.eval()
    weights = generator.weight.detach().cpu().numpy().reshape(-1)
    bound = qc.assign_parameters(dict(zip(qc.parameters, weights)))
    model_p = normalize_probabilities(Statevector.from_instruction(bound).probabilities())
    synthetic = sample_synthetic_od(model_p, model_edges, N_SYNTHETIC, rng,
                                    SAMPLE_WITHIN_BINS)
    metrics = evaluate_trial(od, synthetic, model_p, model_edges, target, fixed_edges)
    metrics["Generator_loss"] = history[-1][1]
    metrics["Discriminator_loss"] = history[-1][2]
    metrics["Duration_seconds"] = time.perf_counter()-start
    metrics["n_qubits"] = n_qubits

    # Save per-trial details so the best run can be inspected/reproduced later.
    trial_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history, columns=["Epoch", "Generator_loss", "Discriminator_loss",
                                    "KL_model_bins_original"]).to_csv(
        trial_dir / "training_history.csv", index=False)
    pd.DataFrame({"OD_bin_left": model_edges[:-1], "OD_bin_right": model_edges[1:],
                  "OD_bin_center": centers, "Real_probability": target,
                  "Generated_probability": model_p}).to_csv(
        trial_dir / "learned_OD_distribution.csv", index=False)
    pd.DataFrame({"OD": synthetic, "Source": "synthetic"}).to_csv(
        trial_dir / "synthetic_OD.csv", index=False)
    pd.concat([pd.DataFrame({"OD": od, "Source": "real"}),
               pd.DataFrame({"OD": synthetic, "Source": "synthetic"})],
              ignore_index=True).to_csv(trial_dir / "OD_real_and_synthetic.csv", index=False)
    torch.save({"generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "circuit_weights": weights,
                "parameters": params, "od_min": float(od.min()),
                "od_max": float(od.max()), "seed": seed}, trial_dir / "trained_models.pt")
    (trial_dir / "parameters_and_metrics.json").write_text(
        json.dumps({**params, **metrics, "Seed": seed}, indent=2, default=float),
        encoding="utf-8")
    plot_trial(od, synthetic, model_edges, history, metrics,
               trial_dir / "OD_comparison.png")
    return metrics


def plot_trial(od: np.ndarray, synthetic: np.ndarray, edges: np.ndarray,
              history: list, metrics: dict, save_path: Path) -> None:
    h = np.asarray(history)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(h[:,0], h[:,1], label="Generator")
    axes[0, 0].plot(h[:,0], h[:,2], label="Discriminator")
    axes[0, 0].set(xlabel="Epoch", ylabel="Weighted BCE", title="Training losses")
    axes[0, 0].legend()
    axes[0, 1].plot(h[:,0], h[:,3])
    axes[0, 1].set(xlabel="Epoch", ylabel="Model-bin KL",
                   title="Original KL diagnostic (model's own bins)")
    axes[1, 0].hist(od, bins=edges, density=True, alpha=.55, label="Real")
    axes[1, 0].hist(synthetic, bins=edges, density=True, alpha=.55, label="Synthetic")
    axes[1, 0].set(xlabel="OD", ylabel="Density", title="Histogram comparison")
    axes[1, 0].legend()
    for arr, label in [(od, "Real"), (synthetic, "Synthetic")]:
        ordered = np.sort(arr)
        axes[1,1].step(ordered, np.arange(1, len(ordered)+1)/len(ordered),
                       where="post", label=label)
    axes[1,1].set(xlabel="OD", ylabel="Cumulative probability",
                   title=f"Empirical CDF (KS={metrics['CDF_Distance_KS_Empirical']:.4g})")
    axes[1,1].legend()
    fig.suptitle(f"QG | fixed-grid KL={metrics['KL_generated_to_real']:.5g} "
                 f"| fixed-grid CDF={metrics['CDF_Distance_FixedGrid']:.5g}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    if SHOW_BEST_PLOT and save_path.parent.name == "Best_Run":
        plt.show()
    plt.close(fig)


def best_row(rows: list[dict]) -> dict | None:
    completed = [r for r in rows if r.get("Status") == "Completed"
                 and all(math.isfinite(float(r[k])) for k in
                         ("KL_generated_to_real", "CDF_Distance_FixedGrid"))]
    if not completed:
        return None
    return min(completed, key=lambda r: (float(r["KL_generated_to_real"]),
                                         float(r["CDF_Distance_FixedGrid"]), int(r["Run"])))


def write_checkpoint_csv(rows: list[dict], path: Path) -> None:
    """CSV is written first, atomically, even if Excel is open/locked."""
    tmp = path.with_name(path.name+".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def read_checkpoint_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    converted = []
    integer_cols = {"Run", "Seed", "n_qubits", "EVAL_NUM_BINS", "N_SYNTHETIC"} | {
        "NUM_BINS", "REPS", "N_EPOCHS", "SHOTS", "DISCRIMINATOR_HIDDEN_WIDTH"}
    numeric_cols = set(PARAM_NAMES) | set(METRIC_NAMES) | integer_cols | {"Duration_seconds"}
    for row in rows:
        obj = {}
        for k in REPORT_COLUMNS:
            v = row.get(k, "")
            if k in numeric_cols and v not in ("", None):
                obj[k] = int(float(v)) if k in integer_cols else float(v)
            else:
                obj[k] = v
        converted.append(obj)
    return converted


def xlsx_value(value):
    if value is None or value == "":
        return None
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def style_sheet(ws, columns: list[str], rows_count: int) -> None:
    ws.freeze_panes = "E2" if ws.title == "All Optimization Results" else "A2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(columns))}{max(rows_count+1,2)}"
    for cell in ws[1]:
        cell.fill = PatternFill("solid", fgColor="173653")
        cell.font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.row_dimensions[1].height = 42
    for j, heading in enumerate(columns, 1):
        width = min(39, max(13, len(heading)+2))
        if heading == "Error": width = 55
        ws.column_dimensions[get_column_letter(j)].width = width
    for i in range(2, rows_count+2):
        if i % 2 == 0:
            for cell in ws[i]:
                cell.fill = PatternFill("solid", fgColor="EFF5FB")
        ws.row_dimensions[i].height = 19
    if ws.title == "All Optimization Results" and "Best_Run" in columns:
        best_col = columns.index("Best_Run")+1
        for row in ws.iter_rows(min_row=2, max_row=rows_count+1):
            if str(row[best_col-1].value) == "YES":
                for cell in row:
                    cell.fill = PatternFill("solid", fgColor="DCF4E4")
                    cell.font = Font(name="Calibri", bold=True, color="16613C")
    for row in ws.iter_rows(min_row=2, max_row=rows_count+1):
        for cell in row:
            if isinstance(cell.value, float):
                cell.number_format = "0.00000000"


def write_excel(rows: list[dict], trials: list[dict], path: Path, mode: str) -> None:
    """Three worksheets; writes after every trial through atomic replace."""
    best = best_row(rows)
    b_id = int(best["Run"]) if best else None
    wb = Workbook()
    all_ws = wb.active
    all_ws.title = "All Optimization Results"
    all_ws.append(REPORT_COLUMNS)
    for row in rows:
        values = dict(row)
        values["Best_Run"] = "YES" if b_id == int(row["Run"]) else ""
        all_ws.append([xlsx_value(values.get(k)) for k in REPORT_COLUMNS])
    style_sheet(all_ws, REPORT_COLUMNS, len(rows))

    best_ws = wb.create_sheet("Best Parameters")
    best_ws.append(["Parameter / Metric", "Winning value"])
    if best:
        for name in ["Run", "Seed"] + PARAM_NAMES + ["n_qubits", "EVAL_NUM_BINS", "N_SYNTHETIC"] + METRIC_NAMES:
            best_ws.append([name, xlsx_value(best.get(name))])
    else:
        best_ws.append(["Status", "No successful trials yet"])
    style_sheet(best_ws, ["Parameter / Metric", "Winning value"], best_ws.max_row-1)
    best_ws.column_dimensions["A"].width = 39
    best_ws.column_dimensions["B"].width = 28

    summary_ws = wb.create_sheet("Optimization Summary")
    summary_ws.append(["Setting / Summary", "Value"])
    completed = [r for r in rows if r["Status"] == "Completed"]
    failed = [r for r in rows if r["Status"] == "Failed"]
    summary = [
        ("Algorithm", "QGAN: quantum generator, classical discriminator"),
        ("Mode", mode),
        ("Optimization method", OPTIMIZATION_METHOD if mode == "optimize" else "single"),
        ("Total planned trials", len(trials)),
        ("Attempted trials", len(rows)),
        ("Successful trials", len(completed)),
        ("Failed trials", len(failed)),
        ("Winner selection", "Lowest fixed-grid KL; then lowest fixed-grid CDF distance"),
        ("Comparability", "All trials evaluated on identical OD bin edges"),
        ("EVAL_NUM_BINS", EVAL_NUM_BINS),
        ("KL_SMOOTHING", KL_SMOOTHING),
        ("N_SYNTHETIC", N_SYNTHETIC),
        ("SAMPLE_WITHIN_BINS", str(SAMPLE_WITHIN_BINS)),
        ("Validation reference", "Original real OD data (training reference)"),
        ("Final evaluation", "Exact circuit statevector; SHOTS used in training"),
        ("Date updated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    ]
    if best:
        summary += [("Best run", int(best["Run"])),
                    ("Best fixed-grid KL", float(best["KL_generated_to_real"])),
                    ("Best CDF distance", float(best["CDF_Distance_FixedGrid"]))]
    if completed:
        summary += [("Median fixed-grid KL", float(np.median([float(x["KL_generated_to_real"]) for x in completed]))),
                    ("Worst fixed-grid KL", float(max(float(x["KL_generated_to_real"]) for x in completed)))]
    summary.append(("", ""))
    summary.append(("Candidate ranges", "Tested values"))
    for k, vs in PARAMETER_VALUES.items():
        summary.append((k, ", ".join(str(x) for x in vs)))
    for name, val in summary:
        summary_ws.append([name, val])
    style_sheet(summary_ws, ["Setting / Summary", "Value"], summary_ws.max_row-1)
    summary_ws.column_dimensions["A"].width = 39
    summary_ws.column_dimensions["B"].width = 98
    for row in summary_ws:
        row[1].alignment = Alignment(vertical="center", wrap_text=True)

    tmp = path.with_name(path.stem+"._writing_.xlsx")
    wb.save(tmp)
    os.replace(tmp, path)


def experiment_fingerprint(od: np.ndarray, trial_list: list[dict]) -> str:
    payload = {"code": CODE_VERSION, "trials": trial_list,
               "seed": SEED, "eval_bins": EVAL_NUM_BINS, "kl_smoothing": KL_SMOOTHING,
               "n_synthetic": N_SYNTHETIC, "within_bins": SAMPLE_WITHIN_BINS,
               "method": OPTIMIZATION_METHOD, "mode": RUN_MODE,
               "file": str(FILE_PATH.resolve()), "column": OD_COLUMN,
               "data_hash": hashlib.sha256(np.asarray(od, dtype=np.float64).tobytes()).hexdigest()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf8")).hexdigest()


def copy_best_run(trial_dir: Path, output_dir: Path, row: dict) -> None:
    best_dir = output_dir / "Best_Run"
    if best_dir.exists():
        shutil.rmtree(best_dir)
    shutil.copytree(trial_dir, best_dir)
    (output_dir / "Best_Parameters.txt").write_text(
        "QG winning settings (paste into SINGLE_RUN_PARAMETERS):\n\n"
        + "SINGLE_RUN_PARAMETERS = "
        + repr({k: row[k] for k in PARAM_NAMES})
        + f"\n\nRun: {row['Run']}\n"
        + f"Fixed-grid KL: {row['KL_generated_to_real']}\n"
        + f"Fixed-grid CDF distance: {row['CDF_Distance_FixedGrid']}\n",
        encoding="utf-8")


def main() -> None:
    validate_settings()
    od = load_od_data(FILE_PATH, SHEET_NAME, OD_COLUMN)
    trials = planned_trials()
    base_dir = OUTPUT_DIR if RUN_MODE == "optimize" else OUTPUT_DIR / "Single_Run"
    base_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = experiment_fingerprint(od, trials)
    manifest_file = base_dir / "experiment_manifest.json"
    checkpoint = base_dir / "Optimization_Results_checkpoint.csv"
    workbook = base_dir / (EXCEL_FILENAME if RUN_MODE == "optimize" else "Single_Run_Results.xlsx")
    exists = manifest_file.is_file()
    if RESUME_PREVIOUS_RUN and exists:
        manifest = json.loads(manifest_file.read_text(encoding="utf8"))
        if manifest.get("fingerprint") != fingerprint:
            raise RuntimeError(
                f"Settings or OD input changed since the last run in {base_dir}. "
                "Use a different OUTPUT_DIR, or set RESUME_PREVIOUS_RUN=False "
                "to intentionally start a NEW experiment."
            )
        rows = read_checkpoint_csv(checkpoint)
        if len(rows) > len(trials):
            raise RuntimeError("Checkpoint has more rows than current planned trials.")
        print(f"Resuming {len(rows)} completed/failed attempts from checkpoint.")
    else:
        rows = []
        if (base_dir / "Best_Run").exists():
            shutil.rmtree(base_dir / "Best_Run")
        if (base_dir / "Trial_Artifacts").exists():
            shutil.rmtree(base_dir / "Trial_Artifacts")
        for p in [checkpoint, workbook, base_dir / "Best_Parameters.txt"]:
            if p.exists():
                p.unlink()
        manifest_file.write_text(json.dumps({"fingerprint": fingerprint,
             "created": datetime.now().isoformat(), "planned": len(trials),
             "mode": RUN_MODE, "method": OPTIMIZATION_METHOD}, indent=2), encoding="utf8")
    if not exists and not manifest_file.exists():
        manifest_file.write_text(json.dumps({"fingerprint": fingerprint}, indent=2), encoding="utf8")

    print(f"\nAlgorithm: QG | {RUN_MODE} | method: {OPTIMIZATION_METHOD}")
    print(f"Planned trials: {len(trials)} | Fixed evaluation bins: {EVAL_NUM_BINS}")
    print(f"Excel: {workbook}\n")
    total_started = time.perf_counter()
    completed_ids = {int(r["Run"]) for r in rows}
    for run, params in enumerate(trials, start=1):
        if run in completed_ids:
            continue
        print(f"\n{'='*68}\nRUN {run}/{len(trials)}: {params}\n{'='*68}", flush=True)
        record: dict = {k: params[k] for k in PARAM_NAMES}
        record.update({"Run": run, "Status": "Failed", "Best_Run": "",
                       "Seed": SEED, "n_qubits": int(math.log2(params["NUM_BINS"])),
                       "EVAL_NUM_BINS": EVAL_NUM_BINS, "N_SYNTHETIC": N_SYNTHETIC,
                       "Error": ""})
        trial_dir = base_dir / "Trial_Artifacts" / f"Run_{run:03d}"
        start = time.perf_counter()
        try:
            metrics = train_one_trial(od, params, trial_dir, SEED)
            record.update(metrics)
            record["Status"] = "Completed"
            print(f"Completed run {run}: fixed KL={metrics['KL_generated_to_real']:.7g}, "
                  f"fixed CDF={metrics['CDF_Distance_FixedGrid']:.7g}")
        except (KeyboardInterrupt, SystemExit):
            # Preserve a completed-trial checkpoint, then respect user interruption.
            write_checkpoint_csv(rows, checkpoint)
            try:
                write_excel(rows, trials, workbook, RUN_MODE)
            except OSError as exc:
                print(f"Excel checkpoint unavailable: {exc}; CSV remains intact.")
            raise
        except Exception as exc:
            record["Error"] = f"{type(exc).__name__}: {exc}"
            print(f"Trial {run} FAILED: {record['Error']}")
            print(traceback.format_exc(limit=3))
        record.setdefault("Duration_seconds", time.perf_counter()-start)
        rows.append(record)
        # FIRST save CSV; opening the Excel workbook in Excel won't lose data.
        write_checkpoint_csv(rows, checkpoint)
        prior_best = best_row(rows[:-1])
        now_best = best_row(rows)
        if record["Status"] == "Completed" and now_best is record:
            try:
                copy_best_run(trial_dir, base_dir, record)
                print(f"NEW BEST run {run}; saved trained model and plots.")
            except OSError as exc:
                print(f"Warning: couldn't update best artifacts: {exc}")
        try:
            write_excel(rows, trials, workbook, RUN_MODE)
        except PermissionError as exc:
            print(f"IMPORTANT: Close Excel to unlock {workbook}; checkpoint CSV is safe. ({exc})")
        except OSError as exc:
            print(f"Excel could not save; checkpoint CSV is safe: {exc}")

    # Rebuild Excel from checkpoint if it was temporarily locked earlier.
    final_best = best_row(rows)
    try:
        write_excel(rows, trials, workbook, RUN_MODE)
    except PermissionError:
        print("Close the Excel workbook, then rerun to rebuild it from checkpoint.csv.")
    print(f"\nCompleted {len(rows)}/{len(trials)} attempted trial(s) "
          f"in {time.perf_counter()-total_started:.1f} seconds.")
    if final_best is None:
        print("No successful trial. Open Excel/CSV 'Error' column; correct environment or settings.")
    else:
        print("\nBEST QG PARAMETERS (lowest fixed-grid KL):")
        print(f"Run: {final_best['Run']}")
        for name in PARAM_NAMES:
            print(f"  {name:30s} = {final_best[name]}")
        for name in METRIC_NAMES:
            print(f"  {name:30s} = {float(final_best[name]):.8g}")
        print(f"Best model and plots: {base_dir / 'Best_Run'}")
    print(f"Excel results: {workbook}")
    if SHOW_BEST_PLOT and final_best is not None:
        best_plot = base_dir / "Best_Run" / "OD_comparison.png"
        if best_plot.exists():
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.imshow(plt.imread(best_plot))
            ax.axis("off")
            plt.show()
            plt.close(fig)
    print(f"Emergency CSV checkpoint: {checkpoint}")
    print("Metrics compare generated OD against the same real OD used for training.")


if __name__ == "__main__":
    main()
