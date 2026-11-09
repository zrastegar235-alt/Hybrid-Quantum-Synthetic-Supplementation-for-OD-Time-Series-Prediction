"""Classical Generator + two-stage Quantum Discriminator for scalar OD.

Runs a baseline + seeded random hyperparameter search, evaluates each trained
model, checkpoints *every* completed/failed trial, and writes a formatted Excel
report with the best trial. Requires a real CSV/XLSX file containing an OD column.

Install on your Windows computer (in your chosen Python environment):
    python -m pip install numpy pandas matplotlib torch pennylane openpyxl
Run:
    python CG_Optimized.py
"""

from __future__ import annotations

# ============================================================================
# 1. USER CONFIGURATION — Change settings here; no model edits are necessary.
# ============================================================================
from pathlib import Path

# Your original OD input file. CSV or XLSX is supported.
FILE_PATH = Path(r"C:\Users\zrast\Desktop\Python Code\MOHAMMAD\GENERAL\SC1\EXCEL\data1.csv")
SHEET_NAME = 0                       # Excel input only: 0 is the first worksheet.
OD_COLUMN = "OD"                    # Column to model in the input data.

# Location and name of the Excel file produced while optimization runs.
OUTPUT_DIR = Path(r"C:\Users\zrast\Desktop\Python Code\MOHAMMAD\GENERAL\SC1\EXCEL\Optimization")
EXCEL_FILENAME = "Optimization_Results.xlsx"

OPTIMIZATION_METHOD = "random"       # "random" (recommended) or "grid".
NUM_TRIALS = 15                      # Max combinations for random search; includes baseline.
RESUME_PREVIOUS_RUN = True           # Continue interrupted runs if settings/data are unchanged.
SEED = 123456                        # Repeatable search/training randomness (same seed per trial).
TORCH_CPU_THREADS = 1                # Small quantum simulators often work well with one thread.

# Candidate values (and ranges) for optimization. To change the training epoch
# budget, edit the "EPOCHS" list. Each trial uses its selected epoch count.
# Both generator hidden layers use GENERATOR_HIDDEN_WIDTH (originally 32, 32).
PARAMETER_VALUES = {
    "G_LEARNING_RATE":       [0.0001, 0.0003, 0.001, 0.003],
    "D_LEARNING_RATE":       [0.0001, 0.0003, 0.001, 0.003, 0.01],
    "BATCH_SIZE":            [16, 32, 64, 128],
    "EPOCHS":                [100, 200, 400, 600, 1000],
    "LATENT_DIM":            [1, 2, 4, 8, 16],
    "n_layers":              [1, 2, 3, 4],
    "n_2nd_layers":          [1, 2, 3, 4, 6],
    "n_qubits":              [2, 3, 4],
    "n_2nd_qubits":          [2, 3, 4],
    "GENERATOR_HIDDEN_WIDTH": [16, 32, 64, 128],
    "LEAKY_RELU_SLOPE":      [0.01, 0.05, 0.1, 0.2, 0.3],
    "ADAM_BETA1":            [0.0, 0.5, 0.7, 0.9],
    "ADAM_BETA2":            [0.9, 0.99, 0.999],
}

# Original CG model defaults. Tested first if ALL of them occur in the lists.
BASELINE_PARAMETERS = {
    "G_LEARNING_RATE": 0.001, "D_LEARNING_RATE": 0.01,
    "BATCH_SIZE": 64, "EPOCHS": 200, "LATENT_DIM": 8,
    "n_layers": 1, "n_2nd_layers": 3,
    "n_qubits": 2, "n_2nd_qubits": 2,
    "GENERATOR_HIDDEN_WIDTH": 32, "LEAKY_RELU_SLOPE": 0.2,
    "ADAM_BETA1": 0.5, "ADAM_BETA2": 0.999,
}

# Evaluation settings; use the SAME histogram bins for EVERY trial. Searching
# bin counts would make the resulting KL values incomparable.
N_SYNTHETIC = 1000                  # Number of fake OD samples per evaluation.
NUM_BINS = 24                       # Original plot: 25 EDGES = 24 bins.
KL_PSEUDOCOUNT = 1e-6               # Prevents log(0) in KL for empty histogram bins.
CHECK_QUANTUM_INPUT_GRADIENT = True  # Detects models that cannot train the generator.
LOG_EVERY_N_EPOCHS = 100            # Console training log frequency; not a training parameter.
SHOW_BEST_PLOT = False              # Set True to display best-run plots when finished.

# Original quantum measurement indices and amplitude padding are unchanged.
MEASURED_QUBIT_IDX = 0
MEASURED_QUBIT_2ND_IDX = 1
AMPLITUDE_PADDING = 0.001

# ============================================================================
# 2. LIBRARIES AND FIXED HELPERS — Normally you do not need to edit below.
# ============================================================================
import csv
import hashlib
import itertools
import json
import math
import os
import random
import time
import traceback
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml
import torch
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from torch import nn
from torch.utils.data import DataLoader


PARAMETER_NAMES = list(PARAMETER_VALUES)
METRIC_NAMES = [
    "Generator_loss", "Discriminator_loss", "KL_generated_to_real",
    "CDF_Distance_KS",
]
REPORT_COLUMNS = (
    ["Run", "Status", "Best_Run", "Seed"] + PARAMETER_NAMES +
    ["NUM_BINS", "N_SYNTHETIC"] + METRIC_NAMES +
    ["Duration_seconds", "Error"]
)
CODE_VERSION = "CG_optimization_v1"
NAVY = "16324F"
BLUE = "EAF2FA"
GRAY = "F3F6F8"
WHITE = "FFFFFF"
GREEN = "E4F3EA"


def load_od(path: Path) -> np.ndarray:
    """Keep the supplied CSV/XLSX reader and original OD filtering."""
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}\nEdit FILE_PATH at the top of the script.")
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path, sheet_name=SHEET_NAME, engine="openpyxl")
    else:
        raise ValueError("FILE_PATH must point to a .csv or .xlsx file.")
    df.columns = [str(name).strip() for name in df.columns]
    if OD_COLUMN not in df.columns:
        raise ValueError(f"Column {OD_COLUMN!r} not found. Available columns: {list(df.columns)}")
    values = pd.to_numeric(df[OD_COLUMN], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(values)
    print(f"Real OD rows: {len(values)}; excluded invalid/missing: {int((~ok).sum())}")
    values = values[ok]
    if len(values) < 2 or float(values.min()) == float(values.max()):
        raise ValueError("At least two finite, different OD values are required.")
    return values


def validate_settings() -> None:
    if OPTIMIZATION_METHOD not in {"random", "grid"}:
        raise ValueError("OPTIMIZATION_METHOD must be 'random' or 'grid'.")
    if NUM_TRIALS < 1 or N_SYNTHETIC < 1 or NUM_BINS < 2 or KL_PSEUDOCOUNT <= 0:
        raise ValueError("NUM_TRIALS, N_SYNTHETIC and KL_PSEUDOCOUNT must be positive; NUM_BINS >= 2.")
    if TORCH_CPU_THREADS < 1 or LOG_EVERY_N_EPOCHS < 1:
        raise ValueError("TORCH_CPU_THREADS and LOG_EVERY_N_EPOCHS must be positive.")
    for name, choices in PARAMETER_VALUES.items():
        if not choices or len(set(choices)) != len(choices):
            raise ValueError(f"Parameter {name} needs a nonempty list of UNIQUE values.")
        for value in choices:
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Invalid numeric value in {name}: {value!r}")
    for name in ("G_LEARNING_RATE", "D_LEARNING_RATE", "LEAKY_RELU_SLOPE"):
        if any(v <= 0 for v in PARAMETER_VALUES[name]):
            raise ValueError(f"{name} values must be positive.")
    for name in ("BATCH_SIZE", "EPOCHS", "LATENT_DIM", "n_layers", "n_2nd_layers", "GENERATOR_HIDDEN_WIDTH"):
        if any(int(v) != v or v < 1 for v in PARAMETER_VALUES[name]):
            raise ValueError(f"{name} values must be positive integers.")
    for name in ("n_qubits", "n_2nd_qubits"):
        if any(int(v) != v or v < 2 for v in PARAMETER_VALUES[name]):
            raise ValueError(f"{name} values must be integers >= 2 for entangling layers.")
    for name in ("ADAM_BETA1", "ADAM_BETA2"):
        if any(v < 0 or v >= 1 for v in PARAMETER_VALUES[name]):
            raise ValueError(f"{name} must be between 0 (inclusive) and 1 (exclusive).")
    if min(PARAMETER_VALUES["n_qubits"]) <= MEASURED_QUBIT_IDX:
        raise ValueError("First-stage measurement index is outside the available wires.")
    if min(PARAMETER_VALUES["n_2nd_qubits"]) <= MEASURED_QUBIT_2ND_IDX:
        raise ValueError("Second-stage measurement index is outside the available wires.")


def planned_trials() -> list[dict]:
    """Random search samples distinct full configurations, with baseline first."""
    choices = [PARAMETER_VALUES[name] for name in PARAMETER_NAMES]
    total = math.prod(len(items) for items in choices)
    if OPTIMIZATION_METHOD == "grid":
        if total > NUM_TRIALS:
            raise ValueError(
                f"Full grid has {total:,} combinations but NUM_TRIALS={NUM_TRIALS}. "
                "Reduce PARAMETER_VALUES, increase NUM_TRIALS, or use 'random'."
            )
        return [dict(zip(PARAMETER_NAMES, values)) for values in itertools.product(*choices)]
    rng = random.Random(SEED)
    limit = min(NUM_TRIALS, total)
    result = []
    seen = set()
    baseline_allowed = all(BASELINE_PARAMETERS[name] in PARAMETER_VALUES[name] for name in PARAMETER_NAMES)
    if baseline_allowed:
        key = tuple(BASELINE_PARAMETERS[name] for name in PARAMETER_NAMES)
        seen.add(key)
        result.append(dict(BASELINE_PARAMETERS))
    while len(result) < limit:
        values = tuple(rng.choice(items) for items in choices)
        if values in seen:
            continue
        seen.add(values)
        result.append(dict(zip(PARAMETER_NAMES, values)))
    return result


class HybridModel(nn.Module):
    """Original two-stage quantum discriminator, rebuilt for each trial."""

    def __init__(self, params: dict):
        super().__init__()
        nq = int(params["n_qubits"])
        nq2 = int(params["n_2nd_qubits"])
        dev = qml.device("default.qubit", wires=nq, shots=None)
        dev2 = qml.device("default.qubit", wires=nq2, shots=None)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def first_stage(inputs, weights):
            qml.AmplitudeEmbedding(inputs, wires=range(nq),
                                   pad_with=AMPLITUDE_PADDING, normalize=True)
            qml.StronglyEntanglingLayers(weights, wires=range(nq))
            return [qml.expval(qml.PauliZ(MEASURED_QUBIT_IDX))]

        @qml.qnode(dev2, interface="torch", diff_method="backprop")
        def second_stage(inputs, weights):
            qml.AmplitudeEmbedding(inputs, wires=range(nq2),
                                   pad_with=AMPLITUDE_PADDING, normalize=True)
            qml.StronglyEntanglingLayers(weights, wires=range(nq2))
            return [qml.expval(qml.PauliZ(MEASURED_QUBIT_2ND_IDX))]

        # There is one input feature: OD. Preserve the original ModuleList route.
        self.qlayer_1 = nn.ModuleList([
            qml.qnn.TorchLayer(first_stage,
                {"weights": (int(params["n_layers"]), nq, 3)})
        ])
        self.qlayer_21 = qml.qnn.TorchLayer(
            second_stage, {"weights": (int(params["n_2nd_layers"]), nq2, 3)}
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for i, layer in enumerate(self.qlayer_1):
            feature = x[:, i:i + 1]
            encoded = torch.cat([torch.ones_like(feature), feature], dim=1)
            outputs.append(layer(encoded).reshape(-1, 1))
        x = torch.cat(outputs, dim=1)
        encoded = torch.cat([torch.ones_like(x[:, :1]), x], dim=1)
        out = self.qlayer_21(encoded).reshape(-1, 1)
        return ((out + 1.0) / 2.0).clamp(1e-6, 1.0 - 1e-6)


class ClassicalGenerator(nn.Module):
    """Original 2-hidden-layer MLP with editable width, slope and latent size."""

    def __init__(self, params: dict):
        super().__init__()
        width = int(params["GENERATOR_HIDDEN_WIDTH"])
        slope = float(params["LEAKY_RELU_SLOPE"])
        self.network = nn.Sequential(
            nn.Linear(int(params["LATENT_DIM"]), width),
            nn.LeakyReLU(slope),
            nn.Linear(width, width),
            nn.LeakyReLU(slope),
            nn.Linear(width, 1),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.network(noise)


def cdf_ks_distance(real: np.ndarray, fake: np.ndarray) -> float:
    """Exact two-sample empirical CDF maximum absolute distance (KS statistic)."""
    r, f = np.sort(real), np.sort(fake)
    positions = np.union1d(r, f)
    real_cdf = np.searchsorted(r, positions, side="right") / len(r)
    fake_cdf = np.searchsorted(f, positions, side="right") / len(f)
    return float(np.max(np.abs(real_cdf - fake_cdf)))


def kl_generated_to_real(real: np.ndarray, fake: np.ndarray,
                         edges: np.ndarray) -> float:
    """D_KL(P_generated || P_real) on FIXED bin edges with pseudocounts."""
    r = np.histogram(real, bins=edges)[0].astype(float) + KL_PSEUDOCOUNT
    f = np.histogram(fake, bins=edges)[0].astype(float) + KL_PSEUDOCOUNT
    r /= r.sum()
    f /= f.sum()
    return float(np.sum(f * np.log(f / r)))


def run_trial(params: dict, od: np.ndarray, edges: np.ndarray):
    """Fresh models, optimizers and data shuffling for each combination."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    od_min, od_max = float(od.min()), float(od.max())
    scaled = 2.0 * (od - od_min) / (od_max - od_min) - 1.0
    real_data = torch.tensor(scaled, dtype=torch.float32).reshape(-1, 1)
    loader = DataLoader(real_data, batch_size=int(params["BATCH_SIZE"]),
                        shuffle=True, generator=torch.Generator().manual_seed(SEED))
    # Match original initialization order: quantum D first, classical G second.
    discriminator = HybridModel(params)
    generator = ClassicalGenerator(params)
    criterion = nn.BCELoss()
    betas = (float(params["ADAM_BETA1"]), float(params["ADAM_BETA2"]))
    optimizer_d = torch.optim.Adam(discriminator.parameters(),
                                   lr=float(params["D_LEARNING_RATE"]), betas=betas)
    optimizer_g = torch.optim.Adam(generator.parameters(),
                                   lr=float(params["G_LEARNING_RATE"]), betas=betas)

    if CHECK_QUANTUM_INPUT_GRADIENT:
        probe = torch.tensor([[-0.5], [0.5]], requires_grad=True)
        grad = torch.autograd.grad(discriminator(probe).sum(), probe,
                                   allow_unused=False)[0]
        if not torch.isfinite(grad).all() or grad.abs().max().item() < 1e-10:
            raise RuntimeError("No usable gradient through the two quantum stages.")

    history = []
    epochs = int(params["EPOCHS"])
    latent_dim = int(params["LATENT_DIM"])
    for epoch in range(1, epochs + 1):
        total_g = total_d = 0.0
        for real_batch in loader:
            batch_n = len(real_batch)
            real_labels = torch.ones(batch_n, 1)
            fake_labels = torch.zeros(batch_n, 1)

            # Original BCE discriminator step; fake is detached here only.
            optimizer_d.zero_grad()
            fake_batch = generator(torch.randn(batch_n, latent_dim))
            d_real = criterion(discriminator(real_batch), real_labels)
            d_fake = criterion(discriminator(fake_batch.detach()), fake_labels)
            d_loss = (d_real + d_fake) / 2.0
            if not torch.isfinite(d_loss):
                raise RuntimeError(f"Non-finite discriminator loss in epoch {epoch}.")
            d_loss.backward()
            optimizer_d.step()

            # Original BCE generator step, with D parameters held fixed.
            for p in discriminator.parameters():
                p.requires_grad_(False)
            try:
                optimizer_g.zero_grad()
                fake_batch = generator(torch.randn(batch_n, latent_dim))
                g_loss = criterion(discriminator(fake_batch), real_labels)
                if not torch.isfinite(g_loss):
                    raise RuntimeError(f"Non-finite generator loss in epoch {epoch}.")
                g_loss.backward()  # Gradient must reach the classical generator.
                optimizer_g.step()
            finally:
                for p in discriminator.parameters():
                    p.requires_grad_(True)

            total_g += float(g_loss.item()) * batch_n
            total_d += float(d_loss.item()) * batch_n
        mean_g, mean_d = total_g / len(real_data), total_d / len(real_data)
        history.append((epoch, mean_g, mean_d))
        if epoch == 1 or epoch % LOG_EVERY_N_EPOCHS == 0 or epoch == epochs:
            print(f"    Epoch {epoch:4d}/{epochs}: G={mean_g:.5f}, D={mean_d:.5f}")

    generator.eval()
    with torch.no_grad():
        generated_scaled = generator(torch.randn(N_SYNTHETIC, latent_dim)).cpu().numpy().ravel()
    fake_od = od_min + (generated_scaled.astype(float) + 1.0) * (od_max - od_min) / 2.0
    metrics = {
        "Generator_loss": float(history[-1][1]),
        "Discriminator_loss": float(history[-1][2]),
        "KL_generated_to_real": kl_generated_to_real(od, fake_od, edges),
        "CDF_Distance_KS": cdf_ks_distance(od, fake_od),
    }
    if not all(math.isfinite(v) for v in metrics.values()):
        raise RuntimeError("An evaluation metric is non-finite.")
    return metrics, fake_od, history, generator, discriminator


def successful(row: dict) -> bool:
    if row.get("Status") != "Completed":
        return False
    try:
        return all(math.isfinite(float(row[k])) for k in METRIC_NAMES)
    except (KeyError, ValueError, TypeError):
        return False


def find_best(rows: list[dict]):
    possible = [row for row in rows if successful(row)]
    if not possible:
        return None
    # Generator/discriminator BCE losses are diagnostic only; NOT rank criteria.
    return min(possible, key=lambda r: (
        float(r["KL_generated_to_real"]),
        float(r["CDF_Distance_KS"]),
        int(r["Run"]),
    ))


def atomic_json(path: Path, content) -> None:
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(content, handle, ensure_ascii=False, indent=2)
    os.replace(temp, path)


def save_csv_checkpoint(rows: list[dict], path: Path) -> None:
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def style_heading(cell) -> None:
    cell.fill = PatternFill("solid", fgColor=NAVY)
    cell.font = Font(name="Aptos", size=11, bold=True, color=WHITE)
    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)


def report_excel(rows: list[dict], planned: list[dict], output: Path, od: np.ndarray) -> None:
    """Write all three requested Excel sheets, atomically after each trial."""
    wb = Workbook()
    ws = wb.active
    ws.title = "All Optimization Results"
    best_ws = wb.create_sheet("Best Parameters")
    summary = wb.create_sheet("Optimization Summary")
    best = find_best(rows)
    best_run = int(best["Run"]) if best else None

    last_col = get_column_letter(len(REPORT_COLUMNS))
    ws.merge_cells(f"A1:{last_col}1")
    ws["A1"] = "CG + QD | All Optimization Results"
    ws["A1"].font = Font(name="Aptos Display", size=15, bold=True, color=NAVY)
    ws["A2"] = "Best trial: lowest binned KL(generated || real); ties: lowest CDF KS distance."
    ws.row_dimensions[1].height = 28
    header_row = 4
    for col, name in enumerate(REPORT_COLUMNS, start=1):
        cell = ws.cell(header_row, col, name)
        style_heading(cell)
        width = 13 if name in {"Run", "Status", "Best_Run", "Seed"} else 22
        if name == "Error":
            width = 58
        ws.column_dimensions[get_column_letter(col)].width = width
    ws.row_dimensions[header_row].height = 35
    for row_number, result in enumerate(rows, start=5):
        is_best = int(result["Run"]) == best_run
        for col, name in enumerate(REPORT_COLUMNS, start=1):
            value = "YES" if (name == "Best_Run" and is_best) else (
                "" if name == "Best_Run" else result.get(name, "")
            )
            if value is None or (isinstance(value, float) and not math.isfinite(value)):
                value = ""
            cell = ws.cell(row_number, col, value)
            cell.alignment = Alignment(vertical="center")
            if is_best:
                cell.fill = PatternFill("solid", fgColor=GREEN)
                cell.font = Font(name="Aptos", bold=True, color=NAVY)
            elif row_number % 2 == 0:
                cell.fill = PatternFill("solid", fgColor=GRAY)
            if name in {"G_LEARNING_RATE", "D_LEARNING_RATE", "KL_generated_to_real", "CDF_Distance_KS"}:
                cell.number_format = "0.000000"
            elif name in {"ADAM_BETA1", "ADAM_BETA2", "LEAKY_RELU_SLOPE"}:
                cell.number_format = "0.000"
            elif name in {"Generator_loss", "Discriminator_loss", "Duration_seconds"}:
                cell.number_format = "0.0000"
        ws.row_dimensions[row_number].height = 19
    ws.auto_filter.ref = f"A4:{last_col}{max(4, len(rows) + 4)}"
    ws.freeze_panes = "E5"

    # Best Parameters: one selected parameter/metric per row, no invented values.
    best_ws.merge_cells("A1:F1")
    best_ws["A1"] = "Best Parameter Combination"
    best_ws["A1"].font = Font(name="Aptos Display", size=15, bold=True, color=NAVY)
    best_ws["A3"] = "Selection criterion"
    best_ws["B3"] = "Minimum KL_generated_to_real (CDF KS tie-breaker)"
    best_ws["A4"] = "Best run"
    best_ws["B4"] = best_run if best_run is not None else "No successful trial yet"
    best_ws["A6"], best_ws["B6"] = "Parameter", "Selected value"
    best_ws["D6"], best_ws["E6"] = "Evaluation metric", "Value"
    for x in ["A6", "B6", "D6", "E6"]:
        style_heading(best_ws[x])
    if best:
        for row_idx, name in enumerate(PARAMETER_NAMES + ["NUM_BINS", "N_SYNTHETIC"], start=7):
            best_ws.cell(row_idx, 1, name)
            best_ws.cell(row_idx, 2, (NUM_BINS if name == "NUM_BINS" else
                                       N_SYNTHETIC if name == "N_SYNTHETIC" else best[name]))
        for row_idx, name in enumerate(METRIC_NAMES + ["Duration_seconds", "Status"], start=7):
            best_ws.cell(row_idx, 4, name)
            best_ws.cell(row_idx, 5, best[name])
            if name in METRIC_NAMES:
                best_ws.cell(row_idx, 5).number_format = "0.000000"
    for col, width in {"A": 28, "B": 47, "C": 4, "D": 30, "E": 20, "F": 12}.items():
        best_ws.column_dimensions[col].width = width
    best_ws.freeze_panes = "A7"

    # Summary includes the full editable search space (including all parameters).
    summary.merge_cells("A1:F1")
    summary["A1"] = "Optimization Summary"
    summary["A1"].font = Font(name="Aptos Display", size=15, bold=True, color=NAVY)
    summary["A3"], summary["B3"] = "Setting / statistic", "Value"
    summary["D3"], summary["E3"] = "Parameter / setting", "Test values or fixed value"
    for x in ["A3", "B3", "D3", "E3"]:
        style_heading(summary[x])
    summary_stats = [
        ("Optimization method", OPTIMIZATION_METHOD + (" + baseline" if OPTIMIZATION_METHOD == "random" else "")),
        ("Planned trials", len(planned)),
        ("Recorded trials", len(rows)),
        ("Successful trials", sum(successful(r) for r in rows)),
        ("Failed trials", sum(r["Status"] == "Failed" for r in rows)),
        ("Best run", best_run if best_run is not None else "Not available yet"),
        ("Best KL divergence", best["KL_generated_to_real"] if best else ""),
        ("Best CDF KS distance", best["CDF_Distance_KS"] if best else ""),
        ("Mean successful KL", (float(np.mean([float(r["KL_generated_to_real"]) for r in rows if successful(r)]))
                                if best else "")),
        ("Ranking rule", "KL ascending; then CDF KS ascending"),
        ("Real OD count", len(od)),
        ("Synthetic OD count per trial", N_SYNTHETIC),
        ("Seed per trial", SEED),
        ("Input file", str(FILE_PATH)),
        ("Excel output", str(output)),
        ("KL smoothing pseudocount", KL_PSEUDOCOUNT),
        ("Note", "Losses track training balance; they do not select the winner."),
        ("Ir note", "No 'Ir' parameter is defined in the supplied CG code."),
        ("NUM_BINS note", "Fixed for all trials to make KL comparisons fair."),
    ]
    for row_idx, (label, value) in enumerate(summary_stats, start=4):
        summary.cell(row_idx, 1, label)
        summary.cell(row_idx, 2, value)
    for row_idx, (name, candidates) in enumerate(
            list(PARAMETER_VALUES.items()) + [("NUM_BINS (fixed)", [NUM_BINS])], start=4):
        summary.cell(row_idx, 4, name)
        summary.cell(row_idx, 5, ", ".join(str(x) for x in candidates))
    for col, width in {"A": 29, "B": 66, "C": 4, "D": 28, "E": 65, "F": 4}.items():
        summary.column_dimensions[col].width = width
    summary.freeze_panes = "A4"

    # Save next to the target first, then replace; interruption cannot corrupt a
    # previously saved good XLSX file. Close Excel if Windows reports file locked.
    temp = output.with_name("." + output.stem + "_writing.xlsx")
    wb.save(temp)
    try:
        os.replace(temp, output)
    except PermissionError as exc:
        raise PermissionError(f"Close {output.name} in Excel before running optimization.") from exc


def save_best_artifacts(best_dir: Path, params: dict, metrics: dict,
                        real_od: np.ndarray, fake_od: np.ndarray,
                        history: list, generator: nn.Module,
                        discriminator: nn.Module, run_number: int,
                        edges: np.ndarray) -> None:
    """Keep the original synthetic CSV, summary, training CSV, model and plots."""
    best_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"OD": fake_od, "Source": "synthetic"}).to_csv(
        best_dir / "synthetic_OD.csv", index=False)
    pd.concat([
        pd.DataFrame({"OD": real_od, "Source": "real"}),
        pd.DataFrame({"OD": fake_od, "Source": "synthetic"}),
    ], ignore_index=True).to_csv(best_dir / "OD_real_and_synthetic.csv", index=False)
    history_df = pd.DataFrame(history, columns=["Epoch", "Generator_loss", "Discriminator_loss"])
    history_df.to_csv(best_dir / "training_history.csv", index=False)
    statistics = pd.DataFrame({
        "Real_OD": pd.Series(real_od).describe(),
        "Synthetic_OD": pd.Series(fake_od).describe(),
    })
    statistics.to_csv(best_dir / "OD_summary.csv", index_label="Statistic")
    model_data = {
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "od_min": float(real_od.min()), "od_max": float(real_od.max()),
        "parameters": params, "metrics": metrics,
        "run": run_number, "seed": SEED,
    }
    temp_model = best_dir / "trained_models.pt.tmp"
    torch.save(model_data, temp_model)
    os.replace(temp_model, best_dir / "trained_models.pt")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history_df["Epoch"], history_df["Generator_loss"], label="Generator")
    axes[0].plot(history_df["Epoch"], history_df["Discriminator_loss"], label="Discriminator")
    axes[0].set(xlabel="Epoch", ylabel="BCE loss", title="Training losses")
    axes[0].legend()
    axes[1].hist(real_od, bins=edges, density=True, alpha=0.55, label="Real")
    axes[1].hist(fake_od, bins=edges, density=True, alpha=0.55, label="Synthetic")
    axes[1].set(xlabel="OD", ylabel="Density", title="OD distribution")
    axes[1].legend()
    for values, label in ((real_od, "Real"), (fake_od, "Synthetic")):
        axes[2].step(np.sort(values), np.arange(1, len(values) + 1) / len(values),
                     where="post", label=label)
    axes[2].set(xlabel="OD", ylabel="Cumulative probability", title="Empirical CDF")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(best_dir / "OD_comparison.png", dpi=180)
    plt.close(fig)


def configuration_signature(real_od: np.ndarray) -> str:
    fields = {
        "code_version": CODE_VERSION, "parameter_values": PARAMETER_VALUES,
        "baseline": BASELINE_PARAMETERS, "method": OPTIMIZATION_METHOD,
        "num_trials": NUM_TRIALS, "seed": SEED, "num_bins": NUM_BINS,
        "n_synthetic": N_SYNTHETIC, "kl_pseudocount": KL_PSEUDOCOUNT,
        "gradient_check": CHECK_QUANTUM_INPUT_GRADIENT,
        "amplitude_padding": AMPLITUDE_PADDING,
        "measured_wire_1": MEASURED_QUBIT_IDX,
        "measured_wire_2": MEASURED_QUBIT_2ND_IDX,
        "od_sha256": hashlib.sha256(real_od.astype("<f8").tobytes()).hexdigest(),
    }
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


def main() -> None:
    validate_settings()
    torch.set_num_threads(TORCH_CPU_THREADS)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    real_od = load_od(FILE_PATH)
    histogram_edges = np.linspace(float(real_od.min()), float(real_od.max()), NUM_BINS + 1)
    xlsx = OUTPUT_DIR / EXCEL_FILENAME
    csv_path = OUTPUT_DIR / "optimization_checkpoint.csv"
    plan_path = OUTPUT_DIR / "optimization_plan.json"
    best_dir = OUTPUT_DIR / "Best_Run_Artifacts"
    fingerprint = configuration_signature(real_od)

    if RESUME_PREVIOUS_RUN and plan_path.exists():
        with plan_path.open("r", encoding="utf-8") as handle:
            previous_plan = json.load(handle)
        if previous_plan.get("fingerprint") != fingerprint:
            raise ValueError(
                "Existing optimization checkpoint uses different data/settings. "
                "Set RESUME_PREVIOUS_RUN=False to start a new search (overwriting "
                "old checkpoint/report), or move the old output folder first."
            )
        trials = previous_plan["trials"]
        rows = pd.read_csv(csv_path, keep_default_na=False).to_dict("records") if csv_path.exists() else []
        print(f"Resuming existing plan: {len(rows)} saved results out of {len(trials)} trials.")
    else:
        trials = planned_trials()
        rows = []
        atomic_json(plan_path, {"fingerprint": fingerprint, "trials": trials,
                                "created_at": datetime.now().isoformat(timespec="seconds")})
        save_csv_checkpoint(rows, csv_path)
    recorded = {int(row["Run"]) for row in rows}
    print(f"Method: {OPTIMIZATION_METHOD} | Target: {len(trials)} trials | "
          f"Evaluation: fixed {NUM_BINS} bins, {N_SYNTHETIC} generated samples.")
    report_excel(rows, trials, xlsx, real_od)
    overall_start = time.perf_counter()

    for run_number, params in enumerate(trials, start=1):
        if run_number in recorded:
            continue
        print(f"\n{'=' * 65}\nRun {run_number}/{len(trials)}: {params}\n{'=' * 65}")
        start = time.perf_counter()
        row = {"Run": run_number, "Status": "Failed", "Seed": SEED,
               **params, "NUM_BINS": NUM_BINS, "N_SYNTHETIC": N_SYNTHETIC,
               **{name: "" for name in METRIC_NAMES}, "Error": ""}
        artifacts = None
        try:
            metrics, fake_od, history, gen, disc = run_trial(params, real_od, histogram_edges)
            row.update(metrics)
            row["Status"] = "Completed"
            artifacts = (fake_od, history, gen, disc)
            print(f"  KL={metrics['KL_generated_to_real']:.6f}, "
                  f"CDF KS={metrics['CDF_Distance_KS']:.6f}")
        except (Exception, KeyboardInterrupt) as exc:
            row["Error"] = f"{type(exc).__name__}: {exc}"[:800]
            print(f"  TRIAL FAILED: {row['Error']}")
            if not isinstance(exc, KeyboardInterrupt):
                traceback.print_exc(limit=3)
        row["Duration_seconds"] = round(time.perf_counter() - start, 3)
        rows.append(row)
        rows.sort(key=lambda x: int(x["Run"]))
        best = find_best(rows)
        try:
            # Save first: records cannot be lost when a later trial fails.
            save_csv_checkpoint(rows, csv_path)
            report_excel(rows, trials, xlsx, real_od)
            # Save original-model artifacts every time a new best is found.
            if (artifacts is not None and best is not None
                    and int(best["Run"]) == run_number):
                save_best_artifacts(best_dir, params, metrics, real_od,
                                    *artifacts, run_number, histogram_edges)
                print(f"  New best run: {run_number}; saved model, CSVs and plot.")
        except Exception as exc:
            raise RuntimeError(
                f"Could not save checkpoint/report: {exc}. "
                f"Check write permissions and close {EXCEL_FILENAME} in Excel."
            ) from exc
        if row["Status"] == "Failed" and row["Error"].startswith("KeyboardInterrupt"):
            print("Interrupted. Completed results are in the Excel/CSV checkpoints.")
            break

    best = find_best(rows)
    print("\n" + "=" * 65)
    print(f"Recorded {len(rows)}/{len(trials)} runs in {time.perf_counter() - overall_start:.1f} seconds.")
    print(f"Excel file: {xlsx}")
    print(f"Backup CSV: {csv_path}")
    if best is None:
        print("NO SUCCESSFUL TRIAL. See the Error column of the Excel file.")
    else:
        print(f"BEST RUN: {int(best['Run'])} | "
              f"KL={float(best['KL_generated_to_real']):.6f} | "
              f"CDF KS={float(best['CDF_Distance_KS']):.6f}")
        print("Best parameter values:")
        for name in PARAMETER_NAMES:
            print(f"  {name}: {best[name]}")
        print(f"  NUM_BINS (fixed): {NUM_BINS}")
        print(f"Best-run outputs: {best_dir}")
        print("OD values are modeled individually; temporal order/covariates are not modeled.")
        if SHOW_BEST_PLOT and (best_dir / "OD_comparison.png").is_file():
            picture = plt.imread(best_dir / "OD_comparison.png")
            fig = plt.figure(figsize=(15, 5))
            plt.imshow(picture)
            plt.axis("off")
            plt.tight_layout()
            plt.show()
            plt.close(fig)


if __name__ == "__main__":
    main()
