"""Train the original Hybrid QGAN on OD and export synthetic OD samples.

CHANGED / ADDED comments identify adaptations from the uploaded tutorial.
The model learns individual OD values, not their time order or other columns.
"""

from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from scipy.stats import entropy
from qiskit import QuantumCircuit
from qiskit.circuit.library import efficient_su2
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals


# CHANGED 1: Settings for a normal Python file in VS Code.
SCRIPT_DIR = Path(__file__).resolve().parent
FILE_PATH = Path(r"C:\Users\zrast\Desktop\Python Code\MOHAMMAD\GENERAL\SC1\EXCEL\data1.csv")
SHEET_NAME = 0
OD_COLUMN = "OD"
OUTPUT_DIR = SCRIPT_DIR / "qgan_od_output"

NUM_BINS = 32  # 32 OD intervals = 5 qubits. Must be a power of 2.
N_EPOCHS = 200  # Original: 50. Adjustable starting value, not a convergence guarantee.
N_SYNTHETIC = 1000  # Number of ADDITIONAL OD values to generate.
SEED = 123456
SHOTS = 4096  # Quantum measurements used to estimate circuit probabilities.
SAMPLE_WITHIN_BINS = True  # False gives only the 32 bin-center values.
SHOW_PLOTS = True  # Plots are saved even if this is False.


# CHANGED 2: Read real OD values instead of creating a normal distribution.
def load_od_data(file_path, sheet_name=0, column="OD"):
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)
    elif file_path.suffix.lower() == ".xlsx":
        df = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")
    else:
        raise ValueError("Use a .csv or .xlsx file.")

    df.columns = [str(name).strip() for name in df.columns]
    if column not in df.columns:
        raise ValueError(f"Column {column!r} not found. Available: {list(df.columns)}")

    od = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(od)
    print(f"OD rows: {len(od)}; invalid/missing rows excluded: {(~keep).sum()}")
    od = od[keep]
    if len(od) < 2 or od.max() == od.min():
        raise ValueError("At least two finite, different OD values are required.")
    if np.any(od < 0):
        raise ValueError("Negative OD values found. Check these measurements first.")
    return od  # Repeated measurements and their frequencies are preserved.


# CHANGED 3: Convert the observed OD histogram to the QGAN target distribution.
def prepare_od_distribution(od, num_bins):
    if num_bins < 2 or num_bins & (num_bins - 1):
        raise ValueError("NUM_BINS must be a power of 2, such as 16, 32 or 64.")
    counts, bin_edges = np.histogram(od, bins=num_bins)
    prob_data = counts / counts.sum()
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # The discriminator sees one scaled OD coordinate per bin, in [-2, 2].
    scaled_centers = 4 * (bin_centers - bin_edges[0]) / np.ptp(bin_edges) - 2
    grid_elements = scaled_centers.reshape(-1, 1)
    return prob_data, bin_edges, bin_centers, grid_elements


# ORIGINAL model: the same 20-unit classical discriminator.
class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.linear_input = nn.Linear(input_size, 20)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.linear20 = nn.Linear(20, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.linear_input(input)
        x = self.leaky_relu(x)
        x = self.linear20(x)
        return self.sigmoid(x)


# ORIGINAL generator structure. Circuit and sampler are explicit arguments.
def create_generator(qc, sampler) -> TorchConnector:
    qnn = SamplerQNN(
        circuit=qc,
        sampler=sampler,
        input_params=[],
        weight_params=qc.parameters,
        sparse=False,
    )
    initial_weights = algorithm_globals.random.random(qc.num_parameters)
    return TorchConnector(qnn, initial_weights)


def adversarial_loss(input, target, w):
    # CHANGED 4: Avoid log(0); keep the original weighted binary cross-entropy.
    input = torch.clamp(input, min=1e-7, max=1 - 1e-7)
    bce_loss = target * torch.log(input) + (1 - target) * torch.log(1 - input)
    return -torch.sum(w * bce_loss)


# ADDED 1: Sample from the TRAINED generator's probabilities.
def sample_synthetic_od(probabilities, bin_edges, n_samples, rng, within_bins):
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    if n_samples < 1:
        raise ValueError("N_SYNTHETIC must be positive.")
    if (len(probabilities) != len(bin_edges) - 1
            or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0) or probabilities.sum() <= 0):
        raise ValueError("The generator returned invalid bin probabilities.")
    probabilities = probabilities / probabilities.sum()
    chosen_bins = rng.choice(len(probabilities), size=n_samples, p=probabilities)
    if within_bins:
        # Additional classical assumption: uniform values inside each chosen bin.
        # The quantum model learns bin probabilities, not within-bin variation.
        return rng.uniform(bin_edges[chosen_bins], bin_edges[chosen_bins + 1])
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers[chosen_bins]


# ADDED 2: Keep real and synthetic values clearly labeled when saving.
def save_od_samples(od, synthetic_od, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    synthetic = pd.DataFrame({"OD": synthetic_od, "Source": "synthetic"})
    real = pd.DataFrame({"OD": od, "Source": "real"})
    synthetic.to_csv(output_dir / "synthetic_OD.csv", index=False)
    pd.concat([real, synthetic], ignore_index=True).to_csv(
        output_dir / "OD_real_and_synthetic.csv", index=False
    )


def main():
    algorithm_globals.random_seed = SEED
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    if N_EPOCHS < 1 or N_SYNTHETIC < 1:
        raise ValueError("N_EPOCHS and N_SYNTHETIC must be positive.")

    od = load_od_data(FILE_PATH, SHEET_NAME, OD_COLUMN)
    prob_data, bin_edges, bin_centers, grid_elements = prepare_od_distribution(od, NUM_BINS)
    num_dim = 1  # CHANGED: only OD, instead of the original two features.
    num_qubits = int(np.log2(NUM_BINS))
    print(f"Using {len(od)} real OD values, from {od.min():.4f} to {od.max():.4f}.")
    print(f"Quantum generator: {num_qubits} qubits and {NUM_BINS} OD bins.")

    # ORIGINAL: Hadamard initialization and efficient_su2 with 6 repetitions.
    qc = QuantumCircuit(num_qubits)
    qc.h(qc.qubits)
    ansatz = efficient_su2(num_qubits, reps=6)
    qc.compose(ansatz, inplace=True)
    # CHANGED: explicit shot count and seed for the same statevector sampler.
    sampler = Sampler(default_shots=SHOTS, seed=SEED)
    generator = create_generator(qc, sampler)
    discriminator = Discriminator(num_dim)
    print(f"Trainable quantum parameters: {qc.num_parameters}")

    # ORIGINAL: Adam settings and alternating standard GAN training.
    lr, b1, b2 = 0.01, 0.7, 0.999
    generator_optimizer = Adam(generator.parameters(), lr=lr, betas=(b1, b2), weight_decay=0.005)
    discriminator_optimizer = Adam(discriminator.parameters(), lr=lr, betas=(b1, b2), weight_decay=0.005)

    # CHANGED: constant tensors are created once rather than every epoch.
    real_dist = torch.tensor(prob_data, dtype=torch.float).reshape(-1, 1)
    samples = torch.tensor(grid_elements, dtype=torch.float)
    valid = torch.ones(NUM_BINS, 1, dtype=torch.float)
    fake = torch.zeros(NUM_BINS, 1, dtype=torch.float)
    history = []
    start = time.time()

    for epoch in range(N_EPOCHS):
        disc_value = discriminator(samples)
        gen_dist = generator(torch.tensor([])).reshape(-1, 1)

        generator_optimizer.zero_grad()
        generator_loss = adversarial_loss(disc_value, valid, gen_dist)
        generator_loss.backward(retain_graph=True)
        generator_optimizer.step()

        discriminator_optimizer.zero_grad()
        real_loss = adversarial_loss(disc_value, valid, real_dist)
        fake_loss = adversarial_loss(disc_value, fake, gen_dist.detach())
        discriminator_loss = (real_loss + fake_loss) / 2
        discriminator_loss.backward()
        discriminator_optimizer.step()

        # CHANGED: tiny smoothing prevents infinite KL if a bin is empty.
        # It affects this diagnostic only, not the training loss or real data.
        p = gen_dist.detach().cpu().numpy().ravel()
        kl = float(entropy(p + 1e-12, prob_data + 1e-12))
        g_loss, d_loss = generator_loss.item(), discriminator_loss.item()
        if not np.isfinite([g_loss, d_loss, kl]).all():
            raise RuntimeError("Training became non-finite. Reduce the learning rate.")
        history.append([epoch + 1, g_loss, d_loss, kl])
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == N_EPOCHS:
            print(f"Epoch {epoch + 1}/{N_EPOCHS}: G={g_loss:.4f}, D={d_loss:.4f}, KL={kl:.4f}")

    print(f"Training completed in {time.time() - start:.1f} seconds.")

    # ADDED 3: Re-evaluate AFTER the final update, then generate new OD samples.
    generator.eval()
    with torch.no_grad():
        generated_probabilities = generator().cpu().numpy().ravel().astype(float)
    synthetic_od = sample_synthetic_od(
        generated_probabilities, bin_edges, N_SYNTHETIC, rng, SAMPLE_WITHIN_BINS
    )
    generated_probabilities /= generated_probabilities.sum()
    save_od_samples(od, synthetic_od, OUTPUT_DIR)

    training = pd.DataFrame(history, columns=["Epoch", "Generator_loss", "Discriminator_loss", "KL_generated_to_real"])
    training.to_csv(OUTPUT_DIR / "training_history.csv", index=False)
    pd.DataFrame({
        "OD_bin_left": bin_edges[:-1], "OD_bin_right": bin_edges[1:],
        "OD_bin_center": bin_centers, "Real_probability": prob_data,
        "Generated_probability": generated_probabilities,
    }).to_csv(OUTPUT_DIR / "learned_OD_distribution.csv", index=False)

    # CHANGED 5: One-dimensional plots, saved before showing them in VS Code.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(training["Epoch"], training["Generator_loss"], label="Generator")
    axes[0, 0].plot(training["Epoch"], training["Discriminator_loss"], label="Discriminator")
    axes[0, 0].set(xlabel="Epoch", ylabel="Loss", title="Training losses")
    axes[0, 0].legend()
    axes[0, 1].plot(training["Epoch"], training["KL_generated_to_real"])
    axes[0, 1].set(xlabel="Epoch", ylabel="Smoothed KL", title="Fit to training distribution")
    axes[1, 0].hist(od, bins=bin_edges, density=True, alpha=0.55, label="Real OD")
    axes[1, 0].hist(synthetic_od, bins=bin_edges, density=True, alpha=0.55, label="Synthetic OD")
    axes[1, 0].set(xlabel="OD", ylabel="Density", title="Real and synthetic OD")
    axes[1, 0].legend()
    for values, label in [(od, "Real OD"), (synthetic_od, "Synthetic OD")]:
        axes[1, 1].step(np.sort(values), np.arange(1, len(values) + 1) / len(values), where="post", label=label)
    axes[1, 1].set(xlabel="OD", ylabel="Cumulative probability", title="Empirical CDF")
    axes[1, 1].legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "OD_comparison.png", dpi=180)

    print(f"Saved {N_SYNTHETIC} synthetic OD values to: {OUTPUT_DIR.resolve()}")
    print(f"Combined table: {len(od)} real + {N_SYNTHETIC} synthetic = {len(od) + N_SYNTHETIC} rows.")
    print("Synthetic rows are independent OD samples; they have no generated dates or paired process variables.")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# ADDED: Standard entry point for VS Code / terminal execution.
if __name__ == "__main__":
    main()
