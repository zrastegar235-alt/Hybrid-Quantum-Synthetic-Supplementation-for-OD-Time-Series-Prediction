"""OD synthetic data: Classical Generator + Quantum Discriminator.

Adapted from the supplied PennyLane HybridModel (default two-stage branch).
CHANGED / ADDED comments identify the adaptations. Standard BCE GAN only.
"""

from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml
import torch
from torch import nn
from torch.utils.data import DataLoader


# ADDED 1: Your input path and simple settings for VS Code.
SCRIPT_DIR = Path(__file__).resolve().parent
FILE_PATH = Path(r"C:\Users\zrast\Desktop\Python Code\MOHAMMAD\GENERAL\SC1\EXCEL\data1.csv")
# To use the CSV included in the ZIP instead: FILE_PATH = SCRIPT_DIR / "data1.csv"
SHEET_NAME = 0  # For .xlsx only: 0 = first sheet, or use the sheet's name.
OD_COLUMN = "OD"
OUTPUT_DIR = SCRIPT_DIR / "od_cg_qd_output"
N_SYNTHETIC = 1000
EPOCHS = 200
BATCH_SIZE = 64
LATENT_DIM = 8
G_LEARNING_RATE = 0.001
D_LEARNING_RATE = 0.01
SEED = 123456
SHOW_PLOTS = True


# CHANGED 1: One OD feature replaces the 45 molecular atom branches.
# Two qubits per stage keep simulation small and permit entangling gates.
N_FEATURES = 1
n_qubits = 2                 # Original: 5
n_2nd_qubits = 2             # Original: 9
MEASURED_QUBIT_IDX = 0        # Must be a valid wire in the smaller circuit.
MEASURED_QUBIT_2ND_IDX = 1
n_layers = 1                 # Same as the original first stage.
n_2nd_layers = 3             # Same as the original second stage.

# ORIGINAL: Local simulator, Torch interface and backpropagation.
dev = qml.device("default.qubit", wires=n_qubits, shots=None)
dev1 = qml.device("default.qubit", wires=n_2nd_qubits, shots=None)


@qml.qnode(dev, interface="torch", diff_method="backprop")
def qnode(inputs, weights):
    qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), pad_with=0.001, normalize=True)
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(MEASURED_QUBIT_IDX))]


@qml.qnode(dev1, interface="torch", diff_method="backprop")
def qnode_(inputs, weights):
    qml.AmplitudeEmbedding(inputs, wires=range(n_2nd_qubits), pad_with=0.001, normalize=True)
    qml.StronglyEntanglingLayers(weights, wires=range(n_2nd_qubits))
    return [qml.expval(qml.PauliZ(MEASURED_QUBIT_2ND_IDX))]


weight_shapes = {"weights": (n_layers, n_qubits, 3)}
weight_shapes_2nd = {"weights": (n_2nd_layers, n_2nd_qubits, 3)}


class HybridModel(nn.Module):
    """The supplied model, adapted into a binary quantum discriminator."""

    def __init__(self):
        super().__init__()
        self.qlayer_1 = nn.ModuleList([
            qml.qnn.TorchLayer(qnode, weight_shapes) for _ in range(N_FEATURES)
        ])
        self.qlayer_21 = qml.qnn.TorchLayer(qnode_, weight_shapes_2nd)
        # CHANGED 2: Keep the original default two-stage route (LAYER3=False).
        # The optional molecular third-stage route is omitted for scalar OD.

    def forward(self, x):
        # CHANGED 3: x has shape [batch, 1]. Add a fixed reference amplitude.
        # Encoding [1, OD] preserves magnitude when the state is normalized.
        outputs = []
        for i, layer in enumerate(self.qlayer_1):
            feature = x[:, i:i + 1]
            encoded = torch.cat([torch.ones_like(feature), feature], dim=1)
            outputs.append(layer(encoded).reshape(-1, 1))
        x = torch.cat(outputs, dim=1)
        encoded = torch.cat([torch.ones_like(x[:, :1]), x], dim=1)
        out = self.qlayer_21(encoded).reshape(-1, 1)

        # CHANGED 4: A one-output Softmax always returns 1.
        # Convert the quantum expectation in [-1, 1] to P(real) in [0, 1].
        return ((out + 1.0) / 2.0).clamp(1e-6, 1.0 - 1e-6)


# ADDED 2: The supplied code had no generator. This one is fully classical.
class ClassicalGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(LATENT_DIM, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
            nn.Tanh(),  # Generated OD is scaled to [-1, 1].
        )

    def forward(self, noise):
        return self.network(noise)


# ADDED 3: Read and clean the real OD column from CSV or Excel.
def load_od(file_path, sheet_name=SHEET_NAME):
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found. Check FILE_PATH: {file_path}")
    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)
    elif file_path.suffix.lower() == ".xlsx":
        df = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")
    else:
        raise ValueError("Use a .csv or .xlsx file.")
    df.columns = [str(column).strip() for column in df.columns]
    if OD_COLUMN not in df.columns:
        raise ValueError(f"Missing {OD_COLUMN!r} column. Found: {list(df.columns)}")
    od = pd.to_numeric(df[OD_COLUMN], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(od)
    print(f"OD rows: {len(od)}; invalid/missing rows excluded: {(~valid).sum()}")
    od = od[valid]
    if len(od) < 2 or od.min() == od.max():
        raise ValueError("At least two finite, different OD values are required.")
    return od


def main(file_path=FILE_PATH, output_dir=OUTPUT_DIR, epochs=EPOCHS,
         n_synthetic=N_SYNTHETIC, show_plots=SHOW_PLOTS):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(1)  # Small simulator operations run on the CPU.
    if epochs < 1 or n_synthetic < 1:
        raise ValueError("EPOCHS and N_SYNTHETIC must be positive.")

    od = load_od(file_path)
    od_min, od_max = float(od.min()), float(od.max())
    od_scaled = 2.0 * (od - od_min) / (od_max - od_min) - 1.0
    real_data = torch.tensor(od_scaled, dtype=torch.float32).reshape(-1, 1)
    loader = DataLoader(real_data, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Training on {len(od)} real OD measurements: {od_min:.4f} to {od_max:.4f}.")

    discriminator = HybridModel()
    generator = ClassicalGenerator()
    criterion = nn.BCELoss()  # Standard GAN loss; no Wasserstein loss.
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=D_LEARNING_RATE, betas=(0.5, 0.999))
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=G_LEARNING_RATE, betas=(0.5, 0.999))

    # ADDED 4: Check that gradients can pass through BOTH quantum stages.
    # This matters because the generator learns through the discriminator.
    probe = torch.tensor([[-0.5], [0.5]], requires_grad=True)
    gradient = torch.autograd.grad(discriminator(probe).sum(), probe)[0]
    if not torch.isfinite(gradient).all() or gradient.abs().max().item() < 1e-10:
        raise RuntimeError("No usable input gradient through the quantum discriminator.")
    print("Gradient through the quantum discriminator: OK")

    # ADDED 5: Alternating discriminator and generator training.
    history = []
    start = time.time()
    for epoch in range(epochs):
        g_total, d_total = 0.0, 0.0
        for real_batch in loader:
            batch_size = real_batch.shape[0]
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            # Update discriminator: real OD = 1, generated OD = 0.
            optimizer_d.zero_grad()
            noise = torch.randn(batch_size, LATENT_DIM)
            fake_batch = generator(noise)
            real_loss = criterion(discriminator(real_batch), real_labels)
            fake_loss = criterion(discriminator(fake_batch.detach()), fake_labels)
            d_loss = (real_loss + fake_loss) / 2.0
            d_loss.backward()
            optimizer_d.step()

            # Update generator while keeping discriminator weights fixed.
            for parameter in discriminator.parameters():
                parameter.requires_grad_(False)
            optimizer_g.zero_grad()
            noise = torch.randn(batch_size, LATENT_DIM)
            fake_batch = generator(noise)
            g_loss = criterion(discriminator(fake_batch), real_labels)
            g_loss.backward()  # Do not detach fake_batch in the generator step.
            optimizer_g.step()
            for parameter in discriminator.parameters():
                parameter.requires_grad_(True)

            if not torch.isfinite(g_loss) or not torch.isfinite(d_loss):
                raise RuntimeError("Training became non-finite. Check learning rates.")
            g_total += g_loss.item() * batch_size
            d_total += d_loss.item() * batch_size

        history.append([epoch + 1, g_total / len(od), d_total / len(od)])
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(f"Epoch {epoch + 1}/{epochs}: G loss={history[-1][1]:.4f}, D loss={history[-1][2]:.4f}")

    # ADDED 6: Generate new values and restore the original OD scale.
    generator.eval()
    with torch.no_grad():
        scaled_samples = generator(torch.randn(n_synthetic, LATENT_DIM)).numpy().ravel()
    synthetic_od = od_min + (scaled_samples.astype(float) + 1.0) * (od_max - od_min) / 2.0
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    real_table = pd.DataFrame({"OD": od, "Source": "real"})
    synthetic_table = pd.DataFrame({"OD": synthetic_od, "Source": "synthetic"})
    synthetic_table.to_csv(output_dir / "synthetic_OD.csv", index=False)
    pd.concat([real_table, synthetic_table], ignore_index=True).to_csv(
        output_dir / "OD_real_and_synthetic.csv", index=False
    )
    training = pd.DataFrame(history, columns=["Epoch", "Generator_loss", "Discriminator_loss"])
    training.to_csv(output_dir / "training_history.csv", index=False)
    statistics = pd.DataFrame({"Real_OD": pd.Series(od).describe(),
                               "Synthetic_OD": pd.Series(synthetic_od).describe()})
    statistics.to_csv(output_dir / "OD_summary.csv", index_label="Statistic")
    torch.save({"generator": generator.state_dict(), "discriminator": discriminator.state_dict(),
                "od_min": od_min, "od_max": od_max, "latent_dim": LATENT_DIM,
                "epochs": epochs, "seed": SEED}, output_dir / "trained_models.pt")

    # ADDED 7: Save plots, then optionally display them in VS Code.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(training["Epoch"], training["Generator_loss"], label="Generator")
    axes[0].plot(training["Epoch"], training["Discriminator_loss"], label="Discriminator")
    axes[0].set(xlabel="Epoch", ylabel="BCE loss", title="Training losses")
    axes[0].legend()
    bins = np.linspace(od_min, od_max, 25)
    axes[1].hist(od, bins=bins, density=True, alpha=0.55, label="Real")
    axes[1].hist(synthetic_od, bins=bins, density=True, alpha=0.55, label="Synthetic")
    axes[1].set(xlabel="OD", ylabel="Density", title="OD distribution")
    axes[1].legend()
    for values, label in [(od, "Real"), (synthetic_od, "Synthetic")]:
        axes[2].step(np.sort(values), np.arange(1, len(values) + 1) / len(values), where="post", label=label)
    axes[2].set(xlabel="OD", ylabel="Cumulative probability", title="Empirical CDF")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "OD_comparison.png", dpi=180)
    print(statistics.round(4).to_string())
    print(f"Finished in {time.time() - start:.1f} seconds. Saved {n_synthetic} synthetic OD values.")
    print(f"Output folder: {output_dir.resolve()}")
    print("These are individual OD samples. Time order and other process variables are not modeled.")
    if show_plots:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
