"""Compare original-style OD LSTMs on MAIN, CG and QG.

Install: pip install numpy pandas openpyxl matplotlib scikit-learn tensorflow
Set BASE_DIR below. All outputs are written to OUTPUT_DIR.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

BASE_DIR = Path(r"C:\Users\zrast\Desktop\Python Code\MOHAMMAD\GENERAL\SC1\EXCEL")
EXCEL_FILE = BASE_DIR / "OD_GENERATED.xlsx"
OUTPUT_DIR = BASE_DIR / "OD_LSTM_comparison"
SHEETS = ("MAIN", "CG", "QG")  # The provided workbook has CG, not CQ.
LOOK_BACK = 3
N_SPLITS = 5
EPOCHS = 100
BATCH_SIZE = 16
LSTM_UNITS = 10
LEARNING_RATE = 0.001
TRAIN_FRACTION = 0.8
START_TIME = pd.Timestamp("2024-03-01")
END_TIME = pd.Timestamp("2024-04-20")


def load_data():
    if not EXCEL_FILE.is_file():
        raise FileNotFoundError(f"Workbook not found: {EXCEL_FILE}")
    book = pd.ExcelFile(EXCEL_FILE)
    missing = set(SHEETS) - set(book.sheet_names)
    if missing:
        raise ValueError(f"Missing sheets {sorted(missing)}; available: {book.sheet_names}")

    main = pd.read_excel(book, sheet_name="MAIN")
    if not {"DATE", "OD"}.issubset(main.columns):
        raise ValueError("MAIN needs DATE and OD columns.")
    dates = pd.to_datetime(main["DATE"], format="%d/%m/%Y %H.%M", errors="coerce")
    if dates.isna().any():
        dates.loc[dates.isna()] = pd.to_datetime(
            main.loc[dates.isna(), "DATE"], dayfirst=True, errors="coerce")
    main["DATE"] = dates
    main["OD"] = pd.to_numeric(main["OD"], errors="coerce")
    main = main.loc[main.DATE.between(START_TIME, END_TIME)].reset_index(drop=True)
    if main.DATE.isna().any() or main.OD.isna().any() or len(main) != 778:
        raise ValueError("MAIN must contain 778 valid DATE/OD rows in the selected period.")
    if not main.DATE.is_monotonic_increasing or main.DATE.duplicated().any():
        raise ValueError("MAIN dates must be unique and in chronological order.")
    main["Real_OD_Number"] = np.arange(1, len(main) + 1)
    main["Source"] = "real"
    datasets = {"MAIN": main}

    for sheet in ("CG", "QG"):
        df = pd.read_excel(book, sheet_name=sheet)
        required = {"OD", "Source", "Real_OD_Number", "Right_Real_Number"}
        if not required.issubset(df.columns):
            raise ValueError(f"{sheet} needs columns {sorted(required)}.")
        df["OD"] = pd.to_numeric(df["OD"], errors="coerce")
        if len(df) != 7780 or df.OD.isna().any():
            raise ValueError(f"{sheet} must contain 7,780 valid OD rows.")
        real = df.loc[df.Source.astype(str).str.lower().eq("real")]
        if (len(real) != len(main)
                or not np.array_equal(real.Real_OD_Number.to_numpy(), main.Real_OD_Number.to_numpy())
                or not np.allclose(real.OD.to_numpy(), main.OD.to_numpy(), atol=1e-10)):
            raise ValueError(f"{sheet} real OD anchors do not match MAIN in order and value.")
        if not pd.to_numeric(df.Right_Real_Number, errors="coerce").between(1, len(main)).all():
            raise ValueError(f"{sheet} has invalid Right_Real_Number values.")
        datasets[sheet] = df.reset_index(drop=True)
    return datasets


def make_windows(values):
    X = np.array([values[i:i + LOOK_BACK] for i in range(len(values) - LOOK_BACK)])
    y = values[LOOK_BACK:]
    return X.reshape(-1, LOOK_BACK, 1), y


def model_factory():
    model = Sequential([LSTM(LSTM_UNITS, input_shape=(LOOK_BACK, 1)), Dense(1)])
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="mean_squared_error")
    return model


def metrics(actual, prediction):
    mse = mean_squared_error(actual, prediction)
    return {"RMSE": np.sqrt(mse), "MAE": mean_absolute_error(actual, prediction),
            "R2": r2_score(actual, prediction), "MSE": mse}


def fit_and_predict(series, train_end, sheet):
    """train_end is exclusive; targets at train_end and later are held out."""
    values = series.OD.to_numpy(dtype=float).reshape(-1, 1)
    X_raw, y_raw = make_windows(values.ravel())
    # Target for window i is row i + LOOK_BACK.
    train_count = train_end - LOOK_BACK
    if train_count <= N_SPLITS + 1 or len(values) <= train_end:
        raise ValueError(f"Insufficient {sheet} training or test rows.")
    cv_rows = []
    for fold, (tr, val) in enumerate(TimeSeriesSplit(N_SPLITS).split(X_raw[:train_count]), 1):
        # Fit each fold's scaler on values available through its last training target.
        fold_scaler = MinMaxScaler().fit(values[:tr[-1] + LOOK_BACK + 1])
        X_tr = fold_scaler.transform(X_raw[tr].reshape(-1, 1)).reshape(-1, LOOK_BACK, 1)
        X_val = fold_scaler.transform(X_raw[val].reshape(-1, 1)).reshape(-1, LOOK_BACK, 1)
        y_tr = fold_scaler.transform(y_raw[tr].reshape(-1, 1)).ravel()
        fold_model = model_factory()
        fold_model.fit(X_tr, y_tr, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        pred = fold_scaler.inverse_transform(fold_model.predict(X_val, verbose=0)).ravel()
        cv_rows.append({"Sheet": sheet, "Fold": fold, **metrics(y_raw[val], pred)})
        print(f"{sheet} fold {fold}: RMSE={cv_rows[-1]['RMSE']:.4f}")

    scaler = MinMaxScaler().fit(values[:train_end])
    X = scaler.transform(X_raw.reshape(-1, 1)).reshape(-1, LOOK_BACK, 1)
    y_train = scaler.transform(y_raw[:train_count].reshape(-1, 1)).ravel()
    model = model_factory()  # Fresh model: test rows never enter training.
    model.fit(X[:train_count], y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    predictions = scaler.inverse_transform(model.predict(X[train_count:], verbose=0)).ravel()
    out = series.iloc[train_end:].copy().reset_index(drop=True)
    out["Prediction"] = predictions
    out["Actual"] = y_raw[train_count:]
    out["Sheet"] = sheet
    return out, cv_rows


def plot_common(common):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(common.DATE, common.Actual, color="black", linewidth=2.2,
            label="Actual OD (real)")
    colors = {"MAIN": "#1565c0", "CG": "#ef6c00", "QG": "#2e7d32"}
    for sheet in SHEETS:
        ax.plot(common.DATE, common[f"{sheet}_Prediction"],
                label=f"{sheet} predicted OD", color=colors[sheet],
                linewidth=1.6, alpha=0.9)
    ax.set(title="Held-out real OD: actual vs MAIN, CG and QG predictions",
           xlabel="Date", ylabel="OD")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "real_OD_aligned_predictions.png", dpi=180)


def plot_each(results):
    fig, axes = plt.subplots(3, 1, figsize=(14, 11))
    for ax, sheet in zip(axes, SHEETS):
        df = results[sheet]
        ax.plot(df.index, df.Actual, label="Actual OD", linewidth=1)
        ax.plot(df.index, df.Prediction, label="Predicted OD", linewidth=1, alpha=0.8)
        ax.set(title=f"{sheet}: all held-out native rows", xlabel="Test row", ylabel="OD")
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "native_step_predictions.png", dpi=180)


def plot_metrics(table):
    fig = plt.figure(figsize=(12, 10))
    grid = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.55], hspace=0.5)
    axes = [fig.add_subplot(grid[i, j]) for i in range(2) for j in range(2)]
    for ax, metric in zip(axes, ("RMSE", "MAE", "R2", "MSE")):
        bars = ax.bar(table.Sheet, table[metric],
                      color=["#3377aa", "#ee9944", "#55aa77"])
        ax.bar_label(bars, labels=[f"{v:.4f}" for v in table[metric]],
                     padding=3, fontsize=9)
        ax.margins(y=0.18)  # Keep the numbers above the bars visible.
        ax.set(title=f"{metric} on the same real test rows", xlabel="Sheet", ylabel=metric)
    table_ax = fig.add_subplot(grid[2, :])
    table_ax.axis("off")
    columns = ["Sheet", "Test real points", "RMSE", "MAE", "R2", "MSE"]
    cells = [[str(row.Sheet), str(int(row["Test real points"])),
              *[f"{row[name]:.4f}" for name in ("RMSE", "MAE", "R2", "MSE")]]
             for _, row in table.iterrows()]
    comparison = table_ax.table(cellText=cells, colLabels=columns,
                                cellLoc="center", loc="center")
    comparison.auto_set_font_size(False)
    comparison.set_fontsize(10)
    comparison.scale(1, 1.55)
    table_ax.set_title("Exact values (rounded to 4 decimal places)", pad=8)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.04)
    fig.savefig(OUTPUT_DIR / "real_OD_metric_comparison.png", dpi=180)


def main():
    datasets = load_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    n_real_train = int(len(datasets["MAIN"]) * TRAIN_FRACTION)
    results, cv_rows = {}, []
    for sheet, series in datasets.items():
        if sheet == "MAIN":
            train_end = n_real_train
        else:
            # Stop at the same REAL observation; do not train on generated
            # rows leading to the first test observation.
            train_end = int(series.index[series.Real_OD_Number.eq(n_real_train)
                                         & series.Source.astype(str).str.lower().eq("real")][0]) + 1
        results[sheet], folds = fit_and_predict(series, train_end, sheet)
        cv_rows.extend(folds)

    # Only compare measured targets shared by all three series, not generated OD.
    common = datasets["MAIN"].iloc[n_real_train:][["DATE", "Real_OD_Number", "OD"]].copy()
    common = common.rename(columns={"OD": "Actual"})
    for sheet, result in results.items():
        real = result.loc[result.Source.astype(str).str.lower().eq("real"),
                          ["Real_OD_Number", "Prediction"]]
        if real.Real_OD_Number.duplicated().any():
            raise ValueError(f"Duplicate real anchors in {sheet} test data.")
        common = common.merge(real.rename(columns={"Prediction": f"{sheet}_Prediction"}),
                              on="Real_OD_Number", how="left", validate="one_to_one")
        if common[f"{sheet}_Prediction"].isna().any():
            raise ValueError(f"Missing {sheet} prediction at a real OD anchor.")

    summary = pd.DataFrame([
        {"Sheet": sheet, "Test real points": len(common),
         **metrics(common.Actual, common[f"{sheet}_Prediction"])}
        for sheet in SHEETS
    ])
    print("\nHeld-out REAL OD comparison (same target observations):")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print("Lower RMSE/MAE/MSE is better; higher R2 is better.")
    print("WARNING: CG/QG generated points may use the next real OD. Their test inputs can\n"
          "contain information about the target; these scores cannot establish which\n"
          "model forecasts unseen real OD most accurately.")

    summary.to_csv(OUTPUT_DIR / "real_OD_metrics.csv", index=False)
    common.to_csv(OUTPUT_DIR / "real_OD_aligned_predictions.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(OUTPUT_DIR / "training_cross_validation_folds.csv", index=False)
    for sheet, result in results.items():
        result[["Actual", "Prediction", "Source", "Real_OD_Number"]].to_csv(
            OUTPUT_DIR / f"{sheet}_test_predictions.csv", index=False)
    plot_common(common)
    plot_each(results)
    plot_metrics(summary)
    print(f"Saved comparison files to {OUTPUT_DIR}")
    # Display all three figures after training. Figures are also saved above.
    plt.show(block=True)


if __name__ == "__main__":
    main()
