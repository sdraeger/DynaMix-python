"""
DynaMix Evaluation Script
Evaluates the trained model on various datasets from the paper:
- 3D Dynamical Systems: Lorenz-63, Chua, Sprott M
- 2D Dynamical Systems: Selkov
- Time Series: ETTh1, Cloud Requests
"""

import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
from pathlib import Path

from dynamix.model.dynamix import DynaMix
from dynamix.model.forecaster import DynaMixForecaster
from dynamix.metrics.metrics import (
    geometrical_misalignment,
    temporal_misalignment,
    MASE,
)
from dynamix.utilities.plotting_eval import (
    plot_3D_attractor,
    plot_2D_attractor,
    plot_TS_forecast,
)
from dynamix.utilities.utilities import load_model


def load_trained_model(model_path, config_path=None, device="cpu"):
    """Load a locally trained DynaMix model"""
    import json

    # Load config if provided
    if config_path:
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        # Default config matching training
        config = {
            "latent_dim": 30,
            "hidden_dim": 50,
            "experts": 10,
            "pwl_units": 2,
            "expert_type": "almost_linear_rnn",
            "probabilistic_expert": False,
        }

    # Create model
    model = DynaMix(
        M=config["latent_dim"],
        N=3,  # 3D output
        Experts=config["experts"],
        expert_type=config["expert_type"],
        P=config["pwl_units"],
        hidden_dim=config["hidden_dim"],
        probabilistic_expert=config.get("probabilistic_expert", False),
    )

    # Load weights
    model = load_model(model_path, model, device=device)
    model.eval()

    return model


def evaluate_3d_system(
    forecaster,
    data_path,
    system_name,
    context_length=2000,
    forecast_length=10000,
    save_dir=None,
    device="cpu",
):
    """Evaluate on a 3D dynamical system"""
    print(f"\n{'=' * 60}")
    print(f"Evaluating {system_name}")
    print(f"{'=' * 60}")

    # Load data
    data = np.load(data_path)
    context = data[:context_length, :]
    ground_truth = data[context_length : context_length + forecast_length, :]

    print(f"Data shape: {data.shape}")
    print(f"Context length: {context_length}")
    print(f"Forecast length: {forecast_length}")

    # Convert to tensor and move to device
    context_tensor = torch.tensor(context, dtype=torch.float32).to(device)

    # Forecast
    with torch.no_grad():
        reconstruction = forecaster.forecast(
            context=context_tensor,
            horizon=forecast_length,
            standardize=True,
        )

    reconstruction_np = reconstruction.cpu().numpy()

    # Calculate metrics
    ground_truth_tensor = torch.tensor(ground_truth, dtype=torch.float32)
    reconstruction_tensor = torch.tensor(reconstruction_np, dtype=torch.float32)

    dstsp = geometrical_misalignment(
        reconstruction_tensor, ground_truth_tensor, n_bins=30
    )
    dh = temporal_misalignment(reconstruction_tensor, ground_truth_tensor, smoothing=20)
    mase = MASE(ground_truth_tensor, reconstruction_tensor, steps=10)

    print(f"\nResults for {system_name}:")
    print(f"  Geometrical Disagreement (D_stsp): {dstsp:.4f}")
    print(f"  Temporal Disagreement (D_H): {dh:.4f}")
    print(f"  Prediction Error (MASE): {mase:.4f}")

    # Create visualization
    fig = plot_3D_attractor(
        context,
        reconstruction_np,
        ground_truth=ground_truth,
        lim_pse=500,
    )
    fig.suptitle(f"{system_name} - Zero-Shot DSR", fontsize=14, y=1.02)

    if save_dir:
        save_path = (
            Path(save_dir) / f"{system_name.lower().replace(' ', '_')}_evaluation.png"
        )
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved figure to: {save_path}")

    plt.close(fig)

    return {
        "system": system_name,
        "D_stsp": dstsp,
        "D_H": dh,
        "MASE": mase,
    }


def evaluate_2d_system(
    forecaster,
    data_path,
    system_name,
    forecast_length=100,
    initial_point=None,
    save_dir=None,
    device="cpu",
):
    """Evaluate on a 2D dynamical system"""
    print(f"\n{'=' * 60}")
    print(f"Evaluating {system_name}")
    print(f"{'=' * 60}")

    # Load context data and move to device
    context_2d = torch.tensor(np.load(data_path), dtype=torch.float32).to(device)
    print(f"Context shape: {context_2d.shape}")

    if initial_point is None:
        initial_point = torch.tensor([1.0, -1.0], dtype=torch.float32).to(device)
    else:
        initial_point = initial_point.to(device)

    # Forecast
    with torch.no_grad():
        reconstruction = forecaster.forecast(
            context=context_2d,
            horizon=forecast_length,
            initial_x=initial_point,
            preprocessing_method="zero_embedding",
            standardize=False,
        )

    reconstruction_np = reconstruction.cpu().numpy()
    print(f"Reconstruction shape: {reconstruction_np.shape}")

    # Visualize
    fig = plot_2D_attractor(context_2d.cpu().numpy(), reconstruction_np)
    fig.suptitle(f"{system_name} - Zero-Shot 2D DSR", fontsize=14)

    if save_dir:
        save_path = (
            Path(save_dir) / f"{system_name.lower().replace(' ', '_')}_evaluation.png"
        )
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved figure to: {save_path}")

    plt.close(fig)

    return {"system": system_name, "forecast_length": forecast_length}


def evaluate_time_series(
    forecaster,
    data_path,
    system_name,
    context_length=200,
    forecast_length=100,
    fit_nonstationary=False,
    save_dir=None,
    device="cpu",
):
    """Evaluate on a time series forecasting task"""
    print(f"\n{'=' * 60}")
    print(f"Evaluating {system_name}")
    print(f"{'=' * 60}")

    # Load data
    ts_data = np.load(data_path)
    if ts_data.ndim == 1:
        ts_data = ts_data.reshape(-1, 1)

    context = ts_data[:context_length]
    ground_truth = ts_data[context_length : context_length + forecast_length]

    print(f"Data shape: {ts_data.shape}")
    print(f"Context length: {context_length}")
    print(f"Forecast length: {forecast_length}")

    # Convert to tensor and move to device
    context_tensor = torch.tensor(context, dtype=torch.float32).to(device)

    # Forecast
    with torch.no_grad():
        reconstruction = forecaster.forecast(
            context=context_tensor,
            horizon=forecast_length,
            preprocessing_method="pos_embedding",
            standardize=True,
            fit_nonstationary=fit_nonstationary,
        )

    reconstruction_np = reconstruction.cpu().numpy()

    # Visualize
    fig = plot_TS_forecast(
        context, reconstruction_np, ground_truth=ground_truth, lim=forecast_length
    )
    fig.suptitle(f"{system_name} - Zero-Shot Forecast", fontsize=14)

    if save_dir:
        save_path = (
            Path(save_dir) / f"{system_name.lower().replace(' ', '_')}_evaluation.png"
        )
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved figure to: {save_path}")

    plt.close(fig)

    return {"system": system_name}


def main():
    # Setup paths
    base_dir = Path("/nadata/cnl/xaos/simon/DynaMix-python")
    test_data_dir = base_dir / "notebooks" / "test_data"
    model_path = (
        base_dir
        / "results"
        / "my_dynamix_model"
        / "checkpoints"
        / "final_model.safetensors"
    )
    config_path = base_dir / "results" / "my_dynamix_model" / "config.json"
    save_dir = base_dir / "evaluation_results"
    save_dir.mkdir(exist_ok=True)

    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load trained model
    print("\nLoading trained DynaMix model...")
    model = load_trained_model(str(model_path), str(config_path), device=device)
    forecaster = DynaMixForecaster(model)
    print("Model loaded successfully!")

    # Store all results
    all_results = []

    # =========================================================================
    # 1. Evaluate on 3D Dynamical Systems (DSR)
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 1: 3D DYNAMICAL SYSTEMS RECONSTRUCTION")
    print("=" * 70)

    # Lorenz-63 (chaotic attractor)
    results = evaluate_3d_system(
        forecaster,
        test_data_dir / "lorenz63.npy",
        "Lorenz-63",
        context_length=2000,
        forecast_length=10000,
        save_dir=save_dir,
        device=device,
    )
    all_results.append(results)

    # Chua circuit (double scroll attractor)
    results = evaluate_3d_system(
        forecaster,
        test_data_dir / "chua.npy",
        "Chua Circuit",
        context_length=1028,
        forecast_length=10000,
        save_dir=save_dir,
        device=device,
    )
    all_results.append(results)

    # Sprott M (chaotic attractor)
    results = evaluate_3d_system(
        forecaster,
        test_data_dir / "sprottM.npy",
        "Sprott M",
        context_length=512,
        forecast_length=10000,
        save_dir=save_dir,
        device=device,
    )
    all_results.append(results)

    # =========================================================================
    # 2. Evaluate on 2D Dynamical Systems
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 2: 2D DYNAMICAL SYSTEMS (LIMIT CYCLES)")
    print("=" * 70)

    # Selkov system (glycolysis oscillations)
    results = evaluate_2d_system(
        forecaster,
        test_data_dir / "selkov.npy",
        "Selkov System",
        forecast_length=100,
        save_dir=save_dir,
        device=device,
    )
    all_results.append(results)

    # =========================================================================
    # 3. Evaluate on Time Series Forecasting
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 3: TIME SERIES FORECASTING")
    print("=" * 70)

    # ETTh1 (Electricity Transformer Temperature)
    results = evaluate_time_series(
        forecaster,
        test_data_dir / "ETTh1.npy",
        "ETTh1 (Electricity)",
        context_length=100,
        forecast_length=100,
        save_dir=save_dir,
        device=device,
    )
    all_results.append(results)

    # Cloud Requests (Huawei Cloud)
    results = evaluate_time_series(
        forecaster,
        test_data_dir / "cloud_requests.npy",
        "Cloud Requests",
        context_length=512,
        forecast_length=512,
        save_dir=save_dir,
        device=device,
    )
    all_results.append(results)

    # Air Passengers (non-stationary)
    results = evaluate_time_series(
        forecaster,
        test_data_dir / "AirPassengers.npy",
        "Air Passengers",
        context_length=200,
        forecast_length=80,
        fit_nonstationary=True,
        save_dir=save_dir,
        device=device,
    )
    all_results.append(results)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    print("\n3D DSR Results:")
    print("-" * 50)
    print(f"{'System':<20} {'D_stsp':<12} {'D_H':<12} {'MASE':<12}")
    print("-" * 50)
    for r in all_results:
        if "D_stsp" in r:
            print(
                f"{r['system']:<20} {r['D_stsp']:<12.4f} {r['D_H']:<12.4f} {r['MASE']:<12.4f}"
            )

    print(f"\nAll figures saved to: {save_dir}")
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
