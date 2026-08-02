from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def load_named(name: str) -> np.ndarray:
    path = DATA / name
    require(path.is_file(), f"Missing data file: {name}")
    return np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8"))


def row_at(arr: np.ndarray, field: str, value: float) -> np.void:
    values = np.asarray(arr[field], dtype=float)
    index = np.flatnonzero(np.isclose(values, value, rtol=0.0, atol=1e-10))
    require(index.size == 1, f"Expected one row with {field}={value}, found {index.size}")
    return arr[index[0]]


def close(actual: float, expected: float, tolerance: float, label: str) -> None:
    require(math.isfinite(float(actual)), f"Non-finite value for {label}")
    require(abs(float(actual) - expected) <= tolerance,
            f"{label}: {actual} differs from {expected} by more than {tolerance}")


def validate_csv_schemas() -> int:
    count = 0
    for path in sorted(DATA.glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.reader(stream))
        require(rows, f"Empty CSV: {path.name}")
        width = max(len(row) for row in rows if row)
        require(width > 0, f"CSV has no fields: {path.name}")
        start = 1
        if len(rows[0]) == width:
            header = rows[0]
            require(len(header) == len(set(header)), f"Duplicate columns in {path.name}")
        for line_number, row in enumerate(rows[start:], start=2):
            require(len(row) == width,
                    f"CSV schema mismatch in {path.name}:{line_number}: {len(row)} != {width}")
        count += 1
    require(count > 0, "No CSV files found")
    return count


def validate_scientific_values() -> dict[str, float]:
    mixing = load_named("generic_mixing_curvature_integrals.csv")
    row = row_at(mixing, "epsilon", 1.2)
    projected = float(row["projected"])
    correction = float(row["gauss_codazzi"])
    total = float(row["determinant_line"])
    close(total, 4.0, 2e-5, "mixed Chern number")
    close(projected + correction, total, 2e-5, "curvature decomposition")

    endpoints = load_named("finite_time_pump_endpoints.csv")
    finite = row_at(endpoints, "ramp_time", 100.0)
    close(float(finite["B_transfer"]), 4.0, 2e-2, "finite-time B transfer")
    require(abs(float(finite["total_transfer"])) < 5e-3, "total transfer is too large")
    require(abs(float(finite["A_transfer"])) < 5e-3, "A transfer is too large")

    c2 = load_named("structured_s4_c2_direct.csv")
    c2_row = row_at(c2, "lambda", 1.25)
    close(float(c2_row["C2_order8"]), 1.0, 2e-3, "second Chern number")

    winding = load_named("clutching_tomography_convergence.csv")
    n6 = row_at(winding, "grid_N", 6.0)
    close(float(n6["W3_odd_bundle"]), 1.0, 1e-2, "odd-bundle winding")
    close(float(n6["W3_factorized_reference"]), 2.0, 1e-2,
          "factorized-reference winding")

    shots = load_named("tomography_finite_shot_summary.csv")
    for code in (0.0, 1.0):
        subset = shots[np.isclose(np.asarray(shots["kind_code"], float), code)]
        subset = subset[np.isclose(np.asarray(subset["shots_per_setting"], float), 200.0)]
        require(subset.size == 1, "Missing finite-shot summary at 200 shots per setting")
        close(float(subset[0]["parity_correct_rate"]), 1.0, 1e-12,
              "finite-shot parity success rate")

    return {
        "mixed_chern": total,
        "finite_time_B_transfer": float(finite["B_transfer"]),
        "second_chern": float(c2_row["C2_order8"]),
        "odd_winding": float(n6["W3_odd_bundle"]),
        "factorized_winding": float(n6["W3_factorized_reference"]),
    }


def validate_figures() -> int:
    names = [
        "fig1_zero_marginal.pdf",
        "fig2_generic_mixing.pdf",
        "fig3_edge_realspace_pump.pdf",
        "fig4_full_reduction_tomography.pdf",
        "figS1_label_resolution.pdf",
        "figS2_pump_robustness.pdf",
        "figS5_tomography_systematics.pdf",
    ]
    for name in names:
        path = FIGURES / name
        require(path.is_file() and path.stat().st_size > 1000, f"Missing or empty figure: {name}")
    return len(names)


def main() -> None:
    csv_count = validate_csv_schemas()
    values = validate_scientific_values()
    figure_count = validate_figures()
    print(f"Validated {csv_count} CSV files and {figure_count} figures.")
    for key, value in values.items():
        print(f"  {key}: {value:.6g}")


if __name__ == "__main__":
    main()
