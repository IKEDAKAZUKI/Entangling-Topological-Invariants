from __future__ import annotations

import argparse
import time

from eti import robustness, tomography

JOBS = {
    "finite_shot": tomography.finite_shot_tomography_data,
    "readout": tomography.readout_confusion_tomography_data,
    "pump": robustness.pump_robustness_data,
    "coupling": robustness.coupling_graph_data,
    "plot": robustness.plot_figures,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce robustness data and figures.")
    parser.add_argument("--part", choices=["all", *JOBS], default="all")
    args = parser.parse_args()
    started = time.time()
    selected = JOBS.values() if args.part == "all" else [JOBS[args.part]]
    for job in selected:
        job()
    print(f"robustness calculations {args.part} complete in {time.time() - started:.2f} s", flush=True)


if __name__ == "__main__":
    main()
