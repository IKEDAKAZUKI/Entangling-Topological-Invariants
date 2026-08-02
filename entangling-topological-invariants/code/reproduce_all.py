from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from eti import label_resolution, mixed_chern, pump_figures

JOBS = {
    "fig1": ("Fig. 1", mixed_chern.generate_fig1),
    "fig2": ("Fig. 2", mixed_chern.generate_fig2),
    "figS1": ("Fig. S1 full regeneration", label_resolution.generate_figS_label_resolution),
    "figS1plot": ("Fig. S1 data plot", label_resolution.plot_figS_label_resolution_from_data),
    "fig3": ("Fig. 3 full regeneration", pump_figures.generate_fig3),
    "fig3plot": ("Fig. 3 data plot", pump_figures.plot_fig3_from_data),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce mixed-Chern and pump figures.")
    parser.add_argument("--part", choices=["all", *JOBS], default="all")
    args = parser.parse_args()
    if args.part == "all":
        driver = Path(__file__).with_name("run_calculations.sh")
        os.execv("/bin/bash", ["bash", str(driver)])
    label, job = JOBS[args.part]
    started = time.time()
    job()
    print(f"{label} complete in {time.time() - started:.2f} s", flush=True)


if __name__ == "__main__":
    main()
