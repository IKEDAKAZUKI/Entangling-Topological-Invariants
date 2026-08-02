from __future__ import annotations

import argparse
import time

from eti import s4_pipeline

JOBS = {
    "base": s4_pipeline.structured_mixing_data,
    "grid": s4_pipeline.tomography_grid_data,
    "hemisphere": s4_pipeline.hemisphere_patch_scan,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the structured S4 calculations.")
    parser.add_argument("--part", choices=["all", *JOBS], default="all")
    args = parser.parse_args()
    started = time.time()
    selected = JOBS.values() if args.part == "all" else [JOBS[args.part]]
    for job in selected:
        job()
    print(f"structured S4 calculations {args.part} complete in {time.time() - started:.2f} s", flush=True)


if __name__ == "__main__":
    main()
