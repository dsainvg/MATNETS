"""Run Benchmark Suite V2 in the requested order."""

from __future__ import annotations

import argparse

from examples.benchmarking_v2.b1_computational_cost import run_b1
from examples.benchmarking_v2.b2_matrix_functions import run_b2
from examples.benchmarking_v2.b3_equivariance_generalization import run_b3
from examples.benchmarking_v2.b4_mnist_scale import run_b4
from examples.benchmarking_v2.b5_cifar10_sample_efficiency import run_b5
from examples.benchmarking_v2.b6_copy_task import run_b6
from examples.benchmarking_v2.b7_transformer_longseq import run_b7
from examples.benchmarking_v2.b8_scaling_law_cifar10 import run_b8


def main(run_ambitious: bool) -> None:
    run_b1()
    run_b2()
    run_b4()
    run_b5()
    run_b3()
    if run_ambitious:
        run_b6()
        run_b7()
        run_b8()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run B1-B8 benchmark suite")
    parser.add_argument(
        "--run-ambitious",
        action="store_true",
        help="Also run B6/B7/B8 (heavier benchmarks).",
    )
    args = parser.parse_args()
    main(run_ambitious=args.run_ambitious)
