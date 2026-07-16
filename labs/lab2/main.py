import argparse

from experiments.task0 import task0
# from experiments.task1_pml import task1
# from experiments.task2_interface import task2
# from experiments.task3_slab import task3
# from experiments.task4_photonic_crystal import task4
# from experiments.task5_cavity import task5


TASKS = {
    "task0": task0,
    # "task1": task1,
    # "task2": task2,
    # "task3": task3,
    # "task4": task4,
    # "task5": task5,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Introduce in numerical electrodynamics: FDTD"
    )

    parser.add_argument(
        "task",
        choices=TASKS,
        help="",
    )

    args = parser.parse_args()
    TASKS[args.task]()


if __name__ == "__main__":
    main()