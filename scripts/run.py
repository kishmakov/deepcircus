#!/usr/bin/env python3

import sys
from pathlib import Path


DEEPCIRCUS_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(DEEPCIRCUS_DIR))


def main() -> None:
    from src.sampler import GeneratorProxy
    from src.train import run_training

    generator = GeneratorProxy(16)
    try:
        run_training(generator)
    finally:
        generator.close()


if __name__ == "__main__":
    main()
