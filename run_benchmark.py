"""Run the audit benchmark without overwriting legacy experiments."""
import argparse
from learnlens.benchmark import run_benchmark

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--updates', type=int, default=400)
    p.add_argument('--seeds', type=int, default=3)
    p.add_argument('--output', default='outputs/benchmark.json')
    args = p.parse_args()
    run_benchmark(updates=args.updates, seeds=tuple(range(args.seeds)), output=args.output)
