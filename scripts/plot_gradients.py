#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt


def read_csv(path: Path):
    """Read CSV with columns: batch, conv_grad, bn_grad.

    Returns three lists: batches, conv_values, bn_values.
    Silently skips rows that are malformed or missing fields.
    """
    xs, conv, bn = [], [], []
    with open(path, 'r', newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                xs.append(int(row['batch']))
                conv.append(float(row['conv_grad']))
                bn.append(float(row['bn_grad']))
            except Exception:
                # Skip malformed rows
                continue
    return xs, conv, bn


def compute_ratio(num_list, den_list, eps: float):
    """Return element-wise ratio: num / (den + eps).

    Length is min(len(num_list), len(den_list)).
    """
    return [num / (den + eps) for num, den in zip(num_list, den_list)]


def main():
    parser = argparse.ArgumentParser(
        description='Plot BN/Conv gradient ratio for attacked vs clean runs.'
    )
    parser.add_argument(
        '--attacked', type=str,
        default=str(Path('output') / 'gradients_epoch1.csv'),
        help='CSV path for attacked run (default: output/gradients_epoch1.csv)'
    )
    parser.add_argument(
        '--clean', type=str,
        default=str(Path('output') / 'gradients_epoch1_clean.csv'),
        help='CSV path for clean run (default: output/gradients_epoch1_clean.csv)'
    )
    parser.add_argument(
        '--save', type=str,
        default=str(Path('output') / 'gradients_epoch1_ratio_compare.png'),
        help='Output image path (default: output/gradients_epoch1_ratio_compare.png)'
    )
    parser.add_argument('--show', action='store_true', help='Show the plot window in addition to saving.')
    parser.add_argument('--xmax', type=float, default=None, help='Max x-axis (index) value (default: auto)')
    parser.add_argument('--ymax', type=float, default=None, help='Max y-axis value (default: auto)')
    parser.add_argument('--eps', type=float, default=1e-12, help='Epsilon to avoid division by zero (default: 1e-12)')
    args = parser.parse_args()

    attacked_path = Path(args.attacked)
    clean_path = Path(args.clean)
    out_path = Path(args.save)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data if available
    ax, aconv, abn = [], [], []
    cx, cconv, cbn = [], [], []

    if attacked_path.exists():
        ax, aconv, abn = read_csv(attacked_path)
    else:
        print(f"[WARN] Attacked CSV not found: {attacked_path}")

    if clean_path.exists():
        cx, cconv, cbn = read_csv(clean_path)
    else:
        print(f"[WARN] Clean CSV not found: {clean_path}")

    # Compute BN/Conv ratios (index-based x-axis)
    attacked_ratio = compute_ratio(abn, aconv, args.eps) if (aconv and abn) else []
    clean_ratio = compute_ratio(cbn, cconv, args.eps) if (cconv and cbn) else []

    if not attacked_ratio and not clean_ratio:
        print('[ERROR] No data to plot: both attacked and clean ratios are empty.')
        return

    # Plot
    fig, axp = plt.subplots(1, 1, figsize=(7.5, 4.5))

    if attacked_ratio:
        axp.plot(range(len(attacked_ratio)), attacked_ratio, label='Attacked BN/Conv', color='C3', alpha=0.95)
    if clean_ratio:
        axp.plot(range(len(clean_ratio)), clean_ratio, label='Clean BN/Conv', color='C2', linestyle='--', alpha=0.95)

    axp.set_xlabel('batch index (sequence)')
    axp.set_ylabel('BN/Conv gradient ratio')

    max_x = max(len(attacked_ratio), len(clean_ratio))
    if args.xmax is not None:
        axp.set_xlim(0.0, args.xmax)
    elif max_x > 0:
        axp.set_xlim(0.0, max_x)

    if args.ymax is not None:
        axp.set_ylim(0.0, args.ymax)

    axp.legend()
    axp.set_title('BN/Conv gradient ratio (epoch 1): Attacked vs Clean')
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0.0)

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == '__main__':
    main()
