"""直接运行，观察相位变化如何改变干涉输出。"""
import numpy as np
from src.mzi import mzi


def main():
    E1 = E2 = 1 / np.sqrt(2)
    for phi in (0, np.pi / 2, np.pi):
        output = np.asarray(mzi(E1, E2, np.pi / 4, phi))
        power = np.abs(output) ** 2
        print(f"phi={phi:.6f} rad")
        print("  output fields:", np.round(output, 6))
        print("  output powers:", np.round(power, 6))
        print(f"  total input power=1; total output power={power.sum():.12f}")


if __name__ == '__main__':
    main()
