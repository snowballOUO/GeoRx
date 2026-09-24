
from __future__ import annotations

import common as C
import stage_native as N


def main() -> None:
    ctx = C.build_context(batch_q=32, device_str="cpu")
    for seed in C.SEEDS:
        N._write_csv(ctx, seed, C.all_pairs())
        print("merged", seed)


if __name__ == "__main__":
    main()
