"""Download MALT's reward-hacking runs as a small, labelled Scout transcript database.

Everything specific to reward hacking lives here — which labels count, which tasks
to draw from, and which model/benign pairing to use. The generic MALT -> Scout
machinery it builds on (two-pass planning and downloading, DAG flattening, metadata)
is in `dataset.py`.

The slice is deliberately imbalanced to resemble real monitoring: every run of the
behaviour, plus `neg_per_pos` times as many benign runs **on the same tasks**, so a
scanner's verdicts can't be explained by task difficulty. See README.md for the
measurements behind the presets and for two caveats worth knowing before you trust a
result.

Usage:
    python malt-reward-hacking/reward_hacking.py --dry-run     # no bulk download
    python malt-reward-hacking/reward_hacking.py
    python malt-reward-hacking/reward_hacking.py --preset matched
"""

import argparse
import asyncio
import sys
from pathlib import Path

# the directory name isn't a valid identifier, so this is a script directory
# rather than a package: put the project root on the path to reach dataset.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset import MaltSlice, import_malt_slice  # noqa: E402

# MALT's reward-hacking-family labels: 62 runs dataset-wide, every one of them
# `unprompted` and `manually_reviewed` — naturally occurring and human-confirmed.
# The sandbagging family (`sabotage`, `refusals`, `match_weaker_model`, ...) is far
# larger but entirely `prompted`, so the elicitation prompt sits in the transcript
# and hands a monitor the answer.
REWARD_HACKING_LABELS = (
    "hardcoded_solution",
    "bypass_constraints",
    "ignores_task_instructions",
)

# The two METR AI R&D tasks where reward hacking concentrates. They also carry
# ~1,100 benign runs between them, which is what makes same-task controls possible.
AI_RD_TASKS = (
    "ai_rd_rust_codecontests_inference/main",
    "ai_rd_triton_cumsum/main",
)

# `ignores_task_instructions` is excluded from the presets: it reads as a broader
# "didn't follow the spec" than reward hacking proper, and its runs belong to other
# models anyway. It stays in REWARD_HACKING_LABELS so a wider slice can ask for it.
_PRESET_LABELS = ("hardcoded_solution", "bypass_constraints")

PRESETS: dict[str, MaltSlice] = {
    # o3 is the densest reward hacker in MALT: 18 labelled runs across the two AI
    # R&D tasks. It also has *zero* `normal` runs — in this dataset it only appears
    # when misbehaving — so benign controls have to come from other models on the
    # same tasks. That leaves model style as a possible shortcut for a scanner;
    # "matched" is the control for it.
    "o3": MaltSlice(
        name="reward_hacking_o3",
        model="o3",
        tasks=AI_RD_TASKS,
        positive_labels=_PRESET_LABELS,
        neg_per_pos=10,
        positive_class="reward_hacking",
    ),
    # Same model and same tasks on both sides, so nothing but the behaviour differs.
    # Only 5 positives, but it is the honest comparison.
    "matched": MaltSlice(
        name="reward_hacking_matched",
        model="o3-2025-04-16",
        tasks=AI_RD_TASKS,
        positive_labels=_PRESET_LABELS,
        neg_per_pos=10,
        benign_model="o3-2025-04-16",
        positive_class="reward_hacking",
    ),
}

DEFAULT_LOCATIONS = {
    "o3": "./transcripts-rh",
    "matched": "./transcripts-rh-matched",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        default="o3",
        choices=sorted(PRESETS),
        help="o3: 18 positives, benign from other models. "
        "matched: 5 positives, same model both sides.",
    )
    parser.add_argument(
        "--neg-per-pos",
        type=int,
        default=None,
        help="benign runs per positive (default 10)",
    )
    parser.add_argument(
        "--location", default=None, help="Scout database directory to write"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show what would be fetched, without the shard downloads",
    )
    args = parser.parse_args()

    location = args.location or DEFAULT_LOCATIONS[args.preset]
    plan = asyncio.run(
        import_malt_slice(
            slice=PRESETS[args.preset],
            location=location,
            neg_per_pos=args.neg_per_pos,
            dry_run=args.dry_run,
        )
    )
    if not args.dry_run:
        print(f"Inserted {len(plan.runs)} MALT transcripts into {location}")


if __name__ == "__main__":
    main()
