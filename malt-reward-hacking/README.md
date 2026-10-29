# MALT reward-hacking slice

A small labelled slice of METR's MALT dataset for investigating Inspect Scout: every
run of a reward-hacking behaviour, plus 10x as many benign runs **on the same tasks**.

```bash
python malt-reward-hacking/reward_hacking.py --dry-run   # what it would fetch (~40s, ~1 MB)
python malt-reward-hacking/reward_hacking.py             # -> ./transcripts-rh
python malt-reward-hacking/reward_hacking.py --preset matched
python malt-reward-hacking/check_slice.py ./transcripts-rh
```

Reward-hacking specifics live in `reward_hacking.py`; the general MALT -> Scout
machinery is in `../dataset.py`.

## The presets

|  | `o3` (default) | `matched` |
|---|---|---|
| positives | 18 `o3` runs | 5 `o3-2025-04-16` runs |
| benign | 180 `normal`, same two tasks | 50 `normal`, same two tasks |
| benign models | 6 models | `o3-2025-04-16` only |
| download | 166 MB (3 of 13 shards) | 148 MB (2 shards) |
| database | `./transcripts-rh` | `./transcripts-rh-matched` |

Positives are `hardcoded_solution` / `bypass_constraints` on
`ai_rd_rust_codecontests_inference/main` and `ai_rd_triton_cumsum/main`.

## Why these labels

Reward hacking is rare in MALT — 62 runs dataset-wide:

| label | runs |
|---|---|
| `ignores_task_instructions` | 28 |
| `bypass_constraints` | 23 |
| `hardcoded_solution` | 15 |

All 62 are `run_source = 'unprompted'` and `manually_reviewed = True`: naturally
occurring and human-confirmed, with no elicitation prompt in the transcript. The
sandbagging family is much larger (`refusals` 324, `sabotage` 224,
`reasoning_about_task` 225, `partial_problem_solving` 165, `match_weaker_model` 121)
but entirely `prompted`, so the prompt that induced the behaviour is part of the
transcript and gives a monitor the answer for free.

`o3` is the densest offender: 18 runs, all on the two AI R&D tasks above.

## Two caveats before trusting a result

**The default slice varies the model across classes.** `o3` has *zero* `normal` runs
in MALT — it only appears when misbehaving — so benign controls must come from other
models on the same tasks. A scanner could score well by keying on model style rather
than behaviour. If it does well on `o3`, re-run it on `matched` (same model on both
sides) before believing it learned the behaviour.

**Negative labels are weaker than positive ones.** Every positive is
`manually_reviewed`; most `normal` runs are not (5,353 unreviewed vs 694 reviewed
dataset-wide). `normal` means "nothing was flagged", not "a human confirmed it clean".

## Transcript metadata

Each transcript carries MALT's own columns (`run_id`, `labels`, `run_source`,
`manually_reviewed`, ...) plus:

- `malt_class` — `"reward_hacking"` / `"normal"` / `"other"`
- `is_positive` — ground truth read by `malt_validation_set()`
- `orphan_tool_results` — tool results whose originating call was compacted away.
  MALT runs can contain context compaction ("Part of the history of the conversation
  has been summarized"), which drops the assistant turn that made a call while
  keeping its output. Those results get a `malt-orphan-*` id that no assistant
  message claims, rather than a fabricated caller. 3 of the 198 default-slice
  transcripts have one.

To score a scanner against the labels:

```python
from dataset import malt_validation_set
validation = malt_validation_set(transcripts)   # ValidationCase per transcript
```

## How the download stays small

The whole config is 7,179 runs over 13 shards (1.66 GB), and `nodes` is ~99.9% of
every shard's bytes. So planning is free and only the needed shards are fetched —
see the module docstring in `../dataset.py` for the two-pass detail. Shards range
from 7 MB to 442 MB, so benign runs are drawn from the cheapest shards that can
supply them, preferring shards the positives already require.
