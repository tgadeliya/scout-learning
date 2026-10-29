"""Sanity checks for a slice written by `reward_hacking.py`.

Usage: python malt-reward-hacking/check_slice.py ./transcripts-rh-matched
"""

import asyncio
import sys
from pathlib import Path

from inspect_scout import TranscriptContent, transcripts_db

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset import ORPHAN_TOOL_CALL_PREFIX, malt_validation_set  # noqa: E402


async def check(location: str) -> None:
    async with transcripts_db(location) as db:
        total = await db.count()
        transcripts = []
        async for info in db.select():
            transcripts.append(await db.read(info, TranscriptContent(messages="all")))

    classes: dict[str, int] = {}
    unmatched = 0
    compacted = 0
    empty = 0
    for transcript in transcripts:
        classes[transcript.metadata["malt_class"]] = (
            classes.get(transcript.metadata["malt_class"], 0) + 1
        )
        if not transcript.messages:
            empty += 1
        seen: set[str] = set()
        for message in transcript.messages:
            for tool_call in getattr(message, "tool_calls", None) or []:
                seen.add(tool_call.id)
            if message.role == "tool" and message.tool_call_id not in seen:
                # unmatched is only legitimate when the call was compacted away
                if str(message.tool_call_id).startswith(ORPHAN_TOOL_CALL_PREFIX):
                    compacted += 1
                else:
                    unmatched += 1

    positive_tasks = {t.task_id for t in transcripts if t.metadata["is_positive"]}
    benign_tasks = {t.task_id for t in transcripts if t.metadata["malt_class"] == "normal"}
    models = {t.metadata["malt_class"]: set() for t in transcripts}
    for transcript in transcripts:
        models[transcript.metadata["malt_class"]].add(transcript.model)

    print(f"transcripts in db:        {total} (read back {len(transcripts)})")
    print(f"class breakdown:          {classes}")
    print(f"models per class:         { {k: sorted(v) for k, v in models.items()} }")
    print(f"empty transcripts:        {empty}")
    print(f"tool results, call compacted away: {compacted}")
    print(f"tool results, unexplained:         {unmatched}")
    print(f"positive tasks:           {sorted(positive_tasks)}")
    print(f"same-task guarantee:      {positive_tasks <= benign_tasks}")
    print(f"validation cases:         {len(malt_validation_set(transcripts).cases)}")

    assert total == len(transcripts), "read back fewer transcripts than the db holds"
    assert empty == 0, "some transcripts have no messages"
    assert unmatched == 0, "tool messages reference a call that was never made"
    assert compacted == sum(
        t.metadata["orphan_tool_results"] for t in transcripts
    ), "compacted-away tool results don't match the recorded metadata count"
    assert positive_tasks <= benign_tasks, "positives occur on tasks with no benign runs"


if __name__ == "__main__":
    asyncio.run(check(sys.argv[1] if len(sys.argv) > 1 else "./transcripts-rh"))
