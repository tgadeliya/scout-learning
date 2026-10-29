"""Download METR's MALT dataset from HuggingFace and convert it to Inspect Scout transcripts.

MALT (METR's Manually-reviewed Agentic Labeled Transcripts) is a gated dataset, so
`HF_API_KEY` must be set in `.env` (accept the terms at
https://huggingface.co/datasets/metr-evals/malt-transcripts-public first).

Each MALT row is one agent run, stored as a DAG of message nodes. Scout transcripts
want a linear `list[ChatMessage]`, so `malt_row_to_transcript()` flattens the main
branch of the DAG (see `_main_branch_nodes`).

The whole default config is 7,179 runs over 13 parquet shards (~1.7 GB), which is
more than an investigation needs. `import_malt_slice()` pulls a small labelled slice
instead — every run carrying some label, plus N benign runs on the same tasks — in
two passes, because the `nodes` column is 284 MB of each 285 MB shard while every
metadata column together is under 1 MB:

1. `plan_malt_slice()` queries the metadata columns over `hf://` with DuckDB. Column
   projection means only those tiny column chunks move, so picking runs out of all
   13 shards takes ~20s and transfers ~a megabyte.
2. `download_malt_slice()` downloads only the shards the plan actually needs and
   filters them locally, row group by row group.

Nothing here is specific to a behaviour: a `MaltSlice` names the labels, model and
tasks it wants. For a slice built around MALT's reward-hacking labels, see
`malt-reward-hacking/`.

Usage:
    # the whole config -> ./transcripts
    python dataset.py --limit 100

    # or from code
    plan = plan_malt_slice(MaltSlice(name="...", model="o3", tasks=(...),
                                     positive_labels=("hardcoded_solution",)))
    transcripts = list(download_malt_slice(plan))
"""

import argparse
import asyncio
import json
import os
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Iterator, Sequence, cast

import duckdb
import pyarrow.parquet as pq
from datasets import Dataset, load_dataset  # type: ignore[import-untyped]
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.tool import ToolCall
from inspect_scout import (
    Transcript,
    ValidationCase,
    ValidationSet,
    transcripts_db,
)

MALT_REPO = "metr-evals/malt-transcripts-public"
MALT_SPLIT = "transcripts"

# All shards of the "default" config, as a DuckDB glob over the HF filesystem.
MALT_PARQUET_GLOB = f"hf://datasets/{MALT_REPO}/data/*.parquet"

# MALT's label for a run where nothing was flagged. Note this is weaker evidence
# than a behaviour label: most `normal` runs are not `manually_reviewed` (5,353 of
# 6,047 dataset-wide), so it means "unflagged", not "confirmed clean".
BENIGN_LABEL = "normal"

# MALT columns carried into transcript metadata. `task_id` and `model` are
# omitted: they are reserved Scout columns and are set on the transcript itself.
MALT_METADATA_FIELDS = (
    "version",
    "run_id",
    "labels",
    "has_chain_of_thought",
    "public",
    "manually_reviewed",
    "run_source",
)


@dataclass(frozen=True)
class MaltSlice:
    """Definition of a small positive/negative slice of MALT.

    The slice holds the *task* fixed across both classes, so a scanner's verdicts
    can't be explained by task difficulty. Holding the *model* fixed too is up to
    the caller, via `benign_model`.
    """

    name: str
    model: str
    """Model whose labelled runs form the positive class."""

    tasks: tuple[str, ...]
    """Tasks to draw from. Negatives are restricted to the tasks the positives
    actually occurred on, which may be a subset of these."""

    positive_labels: tuple[str, ...]
    """MALT labels that make a run a positive."""

    neg_per_pos: int = 10
    """Benign runs per positive run. MALT's real base rate is lower still, but 10x
    keeps an LLM scanner's cost sane."""

    benign_model: str | None = None
    """Model for the negative class. None draws benign runs from any model, which
    is necessary when the positive model has no benign runs of its own."""

    positive_class: str = "positive"
    """Name for the positive class, stamped on each transcript's `malt_class`
    metadata (e.g. "reward_hacking")."""


@dataclass(frozen=True)
class SliceRun:
    """One run selected into a slice, and the shard it has to be read from."""

    run_id: int
    task_id: str
    model: str
    labels: tuple[str, ...]
    shard: str
    """Repo-relative parquet path, e.g. "data/transcripts-00007-of-00013.parquet"."""

    is_positive: bool


@dataclass
class SlicePlan:
    """Which runs to fetch, and which shards they live in."""

    slice: MaltSlice
    positives: list[SliceRun] = field(default_factory=list)
    negatives: list[SliceRun] = field(default_factory=list)
    shard_bytes: dict[str, int] = field(default_factory=dict)
    """Download size per shard. Shards range from 7 MB to 442 MB, so this is what
    makes the plan's cost visible before committing to it."""

    @property
    def runs(self) -> list[SliceRun]:
        return self.positives + self.negatives

    @property
    def shards(self) -> list[str]:
        """Shards to download, positives' shards first (see `_pick_negatives`)."""
        ordered: list[str] = []
        for run in self.runs:
            if run.shard not in ordered:
                ordered.append(run.shard)
        return ordered

    def run_ids_in(self, shard: str) -> set[int]:
        return {run.run_id for run in self.runs if run.shard == shard}

    def summary(self) -> str:
        lines = [
            f"slice '{self.slice.name}': "
            f"{len(self.positives)} positive / {len(self.negatives)} benign "
            f"({self.slice.neg_per_pos}x requested)",
            f"  positive labels: {', '.join(self.slice.positive_labels)}",
            f"  model: {self.slice.model}"
            f" | benign model: {self.slice.benign_model or 'any'}",
            f"  tasks: {', '.join(sorted({r.task_id for r in self.positives}))}",
            f"  shards to download: {len(self.shards)} of 13"
            f" ({self.download_mb():.0f} MB of 1,662 MB)",
        ]
        for shard in self.shards:
            pos = sum(1 for r in self.positives if r.shard == shard)
            neg = sum(1 for r in self.negatives if r.shard == shard)
            size = self.shard_bytes.get(shard, 0) / 1e6
            lines.append(
                f"    {shard}: {pos} positive, {neg} benign ({size:.0f} MB)"
            )
        return "\n".join(lines)

    def download_mb(self) -> float:
        return sum(self.shard_bytes.get(shard, 0) for shard in self.shards) / 1e6


def hf_token() -> str:
    """HuggingFace read token from `.env` (`HF_API_KEY`, or `HF_TOKEN`)."""
    load_dotenv()
    token = os.getenv("HF_API_KEY") or os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_API_KEY not found — add a HuggingFace read token to .env "
            "(MALT is a gated dataset, so its terms must be accepted too)."
        )
    return token


def download_malt(
    name: str = "default",
    repo_id: str = MALT_REPO,
    cache_dir: str | None = None,
) -> Dataset:
    """Download the MALT dataset from HuggingFace.

    Args:
        name: Dataset config — "default" (the transcripts as run) or one of the
            perturbed variants: "irrelevant_detail", "language_mixing",
            "summarize", "vague_cot".
        repo_id: HuggingFace dataset repo.
        cache_dir: Optional HuggingFace cache directory.

    Returns:
        The "transcripts" split (one row per agent run).
    """
    return load_dataset(
        repo_id,
        name=name,
        split=MALT_SPLIT,
        token=hf_token(),
        cache_dir=cache_dir,
    )


def malt_metadata_query(
    where: str, columns: str = "run_id, task_id, model, labels"
) -> list[dict[str, Any]]:
    """Query MALT's metadata columns over `hf://` without downloading content.

    DuckDB pushes the column projection into the parquet reader, so only the
    selected column chunks are fetched over HTTP. `nodes` is 284 MB of each 285 MB
    shard, so leaving it out makes a query across all 13 shards cost ~a megabyte.

    Args:
        where: SQL predicate over MALT's metadata columns.
        columns: Columns to select. `shard` is always added.

    Returns:
        One dict per matching run, including its `shard` (repo-relative path).
    """
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    # the token is a parameter of CREATE SECRET, which doesn't accept bind params
    con.execute("CREATE SECRET hf (TYPE huggingface, TOKEN ?)", [hf_token()])
    sql = f"""
        SELECT {columns},
               'data/' || regexp_extract(filename, '(transcripts-\\d+-of-\\d+\\.parquet)', 1) AS shard
        FROM read_parquet('{MALT_PARQUET_GLOB}', filename=true)
        WHERE {where}
        ORDER BY run_id
    """
    result = con.execute(sql)
    names = [description[0] for description in result.description or []]
    return [dict(zip(names, row)) for row in result.fetchall()]


def plan_malt_slice(slice: MaltSlice) -> SlicePlan:
    """Pass 1: pick the runs for a slice, downloading essentially nothing.

    Raises:
        RuntimeError: If the slice matches no positive runs.
    """
    positive_rows = malt_metadata_query(
        f"model = {_sql_str(slice.model)}"
        f" AND task_id IN {_sql_list(slice.tasks)}"
        f" AND len(list_intersect(labels, {_sql_list(slice.positive_labels)})) > 0"
    )
    if not positive_rows:
        raise RuntimeError(
            f"slice '{slice.name}' matched no runs with labels "
            f"{slice.positive_labels} for model {slice.model}"
        )
    positives = [_slice_run(row, is_positive=True) for row in positive_rows]

    # negatives are restricted to the tasks the positives actually occurred on,
    # which is what makes the comparison same-task
    positive_tasks = sorted({run.task_id for run in positives})
    benign_where = (
        f"task_id IN {_sql_list(positive_tasks)}"
        f" AND list_contains(labels, {_sql_str(BENIGN_LABEL)})"
    )
    if slice.benign_model is not None:
        benign_where += f" AND model = {_sql_str(slice.benign_model)}"
    benign_rows = malt_metadata_query(benign_where)

    shard_bytes = malt_shard_bytes()
    candidates = [_slice_run(row, is_positive=False) for row in benign_rows]
    negatives = _pick_negatives(
        candidates,
        wanted=len(positives) * slice.neg_per_pos,
        positive_shards=[run.shard for run in positives],
        shard_bytes=shard_bytes,
    )
    return SlicePlan(
        slice=slice,
        positives=positives,
        negatives=negatives,
        shard_bytes=shard_bytes,
    )


def malt_shard_bytes() -> dict[str, int]:
    """Download size of every shard of the default config, keyed by repo path."""
    info = HfApi(token=hf_token()).dataset_info(MALT_REPO, files_metadata=True)
    return {
        sibling.rfilename: sibling.size or 0
        for sibling in info.siblings or []
        if sibling.rfilename.startswith("data/")
    }


def download_malt_slice(
    plan: SlicePlan, source_id: str | None = None
) -> Iterator[Transcript]:
    """Pass 2: download only the planned shards and convert the selected runs."""
    for shard in plan.shards:
        yield from _shard_transcripts(plan, shard, source_id=source_id)


async def import_malt_slice(
    slice: MaltSlice,
    location: str,
    neg_per_pos: int | None = None,
    dry_run: bool = False,
) -> SlicePlan:
    """Download a MALT slice and insert it into a Scout transcript database.

    Args:
        slice: The slice to import.
        location: Database location (directory or e.g. `s3://bucket/malt`).
        neg_per_pos: Override the slice's benign-runs-per-positive ratio.
        dry_run: Run pass 1 only and skip the downloads.

    Returns:
        The plan that was executed (or would have been, for `dry_run`).
    """
    if neg_per_pos is not None:
        slice = replace(slice, neg_per_pos=neg_per_pos)

    plan = plan_malt_slice(slice)
    print(plan.summary())
    if dry_run:
        return plan

    async with transcripts_db(location) as db:
        # commit per shard so an interrupted import resumes rather than restarts
        for shard in plan.shards:
            transcripts = list(_shard_transcripts(plan, shard))
            await db.insert(transcripts)
            print(f"  inserted {len(transcripts)} transcripts from {shard}")
    return plan


def malt_validation_set(transcripts: Iterable[Transcript]) -> ValidationSet:
    """Ground truth for a boolean scanner, from MALT's labels.

    Reads the `is_positive` metadata stamped by `malt_row_to_transcript()`. Pass
    to a scanner's `validation_set` to score its results against the labels rather
    than reading them by hand.
    """
    return ValidationSet(
        cases=[
            ValidationCase(
                id=transcript.transcript_id,
                target=bool(transcript.metadata.get("is_positive")),
            )
            for transcript in transcripts
        ]
    )


def malt_to_transcripts(
    dataset: Dataset, source_id: str = "malt", branch_id: int = 0
) -> Iterator[Transcript]:
    """Convert MALT rows into Scout `Transcript` objects.

    Args:
        dataset: A MALT split as returned by `download_malt()`.
        source_id: Value for the transcripts' `source_id`.
        branch_id: DAG branch to flatten into the message thread (0 is the run's
            main thread; other branches are alternate continuations).

    Yields:
        One `Transcript` per run, ready for `transcripts_db().insert()`.
    """
    for row in dataset:
        yield malt_row_to_transcript(
            cast(dict[str, Any], row), source_id=source_id, branch_id=branch_id
        )


def malt_row_to_transcript(
    row: dict[str, Any],
    source_id: str = "malt",
    branch_id: int = 0,
    positive_labels: Sequence[str] = (),
    positive_class: str = "positive",
) -> Transcript:
    """Convert a single MALT row (one agent run) into a Scout `Transcript`.

    Args:
        row: One MALT row.
        source_id: Value for the transcript's `source_id`.
        branch_id: DAG branch to flatten (0 is the run's main thread).
        positive_labels: Labels that mark this run as a positive. Drives the
            `is_positive` / `malt_class` metadata that `malt_validation_set()`
            reads as ground truth; empty means every run is classified from its
            labels alone.
        positive_class: Name for the positive class in `malt_class`.
    """
    nodes = _main_branch_nodes(row["nodes"], branch_id=branch_id)
    messages = _nodes_to_messages(nodes)
    labels = list(row.get("labels") or [])
    is_positive = any(label in positive_labels for label in labels)

    return Transcript(
        transcript_id=f"malt-{row['run_id']}",
        source_type="malt",
        source_id=source_id,
        source_uri=f"hf://datasets/{MALT_REPO}",
        task_set="malt",
        # MALT task ids look like "password_check/6" — family/variant
        task_id=row.get("task_id"),
        model=row.get("model"),
        message_count=len(messages),
        messages=messages,
        metadata={name: row.get(name) for name in MALT_METADATA_FIELDS}
        # ground truth for scanner evaluation (see `malt_validation_set`), and
        # flags so scanners can filter without list ops
        | {
            "label_count": len(labels),
            "is_normal": labels == [BENIGN_LABEL],
            "is_positive": is_positive,
            "malt_class": _malt_class(labels, is_positive, positive_class),
            # tool results whose call was compacted away (see
            # ORPHAN_TOOL_CALL_PREFIX) — rare, but a scanner reasoning about
            # tool use should be able to find the runs with a gap in them
            "orphan_tool_results": sum(
                1
                for message in messages
                if message.role == "tool"
                and str(message.tool_call_id).startswith(ORPHAN_TOOL_CALL_PREFIX)
            ),
        },
    )


def _malt_class(
    labels: Sequence[str], is_positive: bool, positive_class: str
) -> str:
    if is_positive:
        return positive_class
    return BENIGN_LABEL if list(labels) == [BENIGN_LABEL] else "other"


async def import_malt_to_db(
    location: str = "./transcripts",
    name: str = "default",
    limit: int | None = None,
) -> int:
    """Download MALT and insert it into a Scout transcript database.

    Insertion is idempotent per `transcript_id`, so an interrupted import can
    simply be re-run.

    Args:
        location: Database location (directory or e.g. `s3://bucket/malt`).
        name: MALT config to import (see `download_malt`).
        limit: Optional cap on the number of runs to import.

    Returns:
        Number of transcripts inserted.
    """
    dataset = download_malt(name=name)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    transcripts = list(malt_to_transcripts(dataset))
    async with transcripts_db(location) as db:
        await db.insert(transcripts)
    return len(transcripts)


def _sql_str(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _sql_list(values: Sequence[str]) -> str:
    return "[" + ", ".join(_sql_str(value) for value in values) + "]"


def _slice_run(row: dict[str, Any], is_positive: bool) -> SliceRun:
    return SliceRun(
        run_id=row["run_id"],
        task_id=row["task_id"],
        model=row["model"],
        labels=tuple(row["labels"]),
        shard=row["shard"],
        is_positive=is_positive,
    )


def _pick_negatives(
    candidates: list[SliceRun],
    wanted: int,
    positive_shards: Sequence[str],
    shard_bytes: dict[str, int],
) -> list[SliceRun]:
    """Choose benign runs, preferring shards already needed for the positives.

    Benign runs that live in a shard the positives force us to download are free,
    so those come first. Beyond them, shards are ranked by bytes per usable run
    rather than by how many runs they hold: shards range from 7 MB to 442 MB, so
    the shard with the most benign candidates is often not the cheapest way to
    reach `wanted`. Within a shard, candidates are taken in `run_id` order so the
    slice is reproducible.
    """
    by_shard: dict[str, list[SliceRun]] = {}
    for candidate in candidates:
        by_shard.setdefault(candidate.shard, []).append(candidate)

    shards = [shard for shard in dict.fromkeys(positive_shards) if shard in by_shard]
    remaining = [shard for shard in by_shard if shard not in shards]
    shards += sorted(
        remaining,
        key=lambda shard: (
            shard_bytes.get(shard, 0) / min(len(by_shard[shard]), wanted or 1),
            shard,
        ),
    )

    picked: list[SliceRun] = []
    for shard in shards:
        if len(picked) >= wanted:
            break
        take = wanted - len(picked)
        picked.extend(sorted(by_shard[shard], key=lambda run: run.run_id)[:take])
    return picked


def _shard_transcripts(
    plan: SlicePlan, shard: str, source_id: str | None = None
) -> Iterator[Transcript]:
    """Convert the planned runs out of one shard, a few rows at a time."""
    path = hf_hub_download(MALT_REPO, shard, repo_type="dataset", token=hf_token())
    wanted = plan.run_ids_in(shard)
    parquet = pq.ParquetFile(path)

    for group in range(parquet.metadata.num_row_groups):
        # `run_id` alone is cheap to decode, so skip whole row groups up front
        run_ids = parquet.read_row_group(group, columns=["run_id"])["run_id"].to_pylist()
        if not wanted.intersection(run_ids):
            continue
        # row groups hold 200-320 MB of `nodes`, so decode in small batches
        # rather than materializing the group
        for batch in parquet.iter_batches(batch_size=4, row_groups=[group]):
            for row in batch.to_pylist():
                if row["run_id"] in wanted:
                    yield malt_row_to_transcript(
                        row,
                        source_id=source_id or plan.slice.name,
                        positive_labels=plan.slice.positive_labels,
                        positive_class=plan.slice.positive_class,
                    )


def _main_branch_nodes(
    nodes: list[dict[str, Any]], branch_id: int = 0
) -> list[dict[str, Any]]:
    """Flatten the MALT node DAG into the single root→leaf path for `branch_id`.

    Nodes carry `node_id`/`parent_node_id` pointers plus a `branch_id`; branches
    other than 0 are short alternate continuations hanging off the main thread.
    The requested branch's nodes form one chain, which we recover by starting at
    its tail (the node no other node in the branch points to) and walking up.
    """
    in_branch = [node for node in nodes if node["branch_id"] == branch_id]
    if not in_branch:
        return []

    by_id = {node["node_id"]: node for node in in_branch}
    parents = {node["parent_node_id"] for node in in_branch}
    tails = [node for node in in_branch if node["node_id"] not in parents]
    # a well-formed branch has exactly one tail; if not, take the deepest one
    tail = max(tails, key=lambda node: node["node_id"])

    chain: list[dict[str, Any]] = []
    node: dict[str, Any] | None = tail
    while node is not None:
        chain.append(node)
        node = by_id.get(node["parent_node_id"])
    chain.reverse()
    return chain


def _nodes_to_messages(nodes: list[dict[str, Any]]) -> list[ChatMessage]:
    """Map MALT message nodes onto Inspect `ChatMessage`s."""
    messages: list[ChatMessage] = []
    by_id = {node["node_id"]: node for node in nodes}
    for node in nodes:
        message = node["node_data"]["message"]
        role = message["role"]
        content = message["content"] or ""

        match role:
            case "system":
                messages.append(ChatMessageSystem(content=content))
            case "user":
                messages.append(ChatMessageUser(content=content))
            case "assistant":
                function_call = message["function_call"]
                messages.append(
                    ChatMessageAssistant(
                        content=content,
                        tool_calls=(
                            [_tool_call(node["node_id"], function_call)]
                            if function_call
                            else None
                        ),
                    )
                )
            case "function":
                # tool output: MALT points it at the assistant node that called it
                messages.append(
                    ChatMessageTool(
                        content=content,
                        function=message["name"],
                        tool_call_id=_result_tool_call_id(node, by_id),
                    )
                )
            case _:
                raise ValueError(f"Unexpected MALT message role: {role}")

    return messages


def _tool_call(node_id: int, function_call: dict[str, Any]) -> ToolCall:
    arguments = function_call.get("arguments") or ""
    try:
        parsed = json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        parsed = None
    return ToolCall(
        id=_tool_call_id(node_id),
        function=function_call["name"],
        # MALT records raw (often non-JSON) argument text — keep it verbatim so
        # scanners can read it, and only unpack it when it really is a JSON object
        arguments=parsed if isinstance(parsed, dict) else {"arguments": arguments},
    )


def _tool_call_id(node_id: int | None) -> str:
    return f"malt-call-{node_id}"


# A tool result whose originating call didn't survive the run. MALT transcripts
# can contain context compaction ("Part of the history of the conversation has
# been summarized"), which drops the assistant turn that made the call while
# keeping its output. Such a result is given an id no assistant message claims,
# rather than pointing it at whatever node happens to precede it — inventing a
# caller would put words in the agent's mouth in a dataset built to study what
# agents actually did.
ORPHAN_TOOL_CALL_PREFIX = "malt-orphan-"


def _result_tool_call_id(
    node: dict[str, Any], by_id: dict[int, dict[str, Any]]
) -> str:
    parent = by_id.get(node["parent_node_id"])
    if parent is not None:
        parent_message = parent["node_data"]["message"]
        if parent_message["role"] == "assistant" and parent_message["function_call"]:
            return _tool_call_id(parent["node_id"])
    return f"{ORPHAN_TOOL_CALL_PREFIX}{node['node_id']}"


def main() -> None:
    """Import a whole MALT config. For a small labelled slice, see
    `malt-reward-hacking/`."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--name", default="default", help="MALT config to import")
    parser.add_argument("--limit", type=int, default=None, help="cap on runs")
    parser.add_argument("--location", default="./transcripts")
    args = parser.parse_args()

    count = asyncio.run(
        import_malt_to_db(location=args.location, name=args.name, limit=args.limit)
    )
    print(f"Inserted {count} MALT transcripts into {args.location}")


if __name__ == "__main__":
    main()
