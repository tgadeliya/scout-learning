"""Download METR's MALT dataset from HuggingFace and convert it to Inspect Scout transcripts.

MALT (METR's Manually-reviewed Agentic Labeled Transcripts) is a gated dataset, so
`HF_API_KEY` must be set in `.env` (accept the terms at
https://huggingface.co/datasets/metr-evals/malt-transcripts-public first).

Each MALT row is one agent run, stored as a DAG of message nodes. Scout transcripts
want a linear `list[ChatMessage]`, so `malt_row_to_transcript()` flattens the main
branch of the DAG (see `_main_branch_nodes`).

Usage:
    # download + write a Scout transcript database to ./transcripts
    python dataset.py

    # or from code
    ds = download_malt()
    transcripts = list(malt_to_transcripts(ds))
"""

import asyncio
import json
import os
from typing import Any, Iterator, cast

from datasets import Dataset, load_dataset  # type: ignore[import-untyped]
from dotenv import load_dotenv
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.tool import ToolCall
from inspect_scout import Transcript, transcripts_db

MALT_REPO = "metr-evals/malt-transcripts-public"
MALT_SPLIT = "transcripts"

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
    row: dict[str, Any], source_id: str = "malt", branch_id: int = 0
) -> Transcript:
    """Convert a single MALT row (one agent run) into a Scout `Transcript`."""
    nodes = _main_branch_nodes(row["nodes"], branch_id=branch_id)
    messages = _nodes_to_messages(nodes)
    labels = list(row.get("labels") or [])

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
        metadata={
            field: row.get(field) for field in MALT_METADATA_FIELDS
        }
        # convenience flags so scanners can filter on labels without list ops
        | {
            "label_count": len(labels),
            "is_normal": labels == ["normal"],
        },
    )


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
                        tool_call_id=_tool_call_id(node["parent_node_id"]),
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


if __name__ == "__main__":
    count = asyncio.run(import_malt_to_db())
    print(f"Inserted {count} MALT transcripts into ./transcripts")
