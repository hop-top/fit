"""HuggingFace dataset -> fit trace schema converters.

Each converter yields trace dicts matching trace-format-v1.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Iterator


def trace_dict(
    *,
    prompt: str = "",
    domain: str = "general",
    advice_text: str = "Provide a helpful response",
    frontier_output: str = "",
    reward_score: float = 0.5,
    reward_breakdown: dict[str, float] | None = None,
    session_id: str | None = None,
    id: str | None = None,
    timestamp: str | None = None,
    frontier_model: str = "hf-source",
    context: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a valid trace dict with sensible defaults."""
    return {
        "id": id or f"tr-{uuid.uuid4().hex[:8]}",
        "session_id": session_id or f"sess-{uuid.uuid4().hex[:8]}",
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "input": {
            "prompt": prompt,
            "context": context or {},
        },
        "advice": {
            "domain": domain,
            "steering_text": advice_text,
            "confidence": 0.8,
            "version": "1.0",
            "constraints": [],
            "metadata": {},
        },
        "frontier": {
            "model": frontier_model,
            "output": frontier_output,
        },
        "reward": {
            "score": reward_score,
            "breakdown": reward_breakdown or {"quality": reward_score},
        },
        "metadata": metadata or {},
    }


def no_robots_to_traces(
    dataset: Any,
    limit: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Convert HuggingFaceH4/no_robots rows to trace dicts.

    Each row has: category (str), messages (list of {role, content}).
    """
    count = 0
    for row in dataset:
        if limit is not None and count >= limit:
            break

        category = row["category"]
        messages = row["messages"]
        prompt = messages[0]["content"]
        output = messages[-1]["content"]

        yield trace_dict(
            prompt=prompt,
            domain=category,
            advice_text=f"Respond as {category} expert",
            frontier_output=output,
            reward_score=0.8,
        )
        count += 1


def lex_glue_to_traces(
    dataset: Any,
    task: str = "ecthr_a",
    limit: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Convert lex_glue rows to legal-compliance domain traces.

    Each row has: text (str), label (int or list[int]).
    Task name is embedded in the advice for provenance.
    """
    count = 0
    for row in dataset:
        if limit is not None and count >= limit:
            break

        text = row["text"]
        label = row["label"]
        label_str = (
            ",".join(str(l) for l in label)
            if isinstance(label, list)
            else str(label)
        )

        yield trace_dict(
            prompt=text,
            domain="legal-compliance",
            advice_text=f"Classify under {task} labels",
            frontier_output=f"label={label_str}",
            reward_score=0.7,
            metadata={"source": "lex_glue", "task": task},
        )
        count += 1


def oasst2_to_traces(
    dataset: Any,
    limit: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Convert oasst2 rows to multi-turn session traces.

    Groups by ``message_tree_id``, sorts by ``created_date``,
    and yields only assistant turns. Each tree maps to one
    ``session_id`` so downstream can reconstruct episodes.
    """
    from collections import defaultdict

    # Collect all rows grouped by tree
    trees: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dataset:
        trees[row["message_tree_id"]].append(row)

    count = 0
    for tree_id, messages in trees.items():
        messages.sort(key=lambda m: m["created_date"])
        session_id = f"sess-{tree_id[:8]}"

        for msg in messages:
            if limit is not None and count >= limit:
                return
            if msg.get("role") != "assistant":
                continue

            parent_text = msg.get("parent_text", "")
            yield trace_dict(
                prompt=parent_text,
                domain="multi-turn",
                advice_text="Continue the conversation helpfully",
                frontier_output=msg["text"],
                reward_score=0.75,
                session_id=session_id,
                metadata={
                    "source": "oasst2",
                    "message_tree_id": tree_id,
                    "message_id": msg.get("message_id", ""),
                },
            )
            count += 1


def hh_rlhf_to_traces(
    dataset: Any,
    limit: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Convert Anthropic/hh-rlhf rows to reward-labeled trace dicts.

    Each row has: chosen (str), rejected (str).
    Yields TWO traces per row: chosen (reward 1.0), rejected (reward 0.0).
    Domain inferred from content keywords.
    """
    count = 0
    for row in dataset:
        if limit is not None and count >= limit:
            break

        chosen = row["chosen"]
        rejected = row["rejected"]

        harm_keywords = (
            "harm", "dangerous", "unsafe", "kill", "weapon", "illegal"
        )
        chosen_lower = chosen.lower()
        domain = (
            "harmlessness"
            if any(kw in chosen_lower for kw in harm_keywords)
            else "helpfulness"
        )

        prompt = _extract_first_human_turn(chosen)
        session_id = f"sess-{uuid.uuid4().hex[:8]}"

        yield trace_dict(
            prompt=prompt,
            domain=domain,
            advice_text="Provide a helpful and harmless response",
            frontier_output=_extract_last_assistant_turn(chosen),
            reward_score=1.0,
            session_id=session_id,
            frontier_model="hh-rlhf-chosen",
        )

        yield trace_dict(
            prompt=prompt,
            domain=domain,
            advice_text="Provide a helpful and harmless response",
            frontier_output=_extract_last_assistant_turn(rejected),
            reward_score=0.0,
            session_id=session_id,
            frontier_model="hh-rlhf-rejected",
        )
        count += 1


def ultrafeedback_to_traces(
    dataset: Any,
    limit: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Convert HuggingFaceH4/ultrafeedback_binarized rows to multi-dim traces.

    Each row has: chosen (list[{role, content}]),
    rejected (list[{role, content}]), score_chosen, score_rejected.
    Yields TWO traces per row with multi-dimensional reward breakdown.
    """
    count = 0
    for row in dataset:
        if limit is not None and count >= limit:
            break

        chosen_msgs = row.get("chosen", [])
        rejected_msgs = row.get("rejected", [])

        prompt = ""
        if chosen_msgs and isinstance(chosen_msgs, list):
            for msg in chosen_msgs:
                if msg.get("role") == "user":
                    prompt = msg.get("content", "")
                    break

        chosen_output = ""
        if chosen_msgs and isinstance(chosen_msgs, list):
            for msg in reversed(chosen_msgs):
                if msg.get("role") == "assistant":
                    chosen_output = msg.get("content", "")
                    break

        rejected_output = ""
        if rejected_msgs and isinstance(rejected_msgs, list):
            for msg in reversed(rejected_msgs):
                if msg.get("role") == "assistant":
                    rejected_output = msg.get("content", "")
                    break

        score_chosen = float(row.get("score_chosen", 0.8))
        score_rejected = float(row.get("score_rejected", 0.3))

        session_id = f"sess-{uuid.uuid4().hex[:8]}"

        yield trace_dict(
            prompt=prompt,
            domain="instruction-following",
            advice_text="Follow instructions accurately",
            frontier_output=chosen_output,
            reward_score=score_chosen,
            reward_breakdown={
                "helpfulness": min(score_chosen, 1.0),
                "honesty": min(score_chosen * 0.9, 1.0),
                "instruction_following": min(score_chosen * 0.95, 1.0),
                "truthfulness": min(score_chosen * 0.85, 1.0),
            },
            session_id=session_id,
            frontier_model="ultrafeedback-chosen",
        )

        yield trace_dict(
            prompt=prompt,
            domain="instruction-following",
            advice_text="Follow instructions accurately",
            frontier_output=rejected_output,
            reward_score=score_rejected,
            reward_breakdown={
                "helpfulness": min(score_rejected, 1.0),
                "honesty": min(score_rejected * 0.9, 1.0),
                "instruction_following": min(score_rejected * 0.95, 1.0),
                "truthfulness": min(score_rejected * 0.85, 1.0),
            },
            session_id=session_id,
            frontier_model="ultrafeedback-rejected",
        )
        count += 1


def _extract_first_human_turn(text: str) -> str:
    """Extract first Human: turn from hh-rlhf conversation format."""
    parts = text.split("\n\nHuman: ")
    if len(parts) >= 2:
        turn = parts[1].split("\n\nAssistant: ")[0]
        return turn.strip()
    return text[:200]


def _extract_last_assistant_turn(text: str) -> str:
    """Extract last Assistant: turn from hh-rlhf conversation format."""
    parts = text.split("\n\nAssistant: ")
    if len(parts) >= 2:
        return parts[-1].strip()
    return text[:200]
