#!/usr/bin/env python3
"""Extract timestamped robot trajectories from Khronos DSG agent layers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def extract_agents(path: Path) -> dict:
    graph = json.loads(path.read_text())
    poses = []
    for node in graph.get("nodes", []):
        attributes = node.get("attributes", {})
        if attributes.get("type") != "AgentNodeAttributes":
            continue
        poses.append(
            (
                int(attributes["timestamp"]),
                [float(value) for value in attributes["position"]],
                int(node["id"]),
            )
        )
    poses.sort(key=lambda item: item[0])
    if not poses:
        raise RuntimeError(f"No AgentNodeAttributes in {path}")
    return {
        "source": str(path.resolve()),
        "certainty": "DIRECT_DSG_AGENT_LAYER",
        "pose_count": len(poses),
        "timestamps_ns": [item[0] for item in poses],
        "positions": [item[1] for item in poses],
        "node_ids": [item[2] for item in poses],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-a-dsg", type=Path, required=True)
    parser.add_argument("--session-b-dsg", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    result = {
        "schema": "base1_session_trajectories_v1",
        "sessions": {
            "A": extract_agents(args.session_a_dsg),
            "B": extract_agents(args.session_b_dsg),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"SAVED {args.out} "
        f"A={result['sessions']['A']['pose_count']} "
        f"B={result['sessions']['B']['pose_count']}"
    )


if __name__ == "__main__":
    main()
