"""Deterministic image-window construction for 12 GB GPU forward models.

Forward-model windows are constructed from the official COLMAP co-visibility
graph, not from the image filename order.  Numbering is retained only as a
stable tie-breaker; the released collection is not a 400-frame video.
"""

from __future__ import annotations


def sliding_windows(names: list[str], window_size: int, overlap: int) -> list[list[str]]:
    if window_size < 1 or overlap < 0 or overlap >= window_size:
        raise ValueError("window_size must be positive and overlap must satisfy 0 <= overlap < window_size")
    if not names:
        return []
    step = window_size - overlap
    windows = []
    for start in range(0, len(names), step):
        window = names[start : start + window_size]
        if len(window) < 2 and windows:
            windows[-1].extend(window)
            break
        windows.append(window)
        if start + window_size >= len(names):
            break
    return windows


def covisibility_windows(
    graph: dict,
    allowed_names: list[str],
    *,
    window_size: int,
    overlap: int,
    max_windows: int | None = None,
) -> list[dict]:
    """Build deterministic strong-co-visibility windows from an audited graph.

    The first window is rooted at the training camera with the largest sum of
    shared sparse tracks.  Later windows greedily favour previously uncovered
    cameras while requiring the declared overlap with the preceding window.
    This gives the forward model multi-view context without claiming that the
    filenames are a video trajectory.
    """
    if window_size < 2 or overlap < 0 or overlap >= window_size:
        raise ValueError("window_size must be at least two and overlap must satisfy 0 <= overlap < window_size")
    allowed = sorted(set(allowed_names))
    if len(allowed) < window_size:
        raise ValueError(f"Need at least {window_size} allowed images, got {len(allowed)}")
    adjacency = graph.get("adjacency")
    if not isinstance(adjacency, dict):
        raise ValueError("camera graph is missing adjacency")
    node_centers = {
        node.get("name"): node.get("center")
        for node in graph.get("nodes", [])
        if isinstance(node, dict) and node.get("name") and node.get("center") is not None
    }

    allowed_set = set(allowed)

    def edges_for(name: str) -> list[dict]:
        edges = []
        for edge in adjacency.get(name, []):
            target = edge.get("target") if edge.get("source") == name else edge.get("source")
            if target in allowed_set and target != name:
                edges.append({**edge, "_target": target})
        return sorted(
            edges,
            key=lambda edge: (
                -int(edge.get("shared_point3d", 0)),
                float(edge.get("center_distance", float("inf"))),
                edge["_target"],
            ),
        )

    def center_distance(source: str, target: str) -> float:
        source_center, target_center = node_centers.get(source), node_centers.get(target)
        if source_center is None or target_center is None:
            return float("inf")
        return sum((float(a) - float(b)) ** 2 for a, b in zip(source_center, target_center))

    candidates: list[dict] = []
    for root in allowed:
        neighbours = edges_for(root)
        selected_neighbours = neighbours[: window_size - 1]
        selected = [root] + [edge["_target"] for edge in selected_neighbours]
        # A few training cameras have fewer than 15 graph edges after the
        # documented shared-track threshold.  Keep the window size fixed and
        # pad only those cases with nearest official camera centres, recording
        # the weaker context instead of inventing filename/video continuity.
        padding = window_size - len(selected)
        if padding:
            already = set(selected)
            fallback = sorted((name for name in allowed if name not in already), key=lambda name: (center_distance(root, name), name))
            selected.extend(fallback[:padding])
        score = sum(int(edge.get("shared_point3d", 0)) for edge in selected_neighbours)
        candidates.append(
            {
                "root": root,
                "images": selected,
                "score": score,
                "covisibility_neighbor_count": len(selected_neighbours),
                "weak_context_padding_count": padding,
            }
        )

    candidates.sort(key=lambda candidate: (-candidate["score"], candidate["root"]))
    selected: list[dict] = [{**candidates[0], "overlap_with_previous": 0, "chain_restart": True}]
    covered = set(selected[0]["images"])
    desired_count = max_windows
    while len(covered) < len(allowed) and (desired_count is None or len(selected) < desired_count):
        previous = set(selected[-1]["images"])
        unused = [candidate for candidate in candidates if candidate["root"] not in {item["root"] for item in selected}]
        admissible = [
            candidate for candidate in unused if len(previous.intersection(candidate["images"])) >= overlap
        ]
        if not unused:
            raise ValueError("No unused co-visibility window remains to cover training cameras")

        def candidate_key(candidate: dict) -> tuple:
            return (
                -len(set(candidate["images"]) - covered),
                -candidate["score"],
                candidate["root"],
            )

        strongest_global = sorted(unused, key=candidate_key)[0]
        strongest_continuation = sorted(admissible, key=candidate_key)[0] if admissible else None
        # Restart a spatial chain when forcing the prior 4-image overlap would
        # discard substantially more new camera coverage.  The restart is an
        # explicit manifest fact, not a hidden relaxation of the protocol.
        continuation_gain = (
            len(set(strongest_continuation["images"]) - covered) if strongest_continuation is not None else 0
        )
        global_gain = len(set(strongest_global["images"]) - covered)
        # Continue an established chain whenever it retains at least half of
        # its nominal new-view capacity.  This honours the frozen overlap
        # without letting a long bridge sweep degrade into hundreds of nearly
        # duplicate windows; a restart records a genuine spatial context jump.
        minimum_continuation_gain = max(1, (window_size - overlap) // 2)
        chain_restart = strongest_continuation is None or (
            max_windows is None and continuation_gain < minimum_continuation_gain and global_gain > continuation_gain
        )
        if chain_restart:
            # The official graph can contain spatially disjoint flight regions.
            # Do not fake temporal continuity across them: start an explicit
            # independent co-visibility chain instead.  All non-restart
            # neighbours continue to honour the frozen overlap.
            next_candidate = strongest_global
        else:
            next_candidate = strongest_continuation
        assert next_candidate is not None
        actual_overlap = len(previous.intersection(next_candidate["images"]))
        selected.append(
            {
                **next_candidate,
                "overlap_with_previous": actual_overlap,
                "chain_restart": chain_restart,
            }
        )
        covered.update(next_candidate["images"])

    if desired_count is not None and len(selected) < desired_count:
        # A smoke test asks for two windows even when one happens to cover all
        # selected names.  Add the best distinct candidate with the same overlap.
        previous = set(selected[-1]["images"])
        for candidate in candidates:
            if candidate["root"] not in {item["root"] for item in selected} and len(previous.intersection(candidate["images"])) >= overlap:
                selected.append(
                    {
                        **candidate,
                        "overlap_with_previous": len(previous.intersection(candidate["images"])),
                        "chain_restart": False,
                    }
                )
                if len(selected) == desired_count:
                    break
    if desired_count is not None and len(selected) < desired_count:
        raise ValueError(f"Requested {desired_count} co-visibility windows, only built {len(selected)}")

    return [
        {
            "window_id": f"cov-w{index:03d}",
            "root_image": candidate["root"],
            "images": candidate["images"],
            "shared_track_score": candidate["score"],
            "covisibility_neighbor_count": candidate["covisibility_neighbor_count"],
            "weak_context_padding_count": candidate["weak_context_padding_count"],
            "overlap_with_previous": candidate["overlap_with_previous"],
            "chain_restart": candidate["chain_restart"],
            "construction": "official_colmap_covisibility_training_only",
        }
        for index, candidate in enumerate(selected)
    ]
