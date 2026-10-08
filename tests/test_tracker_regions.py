"""Region index tables in ``bedhead.tracker`` stay canonical and duplicate-free.

Canonical vertex sets are copied from the MediaPipe Face Mesh topology exposed
by the installed package as ``mediapipe.tasks.python.vision
.FaceLandmarksConnections`` (contours are order-independent vertex sets).

Note MediaPipe names eyes/brows from the *subject's* perspective, so the
tracker's viewer-left ring is MediaPipe's RIGHT contour and vice versa.
"""

import pytest

import bedhead.tracker as tr

MAX_LANDMARK = 478  # 468 mesh points + 10 iris refinements

ALL_REGIONS = [
    "FACE_OVAL",
    "LEFT_EYE_RING",
    "RIGHT_EYE_RING",
    "LEFT_BROW",
    "RIGHT_BROW",
    "OUTER_LIPS",
    "INNER_LIPS",
    "FOREHEAD_BAND",
    "CHEEK_SAMPLES",
    "FOREHEAD_SAMPLES",
]

# --- canonical Face Mesh contour vertex sets (see module docstring) ---

CANONICAL_FACE_OVAL = frozenset({
    10, 21, 54, 58, 67, 93, 103, 109, 127, 132, 136, 148, 149, 150, 152, 162,
    172, 176, 234, 251, 284, 288, 297, 323, 332, 338, 356, 361, 365, 377, 378,
    379, 389, 397, 400, 454,
})

CANONICAL_RIGHT_EYE = frozenset({  # subject's right = viewer left
    7, 33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173, 246,
})

CANONICAL_LEFT_EYE = frozenset({  # subject's left = viewer right
    249, 263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398,
    466,
})

CANONICAL_RIGHT_EYEBROW = frozenset({46, 52, 53, 55, 63, 65, 66, 70, 105, 107})
CANONICAL_LEFT_EYEBROW = frozenset({276, 282, 283, 285, 293, 295, 296, 300, 334, 336})

CANONICAL_OUTER_LIPS = frozenset({
    0, 17, 37, 39, 40, 61, 84, 91, 146, 181, 185, 267, 269, 270, 291, 314, 321,
    375, 405, 409,
})

CANONICAL_INNER_LIPS = frozenset({
    13, 14, 78, 80, 81, 82, 87, 88, 95, 178, 191, 308, 310, 311, 312, 317, 318,
    324, 402, 415,
})


@pytest.mark.parametrize("name", ALL_REGIONS)
def test_indices_are_ints_in_range(name):
    region = getattr(tr, name)
    assert len(region) > 0
    for i in region:
        assert isinstance(i, int), (name, i)
        assert 0 <= i < MAX_LANDMARK, (name, i)


@pytest.mark.parametrize("name", ALL_REGIONS)
def test_no_duplicate_indices(name):
    region = getattr(tr, name)
    assert len(region) == len(set(region)), f"{name} contains duplicate indices"


@pytest.mark.parametrize(
    ("region_name", "canonical"),
    [
        ("FACE_OVAL", CANONICAL_FACE_OVAL),
        ("LEFT_EYE_RING", CANONICAL_RIGHT_EYE),
        ("RIGHT_EYE_RING", CANONICAL_LEFT_EYE),
        ("LEFT_BROW", CANONICAL_RIGHT_EYEBROW),
        ("RIGHT_BROW", CANONICAL_LEFT_EYEBROW),
        ("OUTER_LIPS", CANONICAL_OUTER_LIPS),
        ("INNER_LIPS", CANONICAL_INNER_LIPS),
    ],
)
def test_named_contours_match_canonical_topology(region_name, canonical):
    region = getattr(tr, region_name)
    assert set(region) == canonical, (
        f"{region_name} diverges from canonical Face Mesh contour "
        f"(missing={sorted(canonical - set(region))}, "
        f"extra={sorted(set(region) - canonical)})"
    )


def test_every_public_index_tuple_is_covered():
    found = {
        name
        for name in dir(tr)
        if name.isupper() and isinstance(getattr(tr, name), tuple)
    }
    assert found == set(ALL_REGIONS)


def _vertices(connections):
    pts = set()
    for conn in connections:
        pts.add(conn.start)
        pts.add(conn.end)
    return frozenset(pts)


def _components(connections):
    adj: dict[int, set[int]] = {}
    for conn in connections:
        adj.setdefault(conn.start, set()).add(conn.end)
        adj.setdefault(conn.end, set()).add(conn.start)
    seen: set[int] = set()
    comps = []
    for start in adj:
        if start in seen:
            continue
        stack = [start]
        comp: set[int] = set()
        while stack:
            node = stack.pop()
            if node in comp:
                continue
            comp.add(node)
            stack.extend(adj[node] - comp)
        seen |= comp
        comps.append(frozenset(comp))
    return comps


def test_copied_canonical_sets_match_installed_mediapipe():
    """The hardcoded contours are the installed package's topology, not a guess."""
    from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections as conn

    assert _vertices(conn.FACE_LANDMARKS_FACE_OVAL) == CANONICAL_FACE_OVAL
    assert _vertices(conn.FACE_LANDMARKS_RIGHT_EYE) == CANONICAL_RIGHT_EYE
    assert _vertices(conn.FACE_LANDMARKS_LEFT_EYE) == CANONICAL_LEFT_EYE
    assert _vertices(conn.FACE_LANDMARKS_RIGHT_EYEBROW) == CANONICAL_RIGHT_EYEBROW
    assert _vertices(conn.FACE_LANDMARKS_LEFT_EYEBROW) == CANONICAL_LEFT_EYEBROW
    assert set(_components(conn.FACE_LANDMARKS_LIPS)) == {
        CANONICAL_OUTER_LIPS,
        CANONICAL_INNER_LIPS,
    }
