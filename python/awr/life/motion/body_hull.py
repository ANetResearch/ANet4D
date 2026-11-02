"""Asset-derived conservative convex bodies, compiled once outside physics ticks.

The hull encloses all non-wheel render vertices. It fills concavities, but does
not add empty bonnet/roof corners from the full bounding box. Tyres remain the
separate authored wheel rig; no terrain or vehicle pose is fitted here.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull, QhullError


@dataclass(frozen=True, eq=False)
class BodyHull:
    vertices: np.ndarray

    def __post_init__(self):
        p = np.array(self.vertices, dtype=float, order="C", copy=True)
        if p.ndim != 2 or p.shape[1:] != (3,) or not 4 <= len(p) <= 4096 or not np.isfinite(p).all():
            raise ValueError("Finite bounded 3D hull vertices required")
        try:
            h = ConvexHull(p)
        except QhullError as exc:
            raise ValueError("Nondegenerate body hull required") from exc
        if h.volume < 1e-9:
            raise ValueError("Nondegenerate body hull required")
        p = p[h.vertices].copy()
        p.flags.writeable = False
        equations = h.equations.copy()
        equations.flags.writeable = False
        # Extra separating axes are needed near a mesh triangle/body edge.
        # Triangulation diagonals are harmless additional directions: every
        # proposed axis is checked against ALL vertices before certifying.
        edge_vectors = np.concatenate(
            [h.points[h.simplices[:, i]] - h.points[h.simplices[:, (i + 1) % 3]] for i in range(3)]
        )
        edge_vectors /= np.linalg.norm(edge_vectors, axis=1)[:, None]
        sign = np.sign(edge_vectors[np.arange(len(edge_vectors)), np.argmax(abs(edge_vectors), axis=1)])
        edge_vectors = np.unique(np.round(edge_vectors * sign[:, None], 12), axis=0)
        edge_vectors.flags.writeable = False
        object.__setattr__(self, "vertices", p)
        object.__setattr__(self, "equations", equations)
        object.__setattr__(self, "edge_vectors", edge_vectors)

    @classmethod
    def from_points(cls, points):
        p = np.asarray(points, float)
        if p.ndim != 2 or p.shape[1:] != (3,) or len(p) < 4 or not np.isfinite(p).all():
            raise ValueError("Finite body vertices required")
        try:
            h = ConvexHull(np.unique(p, axis=0))
        except QhullError as exc:
            raise ValueError("Nondegenerate body hull required") from exc
        return cls(h.points[h.vertices])

    @classmethod
    def from_asset(cls, spec, data, partition):
        from awr.life.appearance.wheels import WheelRig

        rig = WheelRig(spec, data, partition)
        keep = np.ones(len(data["p"]), bool)
        keep[rig.rows] = False
        return cls.from_points(data["p"][keep])

    def separated_triangles(self, triangles, padding=0.0):
        """Sufficient separating-plane proof, never a claim of intersection.

        First test unit hull planes, then triangle normals and edge cross
        products for unresolved triangles. An actual projection gap greater
        than padding encloses the same Euclidean swept-body tube. Failure to
        find an axis remains unproven; it is never called an intersection.
        """
        t = np.asarray(triangles, float)
        if t.ndim != 3 or t.shape[1:] != (3, 3) or not np.isfinite(t).all():
            raise ValueError("Finite triangles required")
        if not np.isfinite(padding) or padding < 0:
            raise ValueError("Nonnegative padding required")
        result = np.zeros(len(t), bool)
        # Bound temporary memory for dense photogrammetric patches.
        for first in range(0, len(t), 64):
            q = t[first : first + 64] @ self.equations[:, :3].T + self.equations[:, 3]
            result[first : first + 64] = (q.min(axis=1) > padding + 1e-9).any(axis=1)
        for index in np.flatnonzero(~result):
            triangle = t[index]
            edges = np.roll(triangle, -1, axis=0) - triangle
            axes = np.vstack(
                [
                    np.cross(edges[0], edges[1])[None, :],
                    np.cross(self.edge_vectors[:, None, :], edges[None, :, :]).reshape(-1, 3),
                ]
            )
            norms = np.linalg.norm(axes, axis=1)
            axes = axes[norms > 1e-12] / norms[norms > 1e-12, None]
            # Unit axes make padding a metric distance, with the same strict
            # numerical margin as the original fast path. Chunk projections
            # to avoid a vertices x all-axes temporary for detailed assets.
            for first in range(0, len(axes), 128):
                a = axes[first : first + 128]
                body = self.vertices @ a.T
                other = triangle @ a.T
                gaps = np.maximum(other.min(axis=0) - body.max(axis=0), body.min(axis=0) - other.max(axis=0))
                if (gaps > padding + 1e-9).any():
                    result[index] = True
                    break
        return result
