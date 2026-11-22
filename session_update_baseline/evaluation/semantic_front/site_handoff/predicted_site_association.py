"""Associate observations of a disputed surface. All inputs are predictions or depth measurements.

Create one SiteEvidence per spatial hypothesis. A track can have several sites.
Its handoff contains only source-view evidence.
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


class TriangleSurface:
    """Test distance to measured faces, including the interior of a face."""

    def __init__(self, vertices, faces):
        self.faces = np.asarray(vertices, dtype=float)[np.asarray(faces, dtype=np.intp)]
        centers = self.faces.mean(axis=1)
        self.radius = np.linalg.norm(self.faces - centers[:, None], axis=2).max(axis=1)
        self.centers = cKDTree(centers)

    def touches(self, points, tolerance):
        hit = np.zeros(len(points), dtype=bool)
        search_radius = self.radius.max() + tolerance
        for start in range(0, len(points), 64):
            query = points[start:start+64]
            nearby = self.centers.query_ball_point(query, search_radius)
            counts = [len(group) for group in nearby]
            if not sum(counts):
                continue
            keys = np.concatenate(nearby).astype(np.intp)
            which = np.repeat(np.arange(len(query)), counts)
            inside_bound = np.linalg.norm(query[which] - self.centers.data[keys], axis=1) <= self.radius[keys] + tolerance
            keys, which = keys[inside_bound], which[inside_bound]
            if not len(keys):
                continue
            p = query[which]
            a, b, c = self.faces[keys].transpose(1, 0, 2)
            normal = np.cross(b-a, c-a)
            norm2 = np.einsum('ij,ij->i', normal, normal)
            height = np.einsum('ij,ij->i', p-a, normal)
            in_face = norm2 > 0
            edge2 = np.full(len(p), np.inf)
            for u, v in ((a, b), (b, c), (c, a)):
                edge, offset = v-u, p-u
                in_face &= np.einsum('ij,ij->i', np.cross(edge, offset), normal) >= 0
                length2 = np.einsum('ij,ij->i', edge, edge)
                dot = np.einsum('ij,ij->i', edge, offset)
                position = np.divide(dot, length2, out=np.zeros_like(dot), where=length2 > 0).clip(0, 1)
                along = offset - position[:, None] * edge
                edge2 = np.minimum(edge2, np.einsum('ij,ij->i', along, along))
            near = (in_face & (height**2 <= tolerance**2 * norm2)) | (edge2 <= tolerance**2)
            hit[start + which[near]] = True
        return hit


@dataclass(frozen=True)
class Surface:
    track: int
    label: int
    points: np.ndarray
    mesh: TriangleSurface | None = None

    def components(self, resolution):
        cells = np.floor(self.points / resolution).astype(np.int64)
        if self.mesh is None:
            pairs = cKDTree(cells).query_pairs(1, p=np.inf, output_type='ndarray')
            graph = coo_matrix((np.ones(len(pairs), dtype=bool), (pairs[:, 0], pairs[:, 1])),
                               shape=(len(cells), len(cells)))
            return connected_components(graph, directed=False)[1]
        vertices = np.floor(self.mesh.faces.reshape(-1, 3) / resolution).astype(np.int64)
        unique, mapping = np.unique(np.vstack((cells, vertices)), axis=0, return_inverse=True)
        triangles = mapping[len(cells):].reshape(-1, 3)
        edges = np.vstack((triangles[:, :2], triangles[:, 1:]))
        graph = coo_matrix((np.ones(len(edges), dtype=bool), (edges[:, 0], edges[:, 1])),
                           shape=(len(unique), len(unique)))
        groups = connected_components(graph, directed=False)[1]
        return groups[mapping[:len(cells)]]


@dataclass(frozen=True)
class Claim:
    host: int
    shared: np.ndarray  # Boolean mask over this site's surface samples.


@dataclass(frozen=True)
class InheritedSite:
    surface: Surface
    claim: Claim | None
    independent: np.ndarray


def name_regions(anchor_regions, anchor_ids, labels_agree, probe_regions, n_anchors):
    """Name a depth-connected image region by agreement with visible map anchors.

    IDs in anchor_ids are dense indices into the visit-entry map. A tied or
    unsupported region stays unknown. A cross-class probe cannot vote for the
    competing region's name through its own anchor.
    """
    regions = np.concatenate((anchor_regions, probe_regions))
    if not regions.size or not n_anchors:
        return np.full(len(probe_regions), -1, dtype=np.int32)
    unique_regions, inverse = np.unique(regions, return_inverse=True)
    anchor = inverse[:len(anchor_regions)]
    query = inverse[len(anchor_regions):]
    votes = np.bincount(
        anchor[labels_agree] * n_anchors + anchor_ids[labels_agree],
        minlength=len(unique_regions) * n_anchors,
    ).reshape(-1, n_anchors)
    winners = votes.argmax(axis=1)
    majority = 2 * votes[np.arange(len(winners)), winners] > votes.sum(axis=1)
    return np.where(majority[query], winners[query], -1)


class SiteEvidence:
    def __init__(self, surface, anchors, previously_supported):
        self.surface = surface
        self.anchors = anchors
        self.previously_supported = previously_supported
        self.own = np.zeros(len(surface.points), dtype=np.uint32)
        self.other = np.zeros((len(surface.points), len(anchors)), dtype=np.uint32)
        self.last_view = None
        self.contact_masks = {}

    def contact(self, index, resolution):
        key = (index, resolution)
        if key not in self.contact_masks:
            anchor = self.anchors[index]
            if anchor.mesh is not None:
                mask = anchor.mesh.touches(self.surface.points, resolution)
            else:
                mask = cKDTree(anchor.points).query(self.surface.points, p=np.inf)[0] <= resolution
            self.contact_masks[key] = mask
        return self.contact_masks[key]

    def observe(self, stamp, indices, raw_labels, named_anchors):
        """Add at most one unmodified depth-matched observation per surfel and view.

        Occluded points, free space and missing depth have no indices here.
        The caller retains those measurements for its normal geometric update.
        """
        if self.last_view == stamp:
            return
        if self.last_view is not None and stamp < self.last_view:
            raise ValueError("The site received an out-of-order camera view")
        indices = np.asarray(indices, dtype=np.intp)
        raw_labels = np.asarray(raw_labels)
        named_anchors = np.asarray(named_anchors, dtype=np.intp)
        if len(indices) > 1 and not np.all(indices[1:] > indices[:-1]):
            # Multiple camera pixels can hit one surfel. Agreeing pixels get
            # one vote. Conflicting identities in the same view abstain.
            votes = np.where(raw_labels == self.surface.label, -2, named_anchors)
            pairs = np.unique(np.column_stack((indices, votes)), axis=0)
            cells, first, counts = np.unique(pairs[:, 0], return_index=True, return_counts=True)
            indices = cells[counts == 1]
            named_anchors = pairs[first[counts == 1], 1]
            own = named_anchors == -2
        else:
            own = raw_labels == self.surface.label
        self.own[indices[own]] += 1
        other = ~own & (named_anchors >= 0)
        self.other[indices[other], named_anchors[other]] += 1
        self.last_view = stamp

    def claim(self, ray_balance, resolution):
        """Infer a local alias; ray_balance is own support minus same-depth replacements."""
        if not self.anchors or ray_balance.get(self.surface.track, 0) >= 0:
            return None
        total = self.own + self.other.sum(axis=1)
        seen = total > 0
        if not seen.any():
            return None
        # Compare evidence that actually distinguishes the hypotheses. A cell
        # on a rival's mesh where that rival wins supports a shared surface.
        # Own-class evidence outside that mesh supports a separate object.
        # An unresolved pixel explains neither. Each spatial cell counts once.
        groups = {}
        for j in np.flatnonzero(self.other.any(axis=0)):
            host = self.anchors[j].track
            if host != self.surface.track and ray_balance.get(host, 0) > 0:
                groups.setdefault(host, []).append(j)
        own_wins = 2 * self.own.astype(np.uint64) > total
        protected = np.count_nonzero(~seen) if self.previously_supported else 0
        best = None
        largest = total_shared = 0
        for host, indices in groups.items():
            rival_wins = 2 * self.other[:, indices].sum(axis=1) > total
            if not rival_wins.any():
                continue
            contact = np.logical_or.reduce([self.contact(j, resolution) for j in indices])
            shared = rival_wins & contact
            area = np.count_nonzero(shared)
            total_shared += area
            if area > largest:
                largest = area
                independent = np.count_nonzero(own_wins & ~contact) + protected
                best = Claim(host, shared) if area > independent else None
        return best if 2 * largest > total_shared else None


    def handoff(self, claim, resolution):
        """Split this site's current input without retiring the physical ID."""
        if claim is None:
            return InheritedSite(self.surface, None, np.ones(len(self.own), dtype=bool))
        candidates = [j for j, anchor in enumerate(self.anchors) if anchor.track == claim.host]
        # A map ID can own more than one site. Use every site of this host
        # when testing whether the source shows a physically separate part.
        contact = np.zeros(len(self.own), dtype=bool)
        for j in candidates:
            contact |= self.contact(j, resolution)
        separate = (self.own > self.other.sum(axis=1)) & ~contact & ~claim.shared
        # Decide independently for disconnected pieces of a source track. A
        # stray surface cell cannot keep an entire attached duplicate alive.
        # A detached piece with its own observations remains an active site.
        component = self.surface.components(resolution)
        shared_area = np.bincount(component, weights=claim.shared)
        independent_area = np.bincount(component, weights=separate)
        keep = separate & (independent_area[component] >= shared_area[component])
        return InheritedSite(self.surface, claim, keep)
