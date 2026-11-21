"""Accumulate pallet observations without retaining a history of raw points."""

import numpy as np
import pyoctomap


class TargetAccumulator:
    def __init__(self, resolution, proximity_threshold):
        self.resolution = resolution
        self.proximity_threshold = proximity_threshold
        self.reset()

    def reset(self):
        self.octree = None
        self._centroid = None
        self._sample_count = 0

    @property
    def target_points_for_planning(self):
        return self.octree.getNumLeafNodes() if self.octree is not None else 0

    def accumulate(self, filtered_points):
        if not len(filtered_points):
            return
        points = filtered_points.cpu().numpy()
        centroid = points.mean(axis=0)
        if self._centroid is not None and np.linalg.norm(centroid[:2] - self._centroid[:2]) > self.proximity_threshold:
            return

        total_samples = self._sample_count + len(points)
        if self.octree is None:
            self.octree = pyoctomap.OcTree(self.resolution)
            self._centroid = centroid
        else:
            self._centroid += (centroid - self._centroid) * (len(points) / total_samples)
        self._sample_count = total_samples

        for point in points:
            self.octree.updateNode(point, True)
        self.octree.updateInnerOccupancy()