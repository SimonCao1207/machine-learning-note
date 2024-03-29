import torch
import math
import numpy as np
import random

class PointSampler:
    def __init__(self, output_size):
        self.output_size = output_size

    def sample_point(self, pt1, pt2, pt3):
        # barycentric coordinates on a triangle
        s, t = sorted([random.random(), random.random()])
        f = lambda i: s * pt1[i] + (t-s)*pt2[i] + (1-t)*pt3[i]
        return (f(0), f(1), f(2))

    def triangle_area(self, pt1, pt2, pt3):
        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)
        s = 0.5 * ( side_a + side_b + side_c )
        return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0)**0.5

    def __call__(self, mesh):
        verts, faces = mesh
        verts = np.array(verts)
        areas = np.zeros(len(faces))

        for i in range(len(areas)):
            areas[i] = self.triangle_area(verts[faces[i][0]],
                                         verts[faces[i][1]],
                                         verts[faces[i][2]]
                                          )
        sampled_faces = random.choices(faces, weights=areas, k=self.output_size)
        sampled_points = np.zeros((self.output_size, 3))
        for i in range(len(sampled_faces)):
            sampled_points[i] = self.sample_point(verts[sampled_faces[i][0]],
                                         verts[sampled_faces[i][1]],
                                         verts[sampled_faces[i][2]]
                                          )
        return sampled_points

class Normalize:
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2
        norm_pc = pointcloud - np.mean(pointcloud, axis=0)
        norm_pc /= np.max(np.linalg.norm(norm_pc, axis=1))
        return norm_pc

class ToTensor:
    def __call__(self, pointcloud):
        assert len(pointcloud.shape) == 2
        return torch.Tensor(pointcloud)

class RandRotation_z:
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2

        theta = random.random() * 2. * math.pi
        rot_matrix = np.array([[ math.cos(theta), - math.sin(theta),    0],
                               [ math.sin(theta),  math.cos(theta),    0],
                               [0,                             0,      1]])
        
        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return  rot_pointcloud

class RandomNoise:
    def __call__(self, pointcloud):
        assert len(pointcloud.shape)==2
        noise = np.random.normal(0, 0.02, (pointcloud.shape))
        noisy_pointcloud = pointcloud + noise
        return  noisy_pointcloud
