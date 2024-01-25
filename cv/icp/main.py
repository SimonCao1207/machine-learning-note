import numpy as np
from sklearn.neighbors import NearestNeighbors 
import trimesh
import os

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(FILE_PATH, "data/bunny_v2")
if not os.path.exists(DATA_PATH):
    assert("Cannot find path to data")

def nearest_neighbors(src, dst):
    neigh = NearestNeighbors(n_neighbors=1, algorithm="kd_tree")
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()

def best_fit_transform(src, dst):
    """
        Calculate the least square best-fit transform that maps src to dst in m spatial dimentions
        Returns: 
        T : (m+1)x(m+1) homogenous transformation matrix that map src to dst
        R : mxm rotation matrix
        t : mx1 translation matrix
    """ 
    assert  src.shape == dst.shape
    # Find the centroids of both dataset
    src_centroid = np.mean(src, axis=1)
    dst_centroid = np.mean(dst, axis=1)

    # Bring both datasets to the origin
    src -= src_centroid[np.newaxis, :]
    dst -= dst_centroid[np.newaxis, :]
    pass 

mesh_dst_bunny = os.path.join(DATA_PATH, "bun000_v2.ply")
mesh_src_bunny = os.path.join(DATA_PATH, "bun045_v2.ply")
dst_tm = trimesh.load(mesh_dst_bunny)
src_tm = trimesh.load(mesh_src_bunny)

# Get points and their normals
dst_pts = np.array(dst_tm.vertices)
dst_pts_normal = np.array(dst_tm.vertex_normals)
src_pts = np.array(src_tm.vertices)
src_pts_normal = np.array(src_tm.vertex_normals)

# Subsampling
sample_rate = 1
src_ids = np.random.uniform(0, 1, size=src_pts.shape[0])
A = src_pts[src_ids < sample_rate, :]
A_normals = src_pts_normal[src_ids < sample_rate, :]

dst_ids = np.random.uniform(0, 1, size=dst_pts.shape[0])
B = dst_pts[dst_ids < sample_rate, :]
B_normals = dst_pts_normal[dst_ids < sample_rate, :]

# make points homogenous
m = A.shape[1]
src = np.ones((m+1, A.shape[0]))
dst = np.ones((m+1, B.shape[0]))
src[:m, :] = np.copy(A.T)
dst[:m, :] = np.copy(B.T)

max_iterations = 1
for i in range(max_iterations):
    # find the nearest neighbors between the current source and destination points.
    distances, indices = nearest_neighbors(src[:m, :].T, dst[:m, :].T)

    # match each point of source-set to closest point of destination set
    matched_src_pts = src[:m, :]
    matched_dst_pts = dst[:m, indices]
    
    # compute angle between 2 matched vertex's normals
    matched_src_pts_normals = A_normals.copy()
    matched_dst_pts_normals = B_normals[indices, :] 
    num_pts = src.shape[1]
    angels = np.zeros(num_pts)
    for k in range(num_pts):
        v1 = matched_src_pts_normals[k, :]
        v2 = matched_dst_pts_normals[k, :]
        cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angels[k] = np.arccos(cos_sim) / np.pi*180
    
    angle_thres = 20 
    mask_pts = (angels < angle_thres)
    matched_src_pts = matched_src_pts[:, mask_pts]
    matched_dst_pts = matched_dst_pts[:, mask_pts]

    # compute the transformation 
    T, _, _ = best_fit_transform(matched_src_pts, matched_dst_pts) 

    # update current src
    src = np.dot(T, src)

    



