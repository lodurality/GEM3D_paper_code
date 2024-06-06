import numpy as np
from scipy.spatial.distance import cdist
import open3d as o3d
from copy import deepcopy
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from torch_cluster import fps as fps_cluster
from torch_cluster import knn
from .visual import get_o3d_mesh
from collections import defaultdict
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
import trimesh

from itertools import combinations
import networkx as nx


def get_triangles(G, node):
    neighbors1 = set(G.neighbors(node))
    triangles = []
    """
    Fill in the rest of the code below.
    """
    for nbr1, nbr2 in combinations(neighbors1, 2):
        if G.has_edge(nbr1, nbr2):
            triangles += [sorted([node, nbr1, nbr2])]
    return triangles


def get_all_triangles(G):
    all_triangles = []
    for cur_node in list(sorted(G.nodes)):
        all_triangles += get_triangles(G, cur_node)

    return all_triangles

def calc_distances(p0, points):
    return ((p0 - points)**2).sum(axis=1)


def calc_distances_another(p0, points):
    return cdist(p0, points).sum(axis=0)


def simple_fps(cloud, N=256, return_idx=False):
    N1, D1 = cloud.shape
    idx = fps_cluster(cloud[:,:3], None, ratio=N / N1)
    if len(idx) != N:
        idx = idx[:N]
    sampled_cloud = cloud[idx]

    if not return_idx:
        return sampled_cloud
    else:
        return sampled_cloud, idx


def fps(pts, K, dim=3):
    farthest_pts = np.zeros((K, dim))
    rand_ind = np.random.randint(len(pts))
    farthest_pts[0] = pts[rand_ind]
    fps_inds = [rand_ind]
    distances = calc_distances_another(farthest_pts[[0]], pts)
    for i in range(1, K):
        fps_ind = np.argmax(distances)
        farthest_pts[i] = pts[fps_ind]
        fps_inds += [fps_ind]

        distances = np.minimum(distances, calc_distances_another(farthest_pts[[i]], pts))
    return farthest_pts, fps_inds


def fps_enhance(pts, K, existing_pts):
    N = K + len(existing_pts)
    farthest_pts = np.zeros((N, 3))
    farthest_pts[:len(existing_pts)] = existing_pts
    fps_inds = [rand_ind]
    distances = calc_distances(existing_pts, pts)
    for i in range(K+2, N):
        fps_ind = np.argmax(distances)
        farthest_pts[i] = pts[fps_ind]
        fps_inds += [fps_ind]

        distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))
    return farthest_pts, fps_inds


def fps_geodesic(pts, K, dist_matrix):
    farthest_pts = np.zeros((K, 3))
    rand_ind = np.random.randint(len(pts))
    farthest_pts[0] = pts[rand_ind]
    fps_inds = [rand_ind]
    distances = calc_distances(farthest_pts[0], pts)
    for i in range(1, K):
        fps_ind = np.argmax(distances)
        farthest_pts[i] = pts[fps_ind]
        fps_inds += [fps_ind]
        
        distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))
    return farthest_pts, fps_inds


def get_sdfs(query_sample, trimesh_mesh):
    tst_o3d = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(trimesh_mesh.vertices),
                                            triangles=o3d.utility.Vector3iVector(trimesh_mesh.faces))

    tst_o3d.compute_vertex_normals()

    sample_o3d = o3d.core.Tensor(query_sample, dtype=o3d.core.Dtype.Float32)
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(tst_o3d)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh)
    # Compute distance of the query point from the surface
    signed_distance = scene.compute_signed_distance(sample_o3d)

    return signed_distance.numpy()


def compute_all_neighbs(ref_points, K):
    dists = torch.cdist(ref_points, ref_points)
    inds = dists.argsort(axis=2)[:, :, :K]

    all_items = []
    for i in range(len(inds)):
        neigbs = torch.index_select(ref_points[i], dim=0, index=inds[i].flatten()).reshape(len(ref_points[i]), K, 3)
        all_items += [neigbs]

    all_neighbs = torch.stack(all_items, axis=0)

    return all_neighbs


def compute_all_neighbs_precomp_inds(ref_points, precomp_inds, K):
    inds = deepcopy(precomp_inds)

    all_items = []
    for i in range(len(inds)):
        neigbs = torch.index_select(ref_points[i], dim=0, index=inds[i].flatten()).reshape(len(ref_points[i]), K, 3)
        all_items += [neigbs]

    all_neighbs = torch.stack(all_items, axis=0)

    return all_neighbs


def get_dijkstra(skel_verts, skel_edges, limit=0.2):
    N = len(skel_verts)
    new_skel_edges = torch.cat((skel_edges, skel_edges[:, [1, 0]]), axis=0)
    graph_data = torch.norm((skel_verts[new_skel_edges] * np.array([1, -1])[np.newaxis, :, np.newaxis]).sum(axis=1),
                            dim=-1)

    graph = csr_matrix(
        (graph_data.cpu().numpy(), (new_skel_edges[:, 0].cpu().numpy(), new_skel_edges[:, 1].cpu().numpy())),
        shape=(N, N))
    skel_geo_dists = dijkstra(graph, directed=False, limit=limit)

    return skel_geo_dists.astype(np.float16)


def fps_from_cloud(cloud, N=256):

    if len(cloud.shape) == 2:
        cloud = cloud.unsqueeze(0)

    B1, N1, D1 = cloud.shape
    cloud_flat = cloud.view(B1 * N1, D1)
    pos_cloud = cloud_flat

    batch1 = torch.arange(B1).to(cloud.device)
    batch1 = torch.repeat_interleave(batch1, N1)
    idx = fps_cluster(pos_cloud, batch1, ratio=N / N1)  # 0.0625
    #idx = idx[:N]
    #print(idx.shape)
    idx = idx.reshape(len(cloud), N)

    sampled_cloud = pos_cloud[idx]
    batch2 = torch.arange(len(idx)).to(cloud.device)
    batch2 = torch.repeat_interleave(batch2, idx.shape[1]).reshape(len(idx), -1)
    #print(idx.shape, batch2.shape)

    return sampled_cloud, idx - N1 * batch2


def get_topk_nns(query, reference, k):
    dists = torch.cdist(query, reference)
    inds = (-dists).topk(k=k, axis=2)[1]

    ret = reference[torch.arange(reference.size(0)).unsqueeze(1).unsqueeze(2), inds]

    return ret, inds


def get_topk_nns_dilated(query, reference, k, dilation=1):
    dists = torch.cdist(query, reference)
    dists_sorted, inds = torch.sort(dists, axis=2)
    presel_inds = list(range(dilation * k))[::dilation]
    sel_inds = inds[:, :, presel_inds]

    ret = reference[torch.arange(reference.size(0)).unsqueeze(1).unsqueeze(2), sel_inds]

    return ret, sel_inds, dists_sorted[:, :, presel_inds]


def get_flat_queries_and_centers(queries, centers, k=5):
    nns, nn_inds, nn_dists = get_topk_nns_dilated(queries, centers, k=k, dilation=1)
    # print('check')
    # print(nns.shape, vol_nn_inds.shape, nn_dists.shape)
    disps = queries.unsqueeze(2) - nns
    scales = torch.linalg.norm(disps, axis=-1, keepdim=True)
    unit_queries = disps / (scales+1e-8)
    gt_scales = scales.reshape(len(scales), -1)

    flat_inds = nn_inds.reshape(len(nn_inds), -1)
    flat_queries = unit_queries.reshape(len(unit_queries), -1, 3)

    return flat_queries, flat_inds, gt_scales, nn_dists, nn_inds, nns, unit_queries


def get_skel_patch_sizes(gt_mesh, skel, mesh_sample=400000, skel_nn=3):
    mesh_legacy = o3d.t.geometry.TriangleMesh.from_legacy(get_o3d_mesh(gt_mesh))
    scene = o3d.t.geometry.RaycastingScene()
    mesh_id = scene.add_triangles(mesh_legacy)

    surface_cloud = gt_mesh.sample(mesh_sample)

    tree_skel = KDTree(skel)
    skel_dists, skel_nn_inds = tree_skel.query(surface_cloud, k=skel_nn)
    skel_nn_inds = skel_nn_inds.reshape(-1, skel_nn)
    skel_pts = skel[skel_nn_inds.reshape(-1)].reshape(skel_nn_inds.shape + (3,))
    nn_directions = (surface_cloud[:, np.newaxis, :] - skel_pts)
    nn_directions /= np.linalg.norm(nn_directions, axis=-1, keepdims=True)
    nn_directions_flat = nn_directions.reshape(-1, 3)
    skel_pts_flat = skel_pts.reshape(-1, 3)

    rays_nn = np.concatenate((skel_pts_flat, nn_directions_flat), axis=1)
    rays_nn = o3d.core.Tensor(list(rays_nn),
                              dtype=o3d.core.Dtype.Float32)
    ans_nn = scene.cast_rays(rays_nn)
    ray_dists_nn = ans_nn['t_hit'].numpy()
    # print(ray_dists_nn.max())

    result_dict = defaultdict(list)
    for key, value in zip(skel_nn_inds.reshape(-1), ray_dists_nn):
        result_dict[key].append(value)

    # print(result_dict)

    all_maxes = []
    for key in sorted(list(result_dict.keys())):
        # print(key, np.max(result_dict[key]))
        all_maxes += [np.quantile(result_dict[key], q=0.99)]

    return sorted(list(result_dict.keys())), all_maxes


def get_patches_from_skel_and_rays(skel, directions, ray_dists):
    print(skel.shape, directions.shape, ray_dists.shape)
    assert len(skel) == ray_dists.shape[0]
    assert len(directions) == ray_dists.shape[1]
    directions_tile = np.tile(directions[np.newaxis, :, :], (len(skel), 1, 1))
    skel_points = np.tile(skel[:, np.newaxis, :], (1, directions_tile.shape[1], 1))

    patches = skel_points + ray_dists * directions_tile

    return patches


def fibonacci_sphere(samples=500):

    points = []
    phi = np.pi * (np.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(samples):
        y = 1. - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append((x, y, z))

    return np.array(points)


def get_ray_if(skel, gt_mesh, clamp_mode='adaptive',
               skel_nn=3, mesh_sample=400000,
               clamp_val=0.1,
               sampling_mode='icosphere',
               num_sphere_points=500,
               rotate_dirs=False,
               icosphere_subdivisions=3):
    mesh_legacy = o3d.t.geometry.TriangleMesh.from_legacy(get_o3d_mesh(gt_mesh))
    scene = o3d.t.geometry.RaycastingScene()
    mesh_id = scene.add_triangles(mesh_legacy)

    if sampling_mode == 'icosphere':
        print('Using icosphere sphere')
        sphere = trimesh.creation.icosphere(subdivisions=icosphere_subdivisions)
        directions_orig = sphere.vertices
    elif sampling_mode == 'fibonacci':
        print('Using fibonacci sphere')
        directions_orig = fibonacci_sphere(samples=num_sphere_points)
    elif sampling_mode == 'random':
        print('Using random sampling')
        sample = np.random.randn(num_sphere_points, 3)
        directions_orig = sample / np.linalg.norm(sample, axis=-1, keepdims=True)
    else:
        raise ValueError('Wrong sampling method. Must be: icosphere/fibonacci/random')


    if rotate_dirs:
        print('Doing random direction rotation')
        rot = Rotation.random().as_matrix()
        directions_orig = directions_orig.dot(rot)

    directions = np.tile(directions_orig[np.newaxis, :, :], (len(skel), 1, 1))
    skel_points = skel
    skel_points = np.tile(skel[:, np.newaxis, :], (1, len(directions_orig), 1))

    rays = np.concatenate((skel_points, directions), axis=-1)
    rays = rays.reshape(-1, 6)
    rays = o3d.core.Tensor(list(rays),
                           dtype=o3d.core.Dtype.Float32)

    ans = scene.cast_rays(rays)
    ray_dists_orig = ans['t_hit'].numpy().reshape((len(skel), len(directions_orig), 1))
    print(ray_dists_orig.max())

    if clamp_mode == 'adaptive':

        skel_inds, patch_sizes = get_skel_patch_sizes(gt_mesh, skel, skel_nn=skel_nn)
        clamp_vals = np.array(patch_sizes)[:, np.newaxis, np.newaxis]
        clamp_vals = np.repeat(clamp_vals, len(directions_orig), axis=1)
        print(ray_dists_orig.shape, clamp_vals.shape)
        ray_dists = np.minimum(ray_dists_orig[skel_inds], clamp_vals)
        patches = skel_points[skel_inds] + ray_dists * directions[skel_inds]
    elif clamp_mode == 'fixed':
        skel_inds = np.array(range(len(skel)))
        ray_dists = np.clip(ray_dists_orig, a_min=0, a_max=clamp_val)
        patches = skel_points + ray_dists * directions  # + np.array([[[0,1,0]]])
    else:
        raise ValueError("Wrong clamp_mode value: must be either adaptive or fixed")

    patch_pts_sdfs = get_sdfs(patches.reshape(-1, 3), gt_mesh)
    patches_surf_mask = np.abs(patch_pts_sdfs) < 1e-5
    patches_surf_mask = patches_surf_mask.reshape(patches.shape[:2] + (1,))

    return patches, ray_dists, directions_orig, skel[skel_inds], patches_surf_mask


def get_reg_pairs(mesh, skel, initial_sample=500000):
    surface_cloud = mesh.sample(initial_sample)

    tree_skel = KDTree(skel)
    skel_dists, skel_nn_inds = tree_skel.query(surface_cloud, k=2)
    sel_points = skel[skel_nn_inds.reshape(-1)].reshape(skel_nn_inds.shape + (3,))
    check_dirs = surface_cloud[:, None, :] - sel_points
    approx_dists = np.linalg.norm(check_dirs, axis=-1, keepdims=True)
    check_dirs /= approx_dists

    mesh_legacy = o3d.t.geometry.TriangleMesh.from_legacy(get_o3d_mesh(mesh))
    scene = o3d.t.geometry.RaycastingScene()
    mesh_id = scene.add_triangles(mesh_legacy)
    rays_nn = np.concatenate((sel_points.reshape(-1, 3), check_dirs.reshape(-1, 3)), axis=1)
    rays_nn = o3d.core.Tensor(list(rays_nn),
                              dtype=o3d.core.Dtype.Float32)
    ans_nn = scene.cast_rays(rays_nn)
    ray_dists_nn = ans_nn['t_hit'].numpy()
    ray_dists_nn = ray_dists_nn.reshape(approx_dists.shape)
    surf_points = sel_points + ray_dists_nn * check_dirs
    fin_inds = (np.linalg.norm((surf_points * np.array([1, -1])[None, :, None]).sum(axis=1), axis=1) < 1e-7)

    return ray_dists_nn[fin_inds], check_dirs[fin_inds], skel_nn_inds[fin_inds]