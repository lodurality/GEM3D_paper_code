from collections import defaultdict
import networkx as nx
import numpy as np
import trimesh
from skimage.morphology import skeletonize_3d
from scipy.spatial.distance import cdist
from scipy.spatial import Delaunay
from itertools import combinations
from copy import deepcopy
import open3d as o3d
import torch
from scipy.spatial import KDTree
#from rdp import rdp

from .spatial import fps, get_sdfs
from .visual import get_o3d_mesh
import subprocess

import sys
import os
from utils.run_utils import time_limit

def get_skeleton_points(mesh, resolution=300):
    orig_vox = mesh.voxelized(pitch= 1 /resolution).fill()
    skel = skeletonize_3d(orig_vox.matrix)
    skel_tst = trimesh.voxel.VoxelGrid( skel > 0)
    skel_pts = orig_vox.indices_to_points(skel_tst.points.astype(int))

    return skel_pts, orig_vox

def get_skeleton_edges(skel_pts, mult=1.03):
    dist = cdist(skel_pts, skel_pts)
    dist[np.diag_indices(len(dist))] = 1

    edge_mask = (dist <= mult *dist.min(axis=1).max())
    edges = np.where(edge_mask)
    edge_tuples = [(edges[0][i] ,edges[1][i]) for i in range(len(edges[0]))]

    return edge_tuples


def mult_filtering(edges, points, mult, prox_obj):
    edge_queries = np.stack([mult.dot(points[item, :]) for item in edges], axis=0)
    edge_dists = prox_obj.signed_distance(edge_queries)
    filtered_edges = [item for i, item in enumerate(edges) if edge_dists[i] > 0]

    return filtered_edges


def mult_filtering_o3d(edges, points, mult, mesh):
    mesh_o3d = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(mesh.vertices),
                                            triangles=o3d.utility.Vector3iVector(mesh.faces))
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_o3d)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh)
    # Compute distance of the query point from the surface

    edge_queries_np = np.stack([mult.dot(points[item, :]) for item in edges], axis=0)
    edge_queries = o3d.core.Tensor(edge_queries_np, dtype=o3d.core.Dtype.Float32)
    edge_dists = scene.compute_signed_distance(edge_queries).numpy()
    filtered_edges = [item for i, item in enumerate(edges) if edge_dists[i] <= 0]
    bad_queries = [item.astype(np.float16) for i, item in enumerate(edge_queries_np) if edge_dists[i] > 0 and edge_dists[i] < 0.02]

    return filtered_edges, bad_queries


def get_skeletons_from_file(in_path, resolution=300, num_subsamples=1, num_points=512, timeout_limit=120,
                            vertex_filter=10000, skel_share=0.25, inside_share=0.25):
    tst = trimesh.load(in_path, force='mesh')
    if vertex_filter is not None:
        print('Filtering mesh fusion artifacts...')
        print('Items before fitering', len(tst.split()))
        if len(tst.split()) > 1:
            print('Doing filtering')
            removed_noise = [item for item in tst.split() if item.vertices.shape[0] > vertex_filter]
            tst = trimesh.util.concatenate(removed_noise)
            print('Items after filtering', len(tst.split()))
        else:
            print('No filtering -- only one submesh.')
    surface_sample = tst.sample(100000)

    try:
        with time_limit(timeout_limit):
            points, G, orig_vox = get_clean_skeleton_graph(tst, resolution=resolution)
    except:
        print('High resolution skeleton timed out, trying skeleton with smaller resolution')
        points, G, orig_vox = get_clean_skeleton_graph(tst, resolution=128)

    subsamples_256, subsamples_512, subsamples_2048 = [], [], []
    surface_subsamples_256, surface_subsamples_512, surface_subsamples_2048, = [], [], []
    tet_graph_points, tet_graph_edges, tet_graph_queries = [], [], []
    data_dict = {}

    con_points_inds, end_points_inds = get_connectivity_and_end_nodes(G)
    sel_inds = con_points_inds  # + end_points_inds
    sel_inds_bool = np.full(len(points), True)
    sel_inds_bool[sel_inds] = False

    for i in range(num_subsamples):
        subsamples_256 += [fps(points, K=256)[0]]
        subsamples_512 += [fps(points, K=512)[0]]
        subsamples_2048 += [fps(points, K=2048)[0]]
        surface_subsamples_256 += [fps(surface_sample, K=256)[0]]
        surface_subsamples_512 += [fps(surface_sample, K=512)[0]]
        surface_subsamples_2048 += [fps(surface_sample, K=2048)[0]]

        fps_graph_pts, fps_graph_inds = fps(points, int(num_points*skel_share) - len(sel_inds))
        comb_inds = list(fps_graph_inds + sel_inds)
        comb_points = points[comb_inds, :]

        sample_pts = orig_vox.points[
            np.random.randint(0, high=len(orig_vox.points), size=100000)]  # trimesh.sample.volume_mesh(tst, 1000000)
        # sample_pts = np.vstack((sample_pts, comb_points))
        inside_points, inside_fps_inds = fps(sample_pts, int(inside_share*num_points))
        surface_points, surface_fps_inds = fps(surface_sample, num_points - len(inside_points) - len(comb_points))
        print(surface_points.shape, inside_points.shape, comb_points.shape)
        full_points = np.vstack((comb_points, inside_points, surface_points))
        print(full_points.shape)
        filtered_edges, bad_queries = get_tet_graph(full_points, tst)
        filtered_edges = np.array(filtered_edges)
        bad_queries = np.stack(bad_queries, axis=0)
        bad_queries = bad_queries[np.random.randint(0, high=len(bad_queries), size=10000)]

        tet_graph_points += [full_points]
        tet_graph_edges += [filtered_edges]
        tet_graph_queries += [bad_queries]


    data_dict['skeleton_full'] = points
    data_dict['skeleton_256'] = np.stack(subsamples_256, axis=0).astype(np.float16)
    data_dict['skeleton_512'] = np.stack(subsamples_512, axis=0).astype(np.float16)
    data_dict['skeleton_2048'] = np.stack(subsamples_2048, axis=0).astype(np.float16)
    data_dict['surface_256'] = np.stack(surface_subsamples_256, axis=0).astype(np.float16)
    data_dict['surface_512'] = np.stack(surface_subsamples_512, axis=0).astype(np.float16)
    data_dict['surface_2048'] = np.stack(surface_subsamples_2048, axis=0).astype(np.float16)
    data_dict[f'tet_graph_points'] = np.stack(tet_graph_points, axis=0).astype(np.float16)
    data_dict[f'tet_graph_edges'] = tet_graph_edges
    data_dict[f'tet_graph_outside_queries'] = tet_graph_queries
    print('Finished processing skeleton!')
    return data_dict


def get_mcfskel_file(in_path, mcfskel_path, config_path, skel_out, skel_corr_out, mesoskel_out=None):

    bash_command = f'{mcfskel_path} --input_files {config_path} {in_path} {skel_out} {skel_corr_out}'

    if mesoskel_out is not None:
        bash_command = bash_command + f' --meso_skeleton {mesoskel_out}'

    print('\nRunning "%s"\n'%(bash_command))

    process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

    return output, error


def get_surface_init(mesh, surface_sample, num_candidates=500, sampling_scale=0.0005):
    gt_mesh = deepcopy(mesh)
    mesh_legacy = o3d.t.geometry.TriangleMesh.from_legacy(get_o3d_mesh(gt_mesh))
    scene = o3d.t.geometry.RaycastingScene()
    mesh_id = scene.add_triangles(mesh_legacy)

    skel = surface_sample.copy()[:, None, :]
    skel = skel + sampling_scale * np.random.randn(len(skel), num_candidates, 3)
    flat_skel = skel.reshape(-1, 3)
    flat_sdfs = get_sdfs(flat_skel, mesh)
    sdfs = flat_sdfs.reshape(*skel.shape[:2])
    min_inds = np.argmin(sdfs, axis=1)
    surface_init = skel[range(len(skel)), min_inds]

    return surface_init


def get_init_skel(mesh, dir_angle=np.pi / 6, num_samples=50, init_type='mesh_verts', num_init_samples=50000):

    if init_type == 'mesh_verts':
        vertex_normals = mesh.vertex_normals
        verts = mesh.vertices
    elif init_type == 'random':
        verts, vert_faces = mesh.sample(num_init_samples, return_index=True)
        vertex_normals = mesh.face_normals[vert_faces]
    elif init_type == 'combined':
        assert num_init_samples > len(mesh.vertices), "num_init_samples needs to be > len(mesh.vertices)"
        residual_num = num_init_samples - len(mesh.vertices)
        verts, vert_faces = mesh.sample(residual_num, return_index=True)
        vertex_normals = mesh.face_normals[vert_faces]
        verts = np.concatenate((mesh.vertices, verts), axis=0)
        vertex_normals = np.concatenate((mesh.vertex_normals, vertex_normals), axis=0)

    theta = np.cos(dir_angle)
    z = torch.rand(num_samples, 1) * (1 - theta) + theta
    angle = 2 * torch.pi * torch.rand(num_samples, 1)
    pts = torch.hstack((torch.sqrt(1 - z ** 2) * torch.cos(angle), torch.sqrt(1 - z ** 2) * torch.sin(angle), z))

    source_vec = torch.FloatTensor([0, 0, 1]).reshape(1, -1)
    target_vec = torch.FloatTensor(np.array(-vertex_normals))

    vs = torch.cross(source_vec, target_vec)
    ss = torch.linalg.norm(vs, axis=-1)
    cc = torch.sum(target_vec * source_vec, axis=-1)
    kmat = torch.zeros(len(vs), 3, 3)
    kmat[:, 0, 1] = -vs[:, 2]
    kmat[:, 0, 2] = vs[:, 1]
    kmat[:, 1, 0] = vs[:, 2]
    kmat[:, 1, 2] = -vs[:, 0]
    kmat[:, 2, 0] = -vs[:, 1]
    kmat[:, 2, 1] = vs[:, 0]
    rrs = torch.eye(3)[None, :, :] + kmat + kmat.bmm(kmat) * ((1 - cc[:, None, None]) / (ss[:, None, None] ** 2 + 1e-8))

    all_rotated_dirs = torch.bmm(rrs, pts[None, :, :].repeat(len(rrs), 1, 1).permute(0, 2, 1))
    all_rotated_dirs = all_rotated_dirs.permute(0, 2, 1)

    mesh_legacy = o3d.t.geometry.TriangleMesh.from_legacy(get_o3d_mesh(mesh))
    scene = o3d.t.geometry.RaycastingScene()
    mesh_id = scene.add_triangles(mesh_legacy)

    surface_init = get_surface_init(mesh, verts)
    tiled_verts = np.tile(surface_init[:, np.newaxis, :], (1, num_samples, 1))
    rays = np.concatenate((tiled_verts, all_rotated_dirs), axis=-1)
    rays = rays.reshape(-1, 6)
    rays = o3d.core.Tensor(list(rays),
                           dtype=o3d.core.Dtype.Float32)

    ans = scene.cast_rays(rays)
    lsd_ray_dists = ans['t_hit'].numpy().reshape((len(verts), num_samples, 1))

    pad_val = np.median(lsd_ray_dists[lsd_ray_dists != np.inf])
    lsd_ray_dists[lsd_ray_dists == np.inf] = pad_val
    lsds = np.nanmean(lsd_ray_dists, axis=1)
    init_skel = verts - 0.5 * lsds * vertex_normals

    return init_skel, lsds, verts


def get_min_sdf_skel(mesh, num_iter=50, lsds_mult=0.6, init_type='mesh_verts', num_init_samples=50000):
    init_skel, lsds, verts = get_init_skel(mesh, init_type=init_type, num_init_samples=num_init_samples)
    skel = init_skel.copy()
    alpha = 0.1
    skel_points = torch.FloatTensor(skel.copy()).type(torch.float64)
    inds_to_check = np.array(range(len(skel_points)))

    for i in range(num_iter):
        if i % 10 == 0:
            print(i)
        cur_skel = skel_points[inds_to_check]
        grad_input = cur_skel.clone().detach().requires_grad_(True)
        k_neighb = 20
        query_pts = skel
        neighbs = cur_skel[:, None, :] + torch.clip(0.003 * torch.randn(len(cur_skel), k_neighb, 3), max=0.01)
        nn_sdfs = get_sdfs(neighbs.reshape(-1, 3).detach().cpu().numpy(), mesh)
        nn_sdfs = torch.FloatTensor(nn_sdfs.reshape(-1, k_neighb))

        dists = grad_input[:, None, :] - neighbs
        dists = torch.linalg.norm(dists, axis=-1)
        weights = torch.exp(-dists ** 2 / 0.002)  # OLD 0.002 IMPORTANT
        weights = weights / weights.sum(axis=1, keepdims=True)
        weighted_sdfs = nn_sdfs * weights
        weighted_sdfs = weighted_sdfs.sum(axis=1)
        loss = weighted_sdfs.sum()
        loss.backward()

        fin_grad = grad_input.grad
        del grad_input
        cur_skel_new = cur_skel - alpha * fin_grad  # alpha*mean_grads
        norms = np.linalg.norm(cur_skel_new - cur_skel, axis=1)

        disps = skel_points.numpy() - verts
        disps = np.linalg.norm(disps, axis=1)
        if i < 10:
            lsds_mask = disps < 1e3
        else:
            lsds_mask = disps < lsds_mult * lsds[:, 0]
        #print(lsds_mask.sum())
        check_mask = lsds_mask[inds_to_check]
        skel_points[inds_to_check[check_mask]] = cur_skel_new[check_mask]
        inds_to_check = inds_to_check[check_mask]
        if len(inds_to_check) == 0:
            break

    return skel_points


def get_full_min_sdf_skeleton(mesh, num_iter=50, lsds_mult=0.6, num_init_samples=50000, init_type='mesh_verts'):
    skel_points = get_min_sdf_skel(mesh, num_iter=num_iter, lsds_mult=lsds_mult,
                                   num_init_samples=num_init_samples, init_type=init_type)
    print('Init skel shape', skel_points.shape)
    reduced_pts = torch.unique(((1 / 2) * skel_points).round(decimals=3), dim=0) * 2
    tree_new = KDTree(reduced_pts)
    _, inds = tree_new.query(skel_points)
    vert_mapper = dict(zip(range(len(skel_points)), inds))
    if init_type == 'mesh_verts':
        reduced_edges = [tuple(sorted((vert_mapper[i], vert_mapper[j]))) for i, j in mesh.edges]
        reduced_edges = [item for item in reduced_edges if item[0] != item[1]]
        reduced_edges = np.unique(np.sort(reduced_edges, axis=1), axis=0)
    else:
        print('No edges since init_type is not mesh_verts')
        reduced_edges = np.array([(None,None)])
    print('Final skel shape')
    print(len(reduced_pts), len(reduced_edges))

    return reduced_pts, reduced_edges


def mult_filtering_o3d_with_list(edges, points, mult_list, mesh):
    mesh_o3d = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(mesh.vertices),
                                         triangles=o3d.utility.Vector3iVector(mesh.faces))
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_o3d)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh)
    # Compute distance of the query point from the surface
    fin_mask = np.ones(len(edges)).astype(bool)
    for mult in mult_list:
        cur_mult = np.array([mult, 1 - mult])
        edge_queries_np = np.stack([cur_mult.dot(points[item, :]) for item in edges], axis=0)
        edge_queries = o3d.core.Tensor(edge_queries_np, dtype=o3d.core.Dtype.Float32)
        edge_dists = scene.compute_signed_distance(edge_queries).numpy()
        cur_mask = edge_dists < 0
        fin_mask = fin_mask & cur_mask

    return fin_mask