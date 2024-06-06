import numpy as np
import trimesh
import os
import torch
import open3d as o3d
from skimage import measure
import sys
import json
from copy import deepcopy
from .spatial import fps_from_cloud, fibonacci_sphere
from scipy.spatial import KDTree
from .visual import get_o3d_mesh


def get_disps(centers, final_latents, model, num_directions=500, disp_bs=50000,
              dir_type='fibonacci',
              num_subdivisions=3, return_dir_faces=False):
    verts = centers[0].detach().cpu()
    latent_device = final_latents.device

    if dir_type == 'fibonacci':
        directions = fibonacci_sphere(num_directions)
    elif dir_type == 'icosphere':
        icosphere = trimesh.creation.icosphere(subdivisions=num_subdivisions)
        directions = icosphere.vertices
        dir_faces = icosphere.faces

    directions = torch.FloatTensor(directions)

    full_queries = directions[None, :, :].repeat(len(verts), 1, 1)
    flat_queries = full_queries.reshape(-1, 3).unsqueeze(0)
    flat_inds = torch.repeat_interleave(torch.arange(len(verts)), len(directions)).unsqueeze(0)

    chunks = [(flat_queries[:, i:i + disp_bs, :], flat_inds[:, i:i + disp_bs]) for i in
              range(0, flat_queries.shape[1], disp_bs)]
    all_disps = []
    for i, (cur_queries, cur_inds) in enumerate(chunks):
        with torch.no_grad():
            cur_disps = model.decoder.decode_queries(final_latents,
                                                     cur_queries.to(latent_device),
                                                     cur_inds.to(latent_device))
        all_disps += [cur_disps.detach().cpu()]

    disps = torch.cat(all_disps, axis=1).reshape(full_queries.shape[0], full_queries.shape[1], 1)
    disps = torch.sigmoid(disps).to(latent_device)

    verts = verts.to(latent_device)
    directions = directions.to(latent_device)

    full_disp_pts = verts[:, None, :] + disps * directions[None, :, :]

    if return_dir_faces:
        return disps, (directions, dir_faces), full_disp_pts
    else:
        return disps, directions, full_disp_pts


def get_if_values_on_queries_spheres(Q, latents, centers, model,
                                     bs=10000, device='cuda',
                                     ball_nn=2,
                                     margin=0.02,
                                     return_debug_info=False,
                                     num_disp_directions=1000):
    Q = torch.FloatTensor(Q).unsqueeze(0)  # .to(latents.device)
    chunks = [Q[:, i:i + bs, :] for i in range(0, Q.shape[1], bs)]

    with torch.no_grad():
        final_latents = model.decoder.get_final_latents(latents, centers).detach()

    disps, directions, full_disp_pts = get_disps(centers, final_latents, model,
                                                 num_directions=num_disp_directions)
    skel_radii = torch.quantile(disps, 0.8, axis=1).unsqueeze(0).to(device)
    all_sdf_estimates = []
    all_sel_inds = []
    hist_counts = []

    for i, chunk in enumerate(chunks):
        # print(chunk.shape)
        queries = chunk.to(device)
        # print(queries.shape)
        query_vecs = queries[:, :, None, :] - centers[:, :, :].to(device)
        #print(query_vecs.shape, centers.shape, queries.shape)
        query_disps = torch.linalg.norm(query_vecs, axis=-1, keepdim=True)
        query_dirs = query_vecs / query_disps
        cur_sdf_estimates = query_disps.squeeze(-1) - skel_radii[:, None, :, 0]
        cur_sdf_estimates = cur_sdf_estimates.squeeze(0)

        cur_sort_inds = torch.argsort(cur_sdf_estimates, axis=1)
        cur_sel_inds = cur_sort_inds[:, :ball_nn]
        cur_sel_estimates = cur_sdf_estimates[torch.arange(len(cur_sort_inds)).unsqueeze(1), cur_sel_inds]
        all_sel_inds += [cur_sel_inds.to('cpu')]
        all_sdf_estimates += [cur_sel_estimates.to('cpu')]
        hist_counts += [torch.sum((cur_sdf_estimates < margin) * 1, dim=-1).cpu()]

    sel_estimates = torch.cat(all_sdf_estimates, axis=0).to('cpu')
    sel_inds = torch.cat(all_sel_inds, axis=0).to('cpu')
    hist_counts = torch.cat(hist_counts, axis=0)

    close_query_ids, close_sel_skel_ids = torch.where(sel_estimates <= margin)
    far_query_ids, far_sel_skel_ids = torch.where(torch.abs(sel_estimates) > margin)
    close_skel_ids = sel_inds[close_query_ids, close_sel_skel_ids]
    far_skel_ids = sel_inds[far_query_ids, far_sel_skel_ids]

    #print(full_queries.shape, close_query_ids.shape)
    close_query_pts = Q[:, close_query_ids].to(device)
    close_query_skel_pts = centers[:, close_skel_ids].to(device)

    flat_close_queries = close_query_pts - close_query_skel_pts
    fin_close_dists = torch.linalg.norm(flat_close_queries, axis=-1, keepdim=True)
    flat_close_queries = flat_close_queries / fin_close_dists
    flat_close_queries = flat_close_queries.to(device)
    #print(close_skel_ids)
    flat_close_inds = close_skel_ids.unsqueeze(0)

    dir_chunks = [flat_close_queries[:, i:i + bs, :] for i in range(0, flat_close_queries.shape[1], bs)]
    ind_chunks = [flat_close_inds[:, i:i + bs] for i in range(0, flat_close_inds.shape[1], bs)]
    assert len(dir_chunks) == len(ind_chunks)

    all_preds = []
    for cur_dirs, cur_inds in zip(dir_chunks, ind_chunks):
        with torch.no_grad():
            cur_preds = model.decoder.decode_queries(final_latents, cur_dirs, cur_inds)
            cur_preds = torch.sigmoid(cur_preds).to('cpu')
            all_preds += [cur_preds[0]]

    # print(full_if.shape)
    # IF = full_if.min(axis=1)[0]
    # print(IF.shape, full_if.shape)
    fin_preds = torch.cat(all_preds)
    precise_sdfs = fin_close_dists[0, :, 0].to('cpu') - fin_preds
    fin_sdfs = sel_estimates.clone().to('cpu')
    fin_sdfs[close_query_ids, close_sel_skel_ids] = precise_sdfs
    sdf_vals = torch.min(fin_sdfs, axis=-1)[0].numpy()

    IF = sdf_vals

    if return_debug_info:
        return IF, skel_radii, hist_counts, fin_sdfs, sel_inds.cpu()
    else:
        return IF


def get_mesh_from_latent_combination(latents, centers, model, grid_size,
                                     padding=5, bs=10000,
                                     level=0.5, color=[1, 0.706, 0],
                                     shift=(0, 0, 0),
                                     shape_scale=1,
                                     max_dimensions=np.array([1, 1, 1]),
                                     min_dimensions=np.array([-1, -1, -1]),
                                     dir_consistency=False,
                                     num_disp_directions=1000,
                                     use_spherical_reconstruction=False,
                                     ball_nn=5,
                                     skel_nn=5,
                                     ball_margin=0.02,
                                     device='cuda'):
    N = grid_size
    bounding_box_dimensions = max_dimensions - min_dimensions  # compute the bounding box dimensions of the point cloud
    grid_spacing = max(bounding_box_dimensions) / N  # each cell in the grid will have the same size
    X, Y, Z = np.meshgrid(
        list(np.arange(min_dimensions[0] - grid_spacing * padding, max_dimensions[0] + grid_spacing * padding,
                       grid_spacing)),
        list(np.arange(min_dimensions[1] - grid_spacing * padding, max_dimensions[1] + grid_spacing * padding,
                       grid_spacing)),
        list(np.arange(min_dimensions[2] - grid_spacing * padding, max_dimensions[2] + grid_spacing * padding,
                       grid_spacing)))

    Q = np.array([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)]).transpose()

    with torch.no_grad():
        final_latents = model.decoder.get_final_latents(latents, centers).detach()

    if use_spherical_reconstruction:
        print('USING SPHERICAL RECON FOR IF')
        print('NUM DISP DIRECTIONS', num_disp_directions)
        IF = get_if_values_on_queries_spheres(Q, latents, centers, model,
                                         bs=bs, device=device,
                                         ball_nn=ball_nn,
                                         margin=ball_margin,
                                         num_disp_directions=num_disp_directions)

    else:
        print('USING EUCLIDEAN NN FOR RECONSTRUCTION: SUBOPTIMAL!')
        IF = get_if_values_on_queries(Q, final_latents, centers, model,
                                      bs=bs,
                                      disp_cloud=None,
                                      disp_inds=None,
                                      dir_consistency=dir_consistency,
                                      skel_nn=skel_nn,
                                      device=device)
    # print(IF.shape)
    IF = IF.reshape(X.shape).transpose(1, 0, 2)

    verts, simplices, verts_normals, val = measure.marching_cubes(IF, level=level,
                                                                  spacing=[1, 1, 1])

    recon_mesh = trimesh.Trimesh(vertices=verts, faces=simplices)
    # recon_mesh.vertices -= recon_mesh.vertices.mean(axis=0)

    normalize_const = len(X)
    #### proper scaling and centering
    recon_mesh.vertices -= (normalize_const) / 2
    recon_mesh.vertices /= (normalize_const - 2 * padding) / shape_scale
    recon_mesh.vertices += np.array(shift)

    o3d_recon = get_o3d_mesh(recon_mesh, color=color)

    return o3d_recon, recon_mesh, IF


def get_if_values_on_queries(queries, final_latents, final_centers, model,
                             bs=10000,
                             dir_consistency=False,
                             dir_thres=0.86,
                             disp_cloud=None,
                             disp_inds=None,
                             skel_nn=5):
    Q = torch.FloatTensor(queries).unsqueeze(0).to(final_latents.device)
    chunks = [Q[:, i:i + bs, :] for i in range(0, Q.shape[1], bs)]
    if_chunks = []

    for chunk in chunks:

        with torch.no_grad():

            IF_chunk = -model.get_skelray_nn_sdfs_optimized(chunk, final_centers, final_latents,
                                                            skel_nn=skel_nn,
                                                            dir_consistency=dir_consistency,
                                                            cosine_thres=dir_thres,
                                                            disp_cloud=disp_cloud,
                                                            disp_inds=disp_inds)[0]

            IF_chunk = IF_chunk.squeeze(-1)
            IF_chunk = IF_chunk.mean(axis=-1)

        if_chunks += [IF_chunk.detach().cpu()]

    full_if = torch.cat(if_chunks, dim=-1)
    IF = full_if.detach().cpu().numpy()

    return IF
