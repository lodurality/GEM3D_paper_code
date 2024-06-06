import torch
import numpy as np
import sys

sys.path.append('/home/dmpetrov/Repos/shape_generation/')
sys.path.append('/home/dmpetrov_umass_edu/Repos/shape_generation/')

from utils.spatial import  fps_from_cloud

def compute_chamfer(x, y):
    dists = torch.cdist(x, y)

    min_row = dists.min(axis=2)[0]
    min_col = dists.min(axis=1)[0]
    cd = min_row.mean(axis=-1) + min_col.mean(axis=-1)
    #print(cd.shape)

    return cd


def compute_chamfer_and_density(pts_true, pts_pred, k=16):
    dists_inside = torch.cdist(pts_true, pts_true)
    dists_between = torch.cdist(pts_true, pts_pred)

    min_row = dists_between.min(axis=2)[0]
    min_col = dists_between.min(axis=1)[0]
    # print(min_row.shape)
    cd = min_row.mean(axis=-1) + min_col.mean(axis=-1)
    # print(cd.shape)

    #knn_between = torch.sort(dists_between, axis=-1)[0][:, :, :k]
    #knn_inside = torch.sort(dists_inside, axis=-1)[0][:, :, :k]
    #print(knn_between, knn_inside)
    #print("ALTERNATIVE")
    knn_between = torch.topk(dists_between, axis=-1, largest=False, k=k)[0]
    knn_inside = torch.topk(dists_inside, axis=-1, largest=False, k=k)[0]
    #print(knn_between, knn_inside)

    density_loss = torch.abs(knn_inside - knn_between).mean(axis=-1)
    # print(density_loss.shape)
    density_loss = density_loss.mean(axis=-1)
    # print(density_loss.shape)

    return cd, density_loss


def compute_regularization(gt1, gt2, pred1, pred2):
    # print(gt1.shape, pred2.shape)
    disp1 = torch.cat((gt1, pred2), axis=2)
    disp2 = torch.cat((gt2, pred1), axis=2)
    dists = torch.cdist(disp1, disp2)
    min_row = dists.min(axis=2)[0]
    min_col = dists.min(axis=1)[0]

    reg_loss = 0.5 * (min_row.mean(axis=-1) + min_col.mean(axis=-1))

    return reg_loss


def sample_points_from_edges(verts, edges, num_points):
    edge_points = verts[edges]

    inds = torch.randint(low=0, high=edge_points.shape[0], size=(num_points,))
    weights = torch.rand(size=(num_points, 1))
    comb_weights = torch.hstack((weights, 1 - weights)).unsqueeze(-1)

    # print(inds.shape, edge_points.shape, comb_weights.shape)

    sel_edges = edge_points[inds]
    weighted_vals = comb_weights * sel_edges
    fin_points = weighted_vals.sum(axis=1)

    return fin_points


def collate_reprs_p2p(data, num_surface_sample=2048, num_skel_sample=2048, return_occupancies=False,
                      occ_sample=1024, return_full_graphs=False, return_dijkstras=False,
                      sample_graph=True,
                      return_meshes=False):
    """
       data: is a list of tuples with (example, label, length)
             where 'example' is a tensor of arbitrary shape
             and label/length are scalars
    """

    all_clouds = []
    all_skeletons = []
    if return_occupancies:
        all_queries, all_labels = [], []

    if return_full_graphs:
        full_graphs = []

    if return_dijkstras:
        all_dijkstras = []

    if return_meshes:
        all_meshes = []

    for i, cur_data in enumerate(data):

        cur_queries, cur_occupancies, cur_cloud, cur_skeleton = cur_data[:4]

        full_verts = cur_skeleton[0]
        skel_edges = cur_skeleton[1]
        inds_cloud = np.random.randint(0, cur_cloud.shape[0], size=num_surface_sample)
        cur_cloud = cur_cloud[inds_cloud].unsqueeze(0)

        if sample_graph:
            skel_verts = sample_points_from_edges(full_verts, skel_edges, num_skel_sample)
            skel_verts = skel_verts.unsqueeze(0)
        else:
            inds_skel = np.random.randint(0, full_verts.shape[0], size=num_skel_sample)
            skel_verts = full_verts[inds_skel]
            skel_verts = skel_verts.unsqueeze(0)
            #skel_verts, inds_skel = fps_from_cloud(full_verts, N=num_skel_sample)
            #print(skel_verts.shape)
            inds_skel = inds_skel[0]
        all_clouds += [cur_cloud]
        all_skeletons += [skel_verts]

        if return_occupancies:
            inds_occ = np.random.randint(0, cur_occupancies.shape[0], size=occ_sample)
            sampled_queries = cur_queries[inds_occ].unsqueeze(0)
            sampled_occupancies = cur_occupancies[inds_occ].unsqueeze(0)
            all_queries += [sampled_queries]
            all_labels += [sampled_occupancies]

        if return_full_graphs:
            full_graphs += [cur_skeleton]

        if return_dijkstras:
            cur_dijkstra = cur_data[4]
            all_dijkstras += [cur_dijkstra]

        if return_meshes:
            all_meshes += [cur_data[-2]]

    all_clouds = torch.cat(all_clouds, axis=0)
    all_skeletons = torch.cat(all_skeletons, axis=0)

    result = [all_clouds, all_skeletons]

    if return_occupancies:
        all_queries = torch.cat(all_queries, axis=0)
        all_labels = torch.cat(all_labels, axis=0)
        result += [all_queries, all_labels]

    if return_full_graphs:
        result += [full_graphs]

    if return_dijkstras:
        result += [all_dijkstras]

    if return_meshes:
        result += [all_meshes]

    return result

    #if not return_occupancies:
    #    return all_clouds, all_skeletons
    #else:
    #    all_queries = torch.cat(all_queries, axis=0)
    #    all_labels = torch.cat(all_labels, axis=0)
    #    return all_clouds, all_skeletons, all_queries, all_labels
