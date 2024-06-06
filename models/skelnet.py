import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin
import numpy as np

from torch_cluster import fps, knn, radius
from .dilg_models import Embedding, PointConv, VisionTransformer, embed
from torch.nn.utils import weight_norm
from torch.nn import ReLU
from functools import partial
from .p2pnet import P2PNetPointnet2
from einops import rearrange
from collections import Counter
from .vecset_encoder import *


def get_topk_nns_dilated(query, reference, k, dilation=1):
    dists = torch.cdist(query, reference)
    dists_sorted, inds = torch.sort(dists, axis=2)
    presel_inds = list(range(dilation * k))[::dilation]
    sel_inds = inds[:, :, presel_inds]

    ret = reference[torch.arange(reference.size(0)).unsqueeze(1).unsqueeze(2), sel_inds]

    return ret, sel_inds, dists_sorted[:, :, presel_inds]


def get_topk_nns_cluster(query, reference, k, compute_dists=False, dilation=1):
    B1, N1, D1 = query.shape
    B2, N2, D2 = reference.shape
    assert B1 == B2 and D1 == D2

    query_flat = query.view(B1 * N1, D1)
    ref_flat = reference.view(B2 * N2, D1)

    batch1 = torch.arange(B1).to(query.device)
    batch1 = torch.repeat_interleave(batch1, N1)
    batch2 = torch.arange(B2).to(reference.device)
    batch2 = torch.repeat_interleave(batch2, N2)

    idx = knn(ref_flat, query_flat, k, batch_x=batch2, batch_y=batch1)
    idx = idx
    ref_pts = ref_flat[idx[1]]
    ref_pts = ref_pts.reshape(B1, N1, k, D1)
    ref_inds = idx[1].reshape(B1, N1, k)
    ref_inds = ref_inds % N2

    if compute_dists:
        disps = query.unsqueeze(2) - ref_pts
        dists = torch.linalg.norm(disps, axis=-1, keepdim=True) + 1e-10
    else:
        dists = None

    return ref_pts, ref_inds, dists


class DisplacementDecoder(nn.Module):
    def __init__(self, latent_channel=128, smoothing=False):
        super().__init__()

        self.fc = Embedding(latent_channel=latent_channel)
        self.log_sigma = nn.Parameter(torch.FloatTensor([3.5]))
        # self.register_buffer('log_sigma', torch.Tensor([-3.0]))

        self.embed = Seq(Lin(48 + 3, latent_channel))  # , nn.GELU(), Lin(128, 128))
        self.smoothing = smoothing
        self.embedding_dim = 48
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(self.embedding_dim // 6),
                       torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), e,
                       torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6),
                       torch.zeros(self.embedding_dim // 6), e]),
        ])
        self.register_buffer('basis', e)  # 3 x 16

        self.transformer = VisionTransformer(embed_dim=latent_channel,
                                             depth=6,
                                             num_heads=6,
                                             mlp_ratio=4.,
                                             qkv_bias=True,
                                             qk_scale=None,
                                             drop_rate=0.,
                                             attn_drop_rate=0.,
                                             drop_path_rate=0.1,
                                             norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                             init_values=0.,
                                             )

    def forward(self, latents, centers, unit_samples=None, latent_ids=None, smooth_latents=None):

        embeddings = embed(centers, self.basis)
        embeddings = self.embed(torch.cat([centers, embeddings], dim=2))
        latents = self.transformer(latents, embeddings)

        if latent_ids is not None:
            sel_latents = latents[torch.arange(latents.size(0)).unsqueeze(1), latent_ids]
        else:
            sel_latents = smooth_latents
        preds = self.fc(unit_samples, sel_latents).squeeze(2)

        return preds

    def get_final_latents(self, latents, centers):

        embeddings = embed(centers, self.basis)
        embeddings = self.embed(torch.cat([centers, embeddings], dim=2))
        latents = self.transformer(latents, embeddings)

        return latents

    def decode_queries(self, final_latents, unit_samples=None,
                       latent_ids=None):

        sel_latents = final_latents[torch.arange(final_latents.size(0)).unsqueeze(1), latent_ids]
        #print(unit_samples.shape, sel_latents.shape)
        preds = self.fc(unit_samples, sel_latents).squeeze(2)

        return preds


class SkelAutoencoder(nn.Module):
    def __init__(self, num_skel=256, latent_dim=256, use_skel_model=True,
                 surface_num=2048, num_skel_nn=64,
                 encoder_object=None,
                 decoder_object=DisplacementDecoder,
                 use_skel_correspondences=False,
                 use_quantizer=False,
                 skel_model_object=P2PNetPointnet2,
                 joint_patches=False,
                 use_aggregator=False,
                 agg_skel_nn=5,
                 num_disps=1,
                 num_encoder_heads=1,
                 clip_sdfs=False,
                 max_sdf_val=0.02):
        super().__init__()

        self.num_skel = num_skel
        self.latent_dim = latent_dim
        self.use_skel_model = use_skel_model
        self.joint_patches = joint_patches #legacy; left for compatibility
        self.encoder = encoder_object(num_skel=num_skel, dim=latent_dim, M=surface_num,
                                      use_skel_correspondences=use_skel_correspondences,
                                      num_skel_nn=num_skel_nn, joint_patches=joint_patches,
                                      num_encoder_heads=num_encoder_heads)
        self.decoder = decoder_object(latent_channel=latent_dim)
        if self.use_skel_model:
            self.skel_model = skel_model_object(noise_length=0, num_disps=num_disps)
        else:
            self.skel_model = None

        self.clip_sdfs = clip_sdfs
        self.max_sdf_val = max_sdf_val
        self.use_aggregator = use_aggregator #legacy; left for compatibility
        self.use_quantizer = use_quantizer #legacy; left for compatibility
        self.agg_skel_nn = agg_skel_nn #legacy; left for compatibility


    def get_skeleton(self, surface):

        assert self.use_skel_model == True, "Needs skeleton model to be defined"

        cloud_disp = self.skel_model(surface)
        cloud2skel = surface + cloud_disp

        return cloud2skel, cloud_disp

    def encode(self, skeleton, surface):

        latents, centers, (row, col) = self.encoder(skeleton, surface)

        if self.use_quantizer:
            quant_latents, quant_loss, _, _ = self.codebook(latents)
            return quant_latents, centers, (row, col), quant_loss
        else:
            return latents, centers, (row, col)

    def encode_with_encodings(self, skeleton, surface):

        assert self.use_quantizer is True
        latents, centers, (row, col) = self.encoder(skeleton, surface)

        quant_latents, quant_loss, _, encodings = self.codebook(latents)
        encodings = encodings.reshape(latents.shape[:2] + (-1,))

        return quant_latents, centers, (row, col), quant_loss, encodings


    def decode(self, latents, centers, flat_queries, flat_inds):

        preds = self.decoder.forward(latents, centers, flat_queries, flat_inds)
        preds = torch.sigmoid(preds)

        return preds

    def get_weighted_sdfs(self, queries, centers, latents, return_unweighted_sdfs=False):
        res, dirs, dists, skel_ids, skel_pts = self.aggregator.forward(queries, centers)
        flat_queries = dirs.reshape(dirs.shape[0], -1, 3)
        flat_inds = skel_ids.reshape(skel_ids.shape[0], -1)

        preds = self.decoder.forward(latents, centers, flat_queries, flat_inds)
        preds = torch.sigmoid(preds)
        preds = preds.reshape(dists.shape)

        pred_surf = skel_pts + preds*dirs
        loc_sdfs = preds - dists  # alpha - d

        weighted_sdfs = loc_sdfs.squeeze(-1) * res
        weighted_sdfs = weighted_sdfs.sum(axis=-1)
        return weighted_sdfs, pred_surf


    def get_weighted_sdfs_optimized(self, queries, centers, final_latents, return_unweighted_sdfs=False):
        res, dirs, dists, skel_ids, skel_pts = self.aggregator.forward(queries, centers)
        flat_queries = dirs.reshape(dirs.shape[0], -1, 3)
        flat_inds = skel_ids.reshape(skel_ids.shape[0], -1)

        preds = self.decoder.decode_queries(final_latents, flat_queries, flat_inds)
        preds = torch.sigmoid(preds)
        preds = preds.reshape(dists.shape)

        pred_surf = skel_pts + preds*dirs
        loc_sdfs = preds - dists  # alpha - d

        weighted_sdfs = loc_sdfs.squeeze(-1) * res
        weighted_sdfs = weighted_sdfs.sum(axis=-1)
        return weighted_sdfs, pred_surf


    def get_skelray_nn_sdfs(self, queries, centers,
                            latents, skel_nn=1,
                            dir_consistency=True,
                            cosine_thres=0.86):

        skel_pts, dirs, dists, skel_ids = self.get_directions(queries, centers, skel_nn=skel_nn)
        flat_queries = dirs.reshape(dirs.shape[0], -1, 3)
        flat_inds = skel_ids.reshape(skel_ids.shape[0], -1)

        preds = self.decoder.forward(latents, centers, flat_queries, flat_inds)
        preds = torch.sigmoid(preds)
        preds = preds.reshape(dists.shape)
        pred_surf = skel_pts + preds * dirs
        loc_sdfs = preds - dists
        if dir_consistency:
            dir_cosines = dirs * dirs[:, :, [0], :]
            dir_cosines = dir_cosines.sum(axis=-1, keepdim=True)
            cosine_mask = dir_cosines < cosine_thres
            loc_sdfs[cosine_mask] = 0

        return loc_sdfs, pred_surf

    def get_skelray_nn_sdfs_optimized(self, queries, centers,
                            final_latents, skel_nn=1,
                            dir_consistency=True,
                            cosine_thres=0.86,
                            disp_cloud=None,
                            disp_inds=None):

        skel_pts, dirs, dists, skel_ids = self.get_directions(queries, centers, skel_nn=skel_nn,
                                                              disp_cloud=disp_cloud, disp_skel_inds=disp_inds)
        flat_queries = dirs.reshape(dirs.shape[0], -1, 3)
        flat_inds = skel_ids.reshape(skel_ids.shape[0], -1)

        preds = self.decoder.decode_queries(final_latents, flat_queries, flat_inds)
        preds = torch.sigmoid(preds)
        preds = preds.reshape(dists.shape)
        pred_surf = skel_pts + preds * dirs
        loc_sdfs = preds - dists
        if self.clip_sdfs:
            loc_sdfs = torch.clip(loc_sdfs, max=self.max_sdf_val)
        if dir_consistency:
            dir_cosines = dirs * dirs[:, :, [0], :]
            dir_cosines = dir_cosines.sum(axis=-1, keepdim=True)
            cosine_mask = dir_cosines < cosine_thres
            loc_sdfs[cosine_mask] = 0

        return loc_sdfs, pred_surf


    def get_skelray_projected_centers(self, queries, projected_centers,
                            projected_latents):

        print(queries.shape, projected_centers.shape)
        assert queries.shape == projected_centers.shape
        diffs = queries - projected_centers
        diffs_norms = torch.linalg.norm(diffs, axis=-1, keepdim=True) + 1e-10
        dirs = diffs / diffs_norms
        flat_queries = dirs.reshape(dirs.shape[0], -1, 3)
        flat_inds = torch.LongTensor(range(flat_queries.shape[1])).unsqueeze(0)
        flat_inds = flat_inds.repeat(len(flat_queries),1)

        preds = self.decoder.decode_queries(projected_latents, flat_queries, flat_inds)
        preds = torch.sigmoid(preds)
        preds = preds.reshape(diffs_norms.shape)
        pred_surf = projected_centers + preds * dirs
        loc_sdfs = preds - diffs_norms

        return loc_sdfs, pred_surf


    def get_directions(self, queries, skel, skel_nn=1, disp_cloud=None, disp_skel_inds=None):
        if disp_cloud is None:
            skel_nns, query_nn_inds, dists = get_topk_nns_cluster(queries, skel, k=skel_nn, dilation=1)
            disps = queries.unsqueeze(2) - skel_nns
            scales = torch.linalg.norm(disps, axis=-1, keepdim=True) + 1e-10
            directions = disps / scales
        else:
            assert disp_skel_inds is not None
            nns_test, disp_nn_inds, _ = get_topk_nns_cluster(queries, disp_cloud, k=skel_nn, dilation=1)
            query_nn_inds = disp_skel_inds[torch.arange(disp_skel_inds.size(0)).unsqueeze(1).unsqueeze(2), disp_nn_inds[:, :, :]]
            skel_nns = skel[torch.arange(skel.size(0)).unsqueeze(1).unsqueeze(2), query_nn_inds.squeeze(-1)]
            disps = queries.unsqueeze(2) - skel_nns
            scales = torch.linalg.norm(disps, axis=-1, keepdim=True) + 1e-10
            directions = disps / scales

        return skel_nns, directions, scales, query_nn_inds

    def forward(self, surface, skeleton, unit_queries, latent_inds):

        if self.use_skel_model:
            cloud2skel = self.get_skeleton(surface)
        else:
            cloud2skel = skeleton

        if self.use_quantizer:
            latents, centers, (row, col), quant_loss, _, _ = self.encoder.encode(cloud2skel, surface)
        else:
            latents, centers, (row, col) = self.encoder.forward(cloud2skel, surface)

        preds = self.decoder(latents, centers, unit_queries, latent_inds)
        preds = torch.sigmoid(preds)

        if self.use_quantizer:
            return preds, cloud2skel, latents, centers, quant_loss
        else:
            return preds, cloud2skel, latents, centers



