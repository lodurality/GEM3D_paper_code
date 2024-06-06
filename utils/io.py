import numpy as np


def load_and_compress_dilg_shapenet_npz(path, keys_to_convert=['vol_points', 'near_points'],
                                        data_format=np.float16, vol_points_sample=150000,
                                        near_points_sample=150000,
                                        subsample=False):
    tst = np.load(path)
    d = dict(zip(("{}".format(k) for k in tst), (tst[k] for k in tst)))
    for key in keys_to_convert:
        d[key] = (d[key] / 10).astype(data_format)

    if subsample:
        d['vol_points'] = d['vol_points'][:vol_points_sample, :]
        d['vol_label'] = d['vol_label'][:vol_points_sample]
        d['near_points'] = d['near_points'][:near_points_sample, :]
        d['near_label'] = d['near_label'][:near_points_sample]

    return d


def get_graph(skel):
    verts = [tuple(item) for item in skel[:, 1:4]]
    verts += [tuple(item) for item in skel[:, 4:]]
    verts = list(set(verts))
    map_dict = dict(zip(verts, range(len(verts))))

    raw_edges = skel[:, 1:]
    all_edges = []
    for item in raw_edges:
        # print(item)
        vert1, vert2 = tuple(item[:3]), tuple(item[3:])
        edge = (map_dict[vert1], map_dict[vert2])
        all_edges += [edge]

    return verts, all_edges