# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. https://creativecommons.org/licenses/by-nc-sa/4.0/

from plyfile import PlyData, PlyElement
import numpy as np
import torch


# vert_props is list of length len(prop_names) all numpy arrays of length N. faces is Mx3
def write_ply(
    save_name,
    vert_props,
    prop_names=["x", "y", "z", "nx", "ny", "nz"],
    prop_types=None,
    faces=None,
    as_text=False,
):

    # make vertex element
    if prop_types is None:
        prop_types = ["float32" for _ in prop_names]

    vert_props_list = [x.tolist() for x in vert_props]

    struct_array_data = [t for t in zip(*vert_props_list)]

    data_dtype = [x for x in zip(prop_names, prop_types)]
    struct_array = np.array(struct_array_data, dtype=data_dtype)

    vert_elem = PlyElement.describe(struct_array, "vertex")

    all_elem = [vert_elem]

    # make face element
    if faces is not None:
        ply_faces = np.empty(len(faces), dtype=[("vertex_indices", "i4", (3,))])
        ply_faces["vertex_indices"] = faces
        face_elem = PlyElement.describe(ply_faces, "face")
        all_elem.append(face_elem)

    ply_data = PlyData(all_elem, text=as_text)
    ply_data.write(save_name)


def write_ply_standard(save_name, verts, normal=None, colors=None, faces=None, as_text=False):
    # verts: (N,3) float
    # normals: (N,3) float
    # colors: (N,3) float in [0,1]
    # faces: (F,3)

    vert_props = torch.unbind(verts.detach().cpu(), dim=1)
    vert_props = [x.numpy() for x in vert_props]
    prop_names = ["x", "y", "z"]
    prop_types = 3 * ["float32"]

    if normal is not None:
        normal_props = torch.unbind(normal.detach().cpu(), dim=1)
        vert_props += [x.numpy() for x in normal_props]
        prop_names += ["nx", "ny", "nz"]
        prop_types += 3 * ["float32"]

    if colors is not None:
        color_prop = torch.unbind(colors.detach().cpu().clamp(min=0.0, max=1.0), dim=1)
        vert_props += [(255 * x.numpy()).astype("uint8") for x in color_prop]
        prop_names += ["red", "green", "blue"]
        prop_types += 3 * ["uint8"]

    write_ply(
        save_name,
        vert_props,
        prop_names=prop_names,
        prop_types=prop_types,
        faces=faces,
        as_text=as_text,
    )


def read_ply(file_name, prop_names=["x", "y", "z", "nx", "ny", "nz"]):
    plydata = PlyData.read(file_name)
    elem_names = [e.name for e in plydata.elements]
    vert_props = []
    for prop_name in prop_names:
        vert_prop = plydata["vertex"][prop_name]
        vert_props.append(vert_prop)
    if "face" in elem_names:
        faces = plydata["face"].data
        faces = np.concatenate(faces.tolist(), axis=0)
    else:
        faces = None

    return vert_props, faces


if __name__ == "__main__":
    pc = np.random.randn(5, 3)
    (*vert_props,) = pc.T
    prop_types = ["float32" for _ in range(0, 3)]
    print(vert_props)
    print(prop_types)
    # write_ply('test.ply',vert_props,prop_names=['x','y','z'],prop_types=prop_types)

    # x = np.array([('Rex', 9, 81.0), ('Fido', 3, 27.0)],dtype=[('name', 'U10'), ('age', 'i4'), ('weight', 'f4')])
    # x = np.array([('Rex', 9, 81.0), ('Fido', 3, 27.0)],dtype=[('name', 'U10'), ('age', 'i4'), ('weight', 'f4')])

    # 23
    # 01
    x = np.array([0, 1, 0, 1])
    y = np.array([0, 0, 1, 1])
    z = np.array([0, 0, 0, 0])

    faces = np.array([[0, 1, 2], [1, 3, 2]])
    nx = np.array([0, 0, 0, 0])
    ny = np.array([0, 0, 0, 0])
    nz = np.array([1, 1, 1, 1])

    write_ply("test_square_new.ply", [x, y, z, nx, ny, nz], faces=faces, as_text=True)
    read_ply("test_square_new.ply")
