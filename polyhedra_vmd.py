"""
Visualize 3D polyhedra in VMD software
This script writes a vtf file that can be opened in VMD software.
VMD is a software for analysing results of molecular dynamics simulations,
but I found it can also visualize 3D polyhedra!
Open VMD, then File -> New Molecule, select the vtf file and Load.
For best visualization, you can set Display -> Orthographic
and Graphics -> Representations -> Drawing Method -> Lines, increase line thickness to 5
"""
import numpy as np
import polyhedra


def quaternion_mul(p, q):
    return np.array([
        p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3],
        p[0] * q[1] + p[1] * q[0] + p[3] * q[2] - p[2] * q[3],
        p[0] * q[2] + p[2] * q[0] + p[1] * q[3] - p[3] * q[1],
        p[0] * q[3] + p[3] * q[0] + p[2] * q[1] - p[1] * q[2],
    ], dtype=float)


def quaternion_norm(p):
    return np.linalg.norm(p)


def quaternion_conj(p):
    return np.array([p[0], -p[1], -p[2], -p[3]])


def quaternion_inv(p):
    p_norm2 = p[0] ** 2 + p[1] ** 2 + p[2] ** 2 + p[3] ** 2
    return quaternion_conj(p) / p_norm2


def vector2quaternion(v):
    return np.array([0, *v], dtype=float)


def quaternion2vector(q):
    return q[1:]


def quaternion_double_rotation(axis, theta):
    return np.array([
        np.cos(theta),
        np.sin(theta) * axis[0],
        np.sin(theta) * axis[1],
        np.sin(theta) * axis[2],
    ])


def quaternion_rotation(axis, theta):
    return quaternion_double_rotation(axis, theta / 2)


def rotation(point, axis, theta):
    rot = quaternion_rotation(axis, theta)
    rot_ = quaternion_conj(rot)
    q = vector2quaternion(point)
    q1 = quaternion_mul(quaternion_mul(rot, q), rot_)
    return quaternion2vector(q1)


def td_rotation(point, t):
    theta = t / 2000 * 2 * np.pi
    axis = np.array([0, 0, 1], dtype=float)
    return rotation(point, axis, theta)


fake_elem = 'C'


def write_vtf_header(fout, poly: polyhedra.Polyhedra, fake_elem: str = 'C'):
    fout.write(f'atom 0:{poly.num_vertices} radius 0.1 name {fake_elem}\n')
    for bond in poly.edges:
        fout.write(f'bond {bond[0]}:{bond[1]}\n')


def write_frame_vtf(fout, frame: int, vertices: list):
    fout.write('timestep\n')
    for v in vertices:
        p_proj = td_rotation(v, frame)
        fout.write(f'{p_proj[0]} {p_proj[1]} {p_proj[2]}\n')


def write_file_vtf(fname: str, poly: polyhedra.Polyhedra):
    with open(fname, 'w') as fout:
        write_vtf_header(fout, poly, fake_elem=fake_elem)
        for i in range(2000):
            write_frame_vtf(fout, i, poly.vertices)


out_dir = 'vtf'
write_file_vtf(f'{out_dir}/tetrahedron.vtf', polyhedra.Tetrahedron())
write_file_vtf(f'{out_dir}/cube.vtf', polyhedra.Cube())
write_file_vtf(f'{out_dir}/octahedron.vtf', polyhedra.Octahedron())
write_file_vtf(f'{out_dir}/dodecahedron.vtf', polyhedra.Dodecahedron())
write_file_vtf(f'{out_dir}/icosahedron.vtf', polyhedra.Icosahedron())
