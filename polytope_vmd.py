"""
Visualize 4D polytopes in VMD software
This script writes a vtf file that can be opened in VMD software.
VMD is a software for analysing results of molecular dynamics simulations,
but I found it can also visualize 4D polytopes!
Open VMD, then File -> New Molecule, select the vtf file and Load.
For best visualization, you can set Display -> Orthographic
and Graphics -> Representations -> Drawing Method -> Lines, increase line thickness to 5
"""
import numpy as np
import polytope_4d


def proj_perspective(point, d):
    return point[1:4] / (1 - point[0] / d)


def proj_orthographic(point):
    return point[1:4]


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


def quaternion_double_rotation(axis, theta):
    return np.array([
        np.cos(theta),
        np.sin(theta) * axis[0],
        np.sin(theta) * axis[1],
        np.sin(theta) * axis[2],
    ])


def quaternion_rotation(axis, theta):
    return quaternion_double_rotation(axis, theta / 2)


def rotation_4d(point, axis, theta):
    rot = quaternion_rotation(axis, theta)
    return quaternion_mul(quaternion_mul(rot, point), rot)


def rotation_3d(point, axis, theta):
    rot = quaternion_rotation(axis, theta)
    rot_ = quaternion_conj(rot)
    return quaternion_mul(quaternion_mul(rot, point), rot_)


def left_rotation(point, axis, theta):
    rot = quaternion_double_rotation(axis, theta)
    return quaternion_mul(rot, point)


def right_rotation(point, axis, theta):
    rot = quaternion_double_rotation(axis, theta)
    return quaternion_mul(point, rot)


# def td_rotation(point, t):
#     theta = t / 2000 * 2 * np.pi
#     axis = np.array([0, 0, 1], dtype=float)
#     return rotation_4d(point, axis, theta)


def td_rotation(point, t):
    theta = t / 2000 * 2 * np.pi
    axis = np.array([0, 0, 1], dtype=float)
    q = quaternion_rotation(axis, 2 * theta)
    return quaternion_mul(q, point)


fake_elem = 'C'
d = 4


def write_vtf_header(fout, polytope: polytope_4d.Polytope4D, fake_elem: str = 'C', highlight_cell: int = -1, highlight_elem: str = 'N'):
    if highlight_cell == -1:
        fout.write(f'atom 0:{polytope.num_vertices} radius 0.1 name {fake_elem}\n')
    else:
        if 0 <= highlight_cell < polytope.num_cells:
            cell = polytope.cells[highlight_cell]
            fout.write(f'atom default radius 0.1 name {fake_elem}\n')
            fout.write(f'atom ')
            for i, v in enumerate(cell):
                if i == 0:
                    fout.write(f'{v}')
                else:
                    fout.write(f',{v}')
            fout.write(f' radius 0.1 name {highlight_elem}\n')
            if polytope.num_vertices - 1 not in cell:
                fout.write(f'atom {polytope.num_vertices - 1} radius 0.1 name {fake_elem}\n')
        else:
            raise Exception('cell index overflows!')
    for bond in polytope.edges:
        fout.write(f'bond {bond[0]}:{bond[1]}\n')


def write_frame_vtf(fout, frame: int, vertices: list):
    fout.write('timestep\n')
    for v in vertices:
        p_proj = proj_perspective(td_rotation(v, frame), d)
        fout.write(f'{p_proj[0]} {p_proj[1]} {p_proj[2]}\n')


def write_file_vtf(fname: str, polytope: polytope_4d.Polytope4D):
    with open(fname, 'w') as fout:
        write_vtf_header(fout, polytope, fake_elem=fake_elem, highlight_cell=-1)
        for i in range(2000):
            write_frame_vtf(fout, i, polytope.vertices)


out_dir = 'vtf'
write_file_vtf(f'{out_dir}/5-cell.vtf', polytope_4d.FiveCell())
write_file_vtf(f'{out_dir}/8-cell.vtf', polytope_4d.EightCell())
write_file_vtf(f'{out_dir}/16-cell.vtf', polytope_4d.SixteenCell())
write_file_vtf(f'{out_dir}/24-cell.vtf', polytope_4d.TwentyFourCell())
write_file_vtf(f'{out_dir}/120-cell.vtf', polytope_4d.OneHundredTwentyCell())
write_file_vtf(f'{out_dir}/600-cell.vtf', polytope_4d.SixHundredCell())
