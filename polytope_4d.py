import permutation
import itertools
import numpy as np
import scipy


def generate_facets_convex_hull(vertices: np.ndarray, tol: float = 1e-5):
    ndim = vertices.shape[1]
    ch = scipy.spatial.ConvexHull(vertices)
    assert ch.vertices.shape[0] == vertices.shape[0]
    num_simplices = ch.simplices.shape[0]

    def same_normal(n1, n2):
        is_same = True
        for i in range(ndim):
            for j in range(i + 1, ndim):
                det_ij = n1[i] * n2[j] - n1[j] * n2[i]
                if abs(det_ij) >= tol:
                    is_same = False
        return is_same

    facets = []
    facet_equations = []

    # merge simplices into facets
    simplex_merged = np.zeros(num_simplices, dtype=bool)
    for si in range(num_simplices):
        if simplex_merged[si]:
            continue
        face = set(ch.simplices[si])
        for sj in range(si + 1, num_simplices):
            if same_normal(ch.equations[si, 0:ndim], ch.equations[sj, 0:ndim]):
                new_face = set(ch.simplices[sj])
                if len(face.intersection(new_face)) > 0:
                    face = face.union(new_face)
                    simplex_merged[sj] = True
        facets.append(face)
        facet_equations.append(ch.equations[si])

    return facets, facet_equations


def get_axis_4d(w_axis):
    # it can be proved that,
    # the following four column vectors constitute an orthogonal 4D coordinate (with det > 0)
    # (prove: the matrix is orthogonal)
    # (or using quaternion to prove (u+vi+sj+tk) * (w+xi+yj+zk))
    u, v, s, t = w_axis
    axis_mat = np.array([
        [u, -v, -s, -t],
        [v, u, -t, s],
        [s, t, u, -v],
        [t, -s, v, u],
    ], dtype=float)
    return axis_mat


class Polytope4D:
    def __init__(self):
        self.vertices = []
        self.edges = []
        self.cells = []
        self.cell_equations = []
        self.faces = []

    @property
    def num_vertices(self):
        return len(self.vertices)

    @property
    def num_edges(self):
        return len(self.edges)

    @property
    def num_faces(self):
        return len(self.faces)

    @property
    def num_cells(self):
        return len(self.cells)

    def to_file(self, fpath):
        with open(fpath, 'w') as fout:
            fout.write(f'vertices {self.num_vertices}\n')
            for i in range(self.num_vertices):
                fout.write(f'{self.vertices[i][0]} {self.vertices[i][1]} {self.vertices[i][2]} {self.vertices[i][3]}\n')
            fout.write(f'edges {self.num_edges}\n')
            for i in range(self.num_edges):
                fout.write(f'{self.edges[i][0]} {self.edges[i][1]}\n')
            fout.write(f'faces {self.num_faces}\n')
            for i in range(self.num_faces):
                for k, v in enumerate(self.faces[i]):
                    if k == 0:
                        fout.write(f'{v}')
                    else:
                        fout.write(f' {v}')
                fout.write(f'\n')
            fout.write(f'cells {self.num_cells}\n')
            for i in range(self.num_cells):
                for k, v in enumerate(self.cells[i]):
                    if k == 0:
                        fout.write(f'{v}')
                    else:
                        fout.write(f' {v}')
                fout.write(f'\n')

    @classmethod
    def from_file(cls, fpath):
        obj = cls()
        with open(fpath, 'r') as fin:
            s = fin.readline()
            while s != '':
                key, num = s.split()
                num = int(num)
                if key == 'vertices':
                    obj.vertices = []
                    for i in range(num):
                        data = fin.readline().split()
                        v_coord = np.zeros(4, dtype=float)
                        for j in range(4):
                            v_coord[j] = float(data[j])
                        obj.vertices.append(v_coord)
                elif key == 'edges':
                    obj.edges = []
                    for i in range(num):
                        data = fin.readline().split()
                        obj.edges.append((int(data[0]), int(data[1])))
                elif key == 'faces':
                    obj.faces = []
                    for i in range(num):
                        obj.faces.append(set(map(int, fin.readline().split())))
                elif key == 'cells':
                    obj.cells = []
                    for i in range(num):
                        obj.cells.append(set(map(int, fin.readline().split())))
                s = fin.readline()
        return obj

    def add_vertices_prod(self, prod):
        for v in itertools.product(*prod):
            self.vertices.append(np.array(v, dtype=float))

    def add_vertices_prod_cyclic(self, prod):
        for v in itertools.product(*prod):
            for indices in permutation.cyclic_permutation_4:
                self.vertices.append(np.array([v[i] for i in indices], dtype=float))

    def add_vertices_prod_even_permutation(self, prod):
        for v in itertools.product(*prod):
            for indices in permutation.even_permutation_4:
                self.vertices.append(np.array([v[i] for i in indices], dtype=float))

    def generate_edges_by_dist(self, edge_length: float, tol: float = 1e-5):
        for i in range(0, self.num_vertices):
            for j in range(i + 1, self.num_vertices):
                dist = np.linalg.norm(self.vertices[i] - self.vertices[j])
                if abs(dist - edge_length) < tol:
                    self.edges.append((i, j))

    def generate_edges_all(self):
        for i in range(0, self.num_vertices):
            for j in range(i + 1, self.num_vertices):
                self.edges.append((i, j))

    def generate_cells_convex_hull(self):
        self.cells, self.cell_equations = generate_facets_convex_hull(np.array(self.vertices))

    def generate_faces_from_cells_convex_hull(self):
        faces = []
        for cell_idx in range(self.num_cells):
            cell = self.cells[cell_idx]
            normal = self.cell_equations[cell_idx][:-1]
            new_coord = get_axis_4d(normal)
            cell_list = list(cell)
            coord_3d = np.zeros([len(cell_list), 3], dtype=float)
            for i, v in enumerate(cell_list):
                coord_3d[i] = (self.vertices[v] @ new_coord)[1:]
            new_faces, _ = generate_facets_convex_hull(coord_3d)
            for new_face in new_faces:
                faces.append({cell_list[i] for i in new_face})
        # remove duplicates
        face_duplicate = np.zeros(len(faces), dtype=bool)
        for i in range(len(faces)):
            if face_duplicate[i]:
                continue
            for j in range(i + 1, len(faces)):
                if faces[i] == faces[j]:
                    face_duplicate[j] = True
            self.faces.append(faces[i])

    def move_center_to_origin(self):
        vert_center = np.zeros(4, dtype=float)
        for v in self.vertices:
            vert_center += v
        vert_center /= self.num_vertices
        for v in self.vertices:
            v -= vert_center


class FiveCell(Polytope4D):
    def __init__(self):
        # 5-cell/4-simplex, analog of tetrahedron
        super().__init__()
        s = (1 - np.sqrt(5)) / 4
        A1 = np.array([1, 0, 0, 0], dtype=float)
        A2 = np.array([0, 1, 0, 0], dtype=float)
        A3 = np.array([0, 0, 1, 0], dtype=float)
        A4 = np.array([0, 0, 0, 1], dtype=float)
        A5 = np.array([s, s, s, s], dtype=float)
        self.vertices = [A1, A2, A3, A4, A5]
        self.move_center_to_origin()

        self.generate_edges_all()
        self.generate_cells_convex_hull()
        self.generate_faces_from_cells_convex_hull()


class EightCell(Polytope4D):
    def __init__(self):
        # 8-cell/tesseract/4-hypercube/4-cube, analog of cube
        super().__init__()
        self.add_vertices_prod([[-1, +1], [-1, +1], [-1, +1], [-1, +1]])
        self.generate_edges_by_dist(2)
        self.generate_cells_convex_hull()
        self.generate_faces_from_cells_convex_hull()


class SixteenCell(Polytope4D):
    def __init__(self):
        # 16-cell, analog of octahedron
        super().__init__()
        self.add_vertices_prod_cyclic([[-1, +1], [0], [0], [0]])
        self.generate_edges_by_dist(np.sqrt(2))
        self.generate_cells_convex_hull()
        self.generate_faces_from_cells_convex_hull()


class TwentyFourCell(Polytope4D):
    def __init__(self):
        # 24-cell, no analog
        super().__init__()
        for value1 in [-1.0, +1.0]:
            for value2 in [-1.0, +1.0]:
                for coord_idx1 in range(0, 4):
                    for coord_idx2 in range(coord_idx1 + 1, 4):
                        v = np.zeros(4, dtype=float)
                        v[coord_idx1] = value1
                        v[coord_idx2] = value2
                        self.vertices.append(v)

        self.generate_edges_by_dist(np.sqrt(2))
        self.generate_cells_convex_hull()
        self.generate_faces_from_cells_convex_hull()


class OneHundredTwentyCell(Polytope4D):
    def __init__(self):
        # 120-cell, analog of dodecahedron
        super().__init__()
        phi = (1 + np.sqrt(5)) / 2
        # 24
        for value1 in [-2.0, +2.0]:
            for value2 in [-2.0, +2.0]:
                for coord_idx1 in range(0, 4):
                    for coord_idx2 in range(coord_idx1 + 1, 4):
                        v = np.zeros(4, dtype=float)
                        v[coord_idx1] = value1
                        v[coord_idx2] = value2
                        self.vertices.append(v)
        # 64
        self.add_vertices_prod_cyclic([
            [-np.power(phi, -2), +np.power(phi, -2)],
            [-phi, +phi],
            [-phi, +phi],
            [-phi, +phi]
        ])
        # 64
        self.add_vertices_prod_cyclic([[-np.sqrt(5), +np.sqrt(5)], [-1, +1], [-1, +1], [-1, +1]])
        # 64
        self.add_vertices_prod_cyclic([
            [-np.power(phi, 2), +np.power(phi, 2)],
            [-1 / phi, +1 / phi],
            [-1 / phi, +1 / phi],
            [-1 / phi, +1 / phi]
        ])
        # 96
        self.add_vertices_prod_even_permutation([[0], [-1 / phi, +1 / phi], [-phi, +phi], [-np.sqrt(5), +np.sqrt(5)]])
        # 96
        self.add_vertices_prod_even_permutation([
            [0],
            [-np.power(phi, -2), +np.power(phi, -2)],
            [-1, +1],
            [-np.power(phi, 2), +np.power(phi, 2)]
        ])
        # 192
        self.add_vertices_prod_even_permutation([[-np.power(phi, -1), +np.power(phi, -1)], [-1, +1], [-phi, +phi], [-2, +2]])

        self.generate_edges_by_dist(3 - np.sqrt(5))
        self.generate_cells_convex_hull()
        self.generate_faces_from_cells_convex_hull()


class SixHundredCell(Polytope4D):
    def __init__(self):
        # 600-cell, analog of icosahedron
        super().__init__()
        phi = (1 + np.sqrt(5)) / 2
        # 8
        self.add_vertices_prod_cyclic([[-1, +1], [0], [0], [0]])
        # 16
        self.add_vertices_prod([[-0.5, +0.5], [-0.5, +0.5], [-0.5, +0.5], [-0.5, +0.5]])
        # 96
        self.add_vertices_prod_even_permutation([[-phi / 2, +phi / 2], [-0.5, +0.5], [-0.5 / phi, +0.5 / phi], [0]])

        self.generate_edges_by_dist(1 / phi)
        self.generate_cells_convex_hull()
        self.generate_faces_from_cells_convex_hull()


if __name__ == '__main__':
    # generate data for all regular convex polytopes in 4D
    data_dir = './data'
    FiveCell().to_file(f'{data_dir}/5-cell.txt')
    EightCell().to_file(f'{data_dir}/8-cell.txt')
    SixteenCell().to_file(f'{data_dir}/16-cell.txt')
    TwentyFourCell().to_file(f'{data_dir}/24-cell.txt')
    OneHundredTwentyCell().to_file(f'{data_dir}/120-cell.txt')
    SixHundredCell().to_file(f'{data_dir}/600-cell.txt')
