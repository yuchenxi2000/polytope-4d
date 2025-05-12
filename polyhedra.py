import permutation
import itertools
import numpy as np
import scipy


class Polyhedra:
    def __init__(self):
        self.vertices = []
        self.edges = []
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

    def to_file(self, fpath):
        with open(fpath, 'w') as fout:
            fout.write(f'vertices {self.num_vertices}\n')
            for i in range(self.num_vertices):
                fout.write(f'{self.vertices[i][0]} {self.vertices[i][1]} {self.vertices[i][2]}\n')
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
                        v_coord = np.zeros(3, dtype=float)
                        for j in range(3):
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
                s = fin.readline()
        return obj

    def add_vertices_prod(self, prod):
        for v in itertools.product(*prod):
            self.vertices.append(np.array(v, dtype=float))

    def add_vertices_prod_cyclic(self, prod):
        for v in itertools.product(*prod):
            for indices in permutation.cyclic_permutation_3:
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

    def generate_faces_convex_hull(self):
        ch = scipy.spatial.ConvexHull(np.array(self.vertices))
        assert ch.vertices.shape[0] == self.num_vertices
        num_simplices = ch.simplices.shape[0]

        def same_normal(n1, n2):
            n3 = np.cross(n1, n2)
            return np.linalg.norm(n3) < 1e-5

        # merge simplices into faces
        simplex_merged = np.zeros(num_simplices, dtype=bool)
        for si in range(num_simplices):
            if simplex_merged[si]:
                continue
            face = set(ch.simplices[si])
            for sj in range(si + 1, num_simplices):
                if same_normal(ch.equations[si, 0:3], ch.equations[sj, 0:3]):
                    new_face = set(ch.simplices[sj])
                    if len(face.intersection(new_face)) > 0:
                        face = face.union(new_face)
                        simplex_merged[sj] = True
            self.faces.append(face)

    def move_center_to_origin(self):
        vert_center = np.zeros(3, dtype=float)
        for v in self.vertices:
            vert_center += v
        vert_center /= self.num_vertices
        for v in self.vertices:
            v -= vert_center

    def write_vtf_header(self, fout, fake_elem: str = 'C'):
        fout.write(f'atom 0:{self.num_vertices} radius 0.1 name {fake_elem}\n')
        for bond in self.edges:
            fout.write(f'bond {bond[0]}:{bond[1]}\n')

    def write_vtf(self, fout, fake_elem: str = 'C'):
        self.write_vtf_header(fout, fake_elem=fake_elem)
        fout.write('timestep\n')
        for v in self.vertices:
            fout.write(f'{v[0]} {v[1]} {v[2]}\n')


class Tetrahedron(Polyhedra):
    def __init__(self):
        super().__init__()
        A1 = np.array([1, 0, 0], dtype=float)
        A2 = np.array([0, 1, 0], dtype=float)
        A3 = np.array([0, 0, 1], dtype=float)
        A4 = np.array([1, 1, 1], dtype=float)
        self.vertices = [A1, A2, A3, A4]
        self.move_center_to_origin()

        self.generate_edges_all()
        self.generate_faces_convex_hull()


class Cube(Polyhedra):
    def __init__(self):
        super().__init__()
        self.add_vertices_prod([[-1, +1], [-1, +1], [-1, +1]])
        self.generate_edges_by_dist(2)
        self.generate_faces_convex_hull()


class Octahedron(Polyhedra):
    def __init__(self):
        super().__init__()
        self.add_vertices_prod_cyclic([[-1, +1], [0], [0]])
        self.generate_edges_by_dist(np.sqrt(2))
        self.generate_faces_convex_hull()


class Dodecahedron(Polyhedra):
    def __init__(self):
        super().__init__()
        h = (np.sqrt(5) - 1) / 2
        self.add_vertices_prod([[-1, +1], [-1, +1], [-1, +1]])
        self.add_vertices_prod_cyclic([[0], [-(1 + h), +(1 + h)], [-(1 - h ** 2), +(1 - h ** 2)]])
        self.generate_edges_by_dist(2 * (1 - h ** 2))
        self.generate_faces_convex_hull()


class Icosahedron(Polyhedra):
    def __init__(self):
        super().__init__()
        phi = (np.sqrt(5) + 1) / 2
        self.add_vertices_prod_cyclic([[-phi, +phi], [-1, +1], [0]])
        self.generate_edges_by_dist(2)
        self.generate_faces_convex_hull()


if __name__ == '__main__':
    # generate data for all regular convex polyhedra in 3D
    data_dir = './data'
    Tetrahedron().to_file(f'{data_dir}/tetrahedron.txt')
    Cube().to_file(f'{data_dir}/cube.txt')
    Octahedron().to_file(f'{data_dir}/octahedron.txt')
    Dodecahedron().to_file(f'{data_dir}/dodecahedron.txt')
    Icosahedron().to_file(f'{data_dir}/icosahedron.txt')
