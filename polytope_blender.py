"""
Script for Blender animation
No extra dependencies required, but your need to generate polytope data files first
"""
import bpy
from mathutils import *
from math import *


class Polytope4D:  # simplified class for Blender animation
    def __init__(self):
        self.vertices = []
        self.edges = []
        self.faces = []
        self.cells = []
    
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
                        obj.vertices.append([float(data[j]) for j in range(4)])
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


class Transform4D:  # abstract class for 4D transformations
    def __init__(self):
        pass
    
    def __call__(self, point: Quaternion):
        return point


class Scale4D(Transform4D):
    def __init__(self, scale, center=None):
        super().__init__()
        self.scale = scale
        self.center = Quaternion(center) if center is not None else Quaternion([0, 0, 0, 0])

    def __call__(self, point: Quaternion):
        return self.scale * (point - self.center) + self.center


class Rotation4D(Transform4D):
    def __init__(self, axis, theta, center=None):
        super().__init__()
        self.axis = axis
        self.theta = theta
        self.center = Quaternion(center) if center is not None else Quaternion([0, 0, 0, 0])
        self.rot = Quaternion(axis, theta)

    def __call__(self, point: Quaternion):
        return self.rot @ (point - self.center) @ self.rot + self.center


class Rotation3D(Transform4D):
    def __init__(self, axis, theta, center=None):
        super().__init__()
        self.axis = axis
        self.theta = theta
        self.center = Quaternion(center) if center is not None else Quaternion([0, 0, 0, 0])
        self.rot = Quaternion(axis, theta)

    def __call__(self, point: Quaternion):
        return self.rot @ (point - self.center) @ self.rot.conjugated() + self.center


def DoubleRotation(axis, theta: float) -> Quaternion:
    cos_theta = cos(theta)
    sin_theta = sin(theta)
    return Quaternion([cos_theta, sin_theta * axis[0], sin_theta * axis[1], sin_theta * axis[2]])


class LeftRotation4D(Transform4D):
    def __init__(self, axis, theta, center=None):
        super().__init__()
        self.axis = axis
        self.theta = theta
        self.center = Quaternion(center) if center is not None else Quaternion([0, 0, 0, 0])
        self.rot = DoubleRotation(axis, theta)

    def __call__(self, point: Quaternion):
        return self.rot @ (point - self.center) + self.center


class RightRotation4D(Transform4D):
    def __init__(self, axis, theta, center=None):
        super().__init__()
        self.axis = axis
        self.theta = theta
        self.center = Quaternion(center) if center is not None else Quaternion([0, 0, 0, 0])
        self.rot = DoubleRotation(axis, theta)

    def __call__(self, point: Quaternion):
        return (point - self.center) @ self.rot + self.center


class Translation4D(Transform4D):
    def __init__(self, trans_vec):
        super().__init__()
        self.trans_vec = Quaternion(trans_vec)

    def __call__(self, point: Quaternion):
        return point + trans_vec


class ProjPerspective(Transform4D):
    def __init__(self, d):
        self.d = d
    
    def __call__(self, point):
        scale1 = 1 / (1 - point[0] / self.d)
        return Quaternion([0, scale1 * point[1], scale1 * point[2], scale1 * point[3]])


class CombinedTransform(Transform4D):
    def __init__(self, trans_list: list[Transform4D]):
        self.trans_list = trans_list
    
    def __call__(self, point):
        result = point
        for trans in self.trans_list:
            result = trans(result)
        return result


def cut_4d_to_3d(point_4d: Quaternion) -> Vector:
    return Vector([point_4d[1], point_4d[2], point_4d[3]])


def add_unit_sphere(name: str, radius=1):
    bpy.ops.mesh.primitive_uv_sphere_add(location=(0, 0, 0), scale=(radius, radius, radius))
    bpy.context.object.name = name
    bpy.ops.object.shade_smooth()
    return bpy.context.object


def add_cylinder(name: str, radius=1):
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=1.0, location=(0, 0, 0))
    bpy.context.object.name = name
    bpy.ops.object.shade_smooth()
    return bpy.context.object


def get_material_for_ball(name: str, color):
    material1 = bpy.data.materials.new(name)
    material1.use_nodes = True

    nodes = material1.node_tree.nodes
    nodes.clear()

    node_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    node_bsdf.inputs[0].default_value = color
    node_bsdf.inputs[1].default_value = 0.1

    node_out = nodes.new(type='ShaderNodeOutputMaterial')

    links = material1.node_tree.links
    link1 = links.new(node_bsdf.outputs[0], node_out.inputs[0])

    return material1


class Polytope4DObj:
    def __init__(self, polytope: Polytope4D, vertex_mtl, edge_mtl, vertex_r: float = 0.1, edge_r: float = 0.05, vertex_suffix: str = 'sphere_', edge_suffix: str = 'cylinder_'):
        self.polytope = polytope
        self.vertex_mtl = vertex_mtl
        self.edge_mtl = edge_mtl
        self.vertex_r = vertex_r
        self.edge_r = edge_r
        self.vertex_suffix = vertex_suffix
        self.edge_suffix = edge_suffix
        # add vertex spheres
        self.poly_vertex_spheres = []
        for i in range(self.polytope.num_vertices):
            sphere1 = add_unit_sphere(f'{self.vertex_suffix}{i}', radius=self.vertex_r)
            sphere1.data.materials.append(self.vertex_mtl)
            self.poly_vertex_spheres.append(sphere1)
        # add edge cylinders
        self.poly_edge_cylinders = []
        for i in range(self.polytope.num_edges):
            cylinder1 = add_cylinder(f'{self.edge_suffix}{i}', radius=self.edge_r)
            cylinder1.data.materials.append(self.edge_mtl)
            self.poly_edge_cylinders.append(cylinder1)
    
    def set_frame(self, frame: int, trans: Transform4D):
        # transforms vertices and edges to desired position
        trans_vertex_locations = []
        for i, sphere in enumerate(self.poly_vertex_spheres):
            sphere.location = cut_4d_to_3d(trans(Quaternion(self.polytope.vertices[i])))
            sphere.keyframe_insert(data_path='location', frame=frame)
            trans_vertex_locations.append(sphere.location)
        for k, cylinder in enumerate(self.poly_edge_cylinders):
            i, j = self.polytope.edges[k]
            vi = trans_vertex_locations[i]
            vj = trans_vertex_locations[j]
            dv = vj - vi
            dist = dv.length
            dv.normalize()
            cylinder.rotation_mode = 'QUATERNION'
            theta = acos(dv.z)
            cylinder.rotation_quaternion = Quaternion((-dv.y, dv.x, 0), theta)
            cylinder.keyframe_insert(data_path='rotation_quaternion', frame=frame)
            cylinder.location = (vi + vj) / 2
            cylinder.keyframe_insert(data_path='location', frame=frame)
            cylinder.scale.z = dist
            cylinder.keyframe_insert(data_path='scale', frame=frame)


if __name__ == '__main__':
    # example script, a 24-cell rotating in 4D space
    # you need to edit file path, materials, transformations, etc. for your own animation
    material_ball = get_material_for_ball('material_ball', (1, 0.87, 0.1, 1))
    material_cylinder = get_material_for_ball('material_cylinder', (0.1, 0.87, 1, 1))
    polytope = Polytope4D.from_file('./data/24-cell.txt')
    polytope_obj = Polytope4DObj(polytope, material_ball, material_cylinder)

    frame_per_second = 25
    rot_angle1 = pi*0.25*1.5
    rot_time1 = 12
    rot_angle2 = pi*0.25*1.5
    rot_time2 = 12
    rot_angle3 = pi*0.25*1.5
    rot_time3 = 12
    scale = 2.0
    d_proj = 8.0

    for frame in range(1, 1 + rot_time1 * frame_per_second):
        theta = (frame - 1) / (rot_time1 * frame_per_second) * rot_angle1
        trans = CombinedTransform([
            Scale4D(scale),
            Rotation3D((0, 0, -1), pi/8), 
            Rotation3D((0, 1, 0), pi/8), 
            Rotation4D((sin(pi/8.1), 0, cos(pi/8.1)), theta), 
            ProjPerspective(d_proj),
        ])
        polytope_obj.set_frame(frame, trans)

    for frame in range(1 + rot_time1 * frame_per_second, 1 + rot_time1 * frame_per_second + rot_time2 * frame_per_second):
        theta = (frame - 1 - rot_time1 * frame_per_second) / (rot_time2 * frame_per_second) * rot_angle2
        trans = CombinedTransform([
            Scale4D(scale),
            Rotation3D((0, 0, -1), pi/8), 
            Rotation3D((0, 1, 0), pi/8), 
            Rotation4D((sin(pi/8.1), 0, cos(pi/8.1)), rot_angle1), 
            LeftRotation4D((sin(pi/6), 0, cos(pi/6)), theta), 
            ProjPerspective(d_proj),
        ])
        polytope_obj.set_frame(frame, trans)

    start_frame = 1 + rot_time1 * frame_per_second + rot_time2 * frame_per_second
    end_frame = start_frame + rot_time3 * frame_per_second
    for frame in range(start_frame, end_frame):
        theta = (frame - start_frame) / (rot_time3 * frame_per_second) * rot_angle2
        trans = CombinedTransform([
            Scale4D(scale),
            Rotation3D((0, 0, -1), pi/8), 
            Rotation3D((0, 1, 0), pi/8), 
            Rotation4D((sin(pi/8.1), 0, cos(pi/8.1)), rot_angle1), 
            LeftRotation4D((sin(pi/6), 0, cos(pi/6)), rot_angle2), 
            RightRotation4D((0, sin(pi/3), cos(pi/3)), theta), 
            ProjPerspective(d_proj),
        ])
        polytope_obj.set_frame(frame, trans)
