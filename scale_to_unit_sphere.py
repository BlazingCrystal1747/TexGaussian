import os
import trimesh
import numpy as np

def scale_to_unit_sphere(mesh, evaluate_metric = False):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    vertices = mesh.vertices - mesh.bounding_box.centroid
    distances = np.linalg.norm(vertices, axis=1)
    vertices /= np.max(distances)
    if evaluate_metric:
        vertices /= 2
    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)

mesh_path = 'input_mesh_path'
mesh = trimesh.load(mesh_path, force = 'mesh')
scaled_mesh = scale_to_unit_sphere(mesh)
scaled_mesh.export('output_mesh_path')