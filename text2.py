import trimesh
import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.decomposition import PCA

mesh= trimesh.load_mesh('/Users/darkqian/PycharmProjects/MR/benchmark/db/0/m6/m6.off')



#Ori
#print('特征值：{}\n特征向量：{}'.format(pca.explained_variance_, pca.components_),"\n")
print('Original')
print('Barycenter:', mesh.center_mass)
print('The size of the bounding box(length,width,height):', mesh.bounding_box.extents)
mesh.show()


#Centering

matrix = np.array([-mesh.center_mass[0] if mesh.center_mass[0] >= 0 else np.absolute(mesh.center_mass[0]),
                   -mesh.center_mass[1] if mesh.center_mass[1] >= 0 else np.absolute(mesh.center_mass[1]),
                   -mesh.center_mass[2] if mesh.center_mass[2] >= 0 else np.absolute(mesh.center_mass[2])])

mesh.apply_translation(matrix)  # move the tresh to the originz`
print('Centering done')
print('Barycenter:', mesh.center_mass)
print('The size of the bounding box(length,width,height):', mesh.bounding_box.extents)
mesh.show()

#Scaling
mesh.apply_scale(pow(1 / mesh.bounding_box_oriented.volume, 1 / 3))
print('Scaling done')
print('Barycenter:', mesh.center_mass)
print('The size of the bounding box(length,width,height):', mesh.bounding_box.extents)
mesh.show()

#print(mesh.center_mass)
#print(mesh.bounding_box_oriented.volume)


#Alignment
pca = PCA(n_components=2)
Reduced_mesh = pca.fit_transform(mesh.vertices)
print(pca.components_)

transform_x = trimesh.geometry.align_vectors(pca.components_[0], [1, 0, 0])
mesh.apply_transform(transform_x)
Reduced_mesh_newx = pca.fit_transform(mesh.vertices)


transform_y = trimesh.geometry.align_vectors(pca.components_[1], [0, 1, 0])
mesh.apply_transform(transform_y)
print(transform_x)
Reduced_mesh_newy = pca.fit_transform(mesh.vertices)
print(pca.components_)


print('Alignment done')
print('Barycenter:', mesh.center_mass)
print('The size of the bounding box(length,width,height):', mesh.bounding_box.extents)
mesh.show()

#Flipping

flipping_matrix = trimesh.geometry.align_vectors([0, 1, 0], [0, -1, 0])
mesh.apply_transform(flipping_matrix)
mesh.show()
