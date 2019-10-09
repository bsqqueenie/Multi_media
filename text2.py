import trimesh
import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.decomposition import PCA

ori = [0, 0, 0]

def translation(mesh, center):
    i = 0
    for vertex in mesh.vertices:
        mesh.vertices[i] = vertex - center
        i += 1
    return mesh

def Normalization(path):

    try:
        mesh = trimesh.load_mesh(path)
    except:
        print("The file doesn't exist")
        return

    #Ori
    #print('特征值：{}\n特征向量：{}'.format(pca.explained_variance_, pca.components_),"\n")
    numOfvertives = mesh.vertices.shape[0]
    numOffaces = mesh.faces.shape[0]
    print('Original')
    print('Number of vertices and faces of the mesh:', numOfvertives, numOffaces)
    print('Barycenter:', mesh.center_mass)
    print('The size of the bounding box(length,width,height):', mesh.bounding_box_oriented.primitive.extents,"\n")
    mesh.show()

    #Centering

    center = mesh.center_mass
    Dis = np.linalg.norm(center - ori)

    while(Dis >= 0.05):

        mesh = translation(mesh, center)  # move the mesh to the originz
        center = mesh.center_mass
        print("New center:", center)
        Dis = np.linalg.norm(center - ori)
        print("Dis:", Dis)

    print('Centering done')
    print('Barycenter:', mesh.center_mass)
    print('The size of the bounding box(length,width,height):', mesh.bounding_box_oriented.primitive.extents,"\n")
    mesh.show()
    
    '''
    #Scaling

    maxLengthOfSide = max(mesh.bounding_box_oriented.primitive.extents)
    mesh.apply_scale(1 / maxLengthOfSide)

    print('Scaling done')
    print('Barycenter:', mesh.center_mass)
    print('The size of the bounding box(length,width,height):', mesh.bounding_box_oriented.primitive.extents,"\n")
    mesh.show()
    '''

    #Alignment

    pca = PCA(n_components=2)
    Reduced_mesh = pca.fit_transform(mesh.vertices)
    print(pca.components_)

    transform_x = trimesh.geometry.align_vectors(pca.components_[0], [1, 0, 0])
    mesh.apply_transform(transform_x)
    Reduced_mesh_newx = pca.fit_transform(mesh.vertices)


    transform_y = trimesh.geometry.align_vectors(pca.components_[1], [0, 1, 0])
    mesh.apply_transform(transform_y)
    Reduced_mesh_newy = pca.fit_transform(mesh.vertices)
    print(pca.components_)


    print('Alignment done')
    print('Barycenter:', mesh.center_mass)
    print('The size of the bounding box(length,width,height):', mesh.bounding_box_oriented.primitive.extents,"\n")
    mesh.show()

    #Flipping

    moment_lx = 0
    moment_rx = 0
    moment_ly = 0
    moment_ry = 0
    moment_lz = 0
    moment_rz = 0

    for vertex in mesh.vertices:
        if vertex[2] <= 0:
            moment_lz += np.linalg.norm(vertex - ori)
        else:
            moment_rz += np.linalg.norm(vertex - ori)
    if moment_lz < moment_rz:  # right side of z axis should be the moment higher side
        transform = trimesh.geometry.align_vectors([0, 0, 1], [0, 0, -1])
        mesh.apply_transform(transform)

    for vertex in mesh.vertices:
        if vertex[0] <= 0:
            moment_lx += np.linalg.norm(vertex - ori)
        else:
            moment_rx += np.linalg.norm(vertex - ori)
    if moment_lx < moment_rx: #right side of x axis should be the moment higher side
        transform = trimesh.geometry.align_vectors([1, 0, 0], [-1, 0, 0])
        mesh.apply_transform(transform)


    for vertex in mesh.vertices:
        if vertex[1] <= 0:
            moment_ly += np.linalg.norm(vertex - ori)
        else:
            moment_ry += np.linalg.norm(vertex - ori)
    if moment_ly < moment_ry:  # right side of y axis should be the moment higher side
        transform = trimesh.geometry.align_vectors([0, 1, 0], [0, -1, 0])
        mesh.apply_transform(transform)



    print('Flipping done')
    print('Barycenter:', mesh.center_mass)
    print('The size of the bounding box(length,width,height):', mesh.bounding_box_oriented.primitive.extents)
    mesh.show()

    # Scaling

    maxLengthOfSide = max(mesh.bounding_box_oriented.primitive.extents)
    mesh.apply_scale(1 / maxLengthOfSide)

    print('Scaling done')
    print('Barycenter:', mesh.center_mass)
    print('The size of the bounding box(length,width,height):', mesh.bounding_box_oriented.primitive.extents, "\n")
    mesh.show()


Normalization('/Users/darkqian/PycharmProjects/MR/benchmark/db/0/m6/m6.off')
