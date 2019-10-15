import trimesh
import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
from pandas import Series
import networkx as nx
from sklearn.decomposition import PCA
from scipy import ndimage


ori = [0, 0, 0]

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

    center = sum(mesh.vertices)/mesh.vertices.shape[0]
    mesh.apply_translation(ori-center)
    center = sum(mesh.vertices)/mesh.vertices.shape[0]
    Dis = np.linalg.norm(center - ori)
    print("New center:", center)
    print("Dis:", Dis)

    while(Dis >= 0.05):

        mesh.apply_translation(ori - center)  # move the mesh to the originz
        center = sum(mesh.vertices)/mesh.vertices.shape[0]
        Dis = np.linalg.norm(center - ori)
        print("New center:", center)
        print("Dis:", Dis)

    print('Centering done')
    print('Barycenter:', sum(mesh.vertices)/mesh.vertices.shape[0])
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
        if vertex[2] <= mesh.center_mass[2]:
            moment_lz += np.linalg.norm(vertex - ori)
        else:
            moment_rz += np.linalg.norm(vertex - ori)
    if moment_lz < moment_rz:  # right side of z axis should be the moment higher side
        transform = trimesh.geometry.align_vectors([0, 0, 1], [0, 0, -1])
        mesh.apply_transform(transform)

    for vertex in mesh.vertices:
        if vertex[0] <= mesh.center_mass[0]:
            moment_lx += np.linalg.norm(vertex - ori)
        else:
            moment_rx += np.linalg.norm(vertex - ori)
    if moment_lx < moment_rx: #right side of x axis should be the moment higher side
        transform = trimesh.geometry.align_vectors([1, 0, 0], [-1, 0, 0])
        mesh.apply_transform(transform)


    for vertex in mesh.vertices:
        if vertex[1] <= mesh.center_mass[1]:
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


Normalization('/Users/darkqian/PycharmProjects/MR/benchmark/db/0/m9/m9.off')

def querying(path):
    filename_list = []
    dislist = [0]
    data = pd.read_csv(path)
    norm_data = (data.iloc[:,1:] - data.iloc[:,1:].min()) / (data.iloc[:,1:].max() - data.iloc[:,1:].min())
    #norm_data = (data.iloc[:,1:] - data.iloc[:,1:].mean()) / (data.iloc[:,1:].std())
    new_data = pd.concat([data.iloc[:,0], norm_data], 1)
    print(new_data)
    row = new_data.shape[0]
    Target = new_data.iloc[1785, 1:] #set target model

    for i in range(row):

        Com = new_data.iloc[i, 1:]

        Dis = np.linalg.norm(Target - Com)
        dislist.append(Dis)
    del(dislist[0])
    new_data.insert(0, "Distance", dislist)
    new_data = new_data.sort_values(by = "Distance")
    print(new_data.head(10))

    for i in range(5): #save the target file itself and its 5 top matching
        filename_list.append(new_data.iloc[i,1])

    for j in range(5): #show meshes
        number = float(filename_list[j][1:-4])
        direname1 = number // 100
        mesh = trimesh.load_mesh('/Users/darkqian/PycharmProjects/MR/benchmark/db/'+ str(int(direname1)) + '/m' + str(int(number)) +'/m' + str(int(number)) +'.off')
        mesh.show()


#querying("/Users/darkqian/PycharmProjects/MR/Multi_meadia/feature/allfeature.csv")
#mesh = trimesh.load_mesh('/Users/darkqian/PycharmProjects/MR/benchmark/db/0/m9/m9.off')
#print(sum(mesh.vertices)/mesh.vertices.shape[0])
#mesh.apply_translation((-0.291303,-0.210208,-0.521506))

#barycenter = ndimage.m
#print(barycenter)