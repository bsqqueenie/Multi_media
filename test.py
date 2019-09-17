import trimesh
import tkinter as tk
from tkinter import filedialog
import numpy as np
import networkx as nx


#scan the whole database
def scanDB():

    import os
    path = "/Users/darkqian/PycharmProjects/MR/LabeledDB"
    files = os.listdir(path)
    i = 0
    for lables in files:
        try:
            lable = os.listdir(path + '/' + lables)
        except:
            continue
        for file in lable:
            if not os.path.isdir(file) and not os.path.splitext(file)[-1] == '.txt':
                f = path + '/' + lables + "/" + file
                print(file + ' ' + lables)
                Meshfilter(f)
                #viewMesh(f)
                i += 1

    print(i)

def Meshfilter(filepath):

    mesh = trimesh.load_mesh(filepath)

    print("Number of vertices: ", mesh.vertices.shape[0])
    print("Number of faces: ", mesh.faces.shape[0])
    print("Bounding box volume: ", mesh.bounding_box_oriented.volume)





def viewMesh(filepath):

    #offPath= '/Users/darkqian/PycharmProjects/MR/LabeledDB/Fourleg/397.off'
    #plyPath= '/Users/darkqian/PycharmProjects/MR/benchmark/db/0/m1/m1.off'
    mesh = trimesh.load_mesh(filepath)
    # print(len(mesh.edges))
    '''
    mesh.show()
    
    # edges without duplication
    edges = mesh.edges_unique
    # the actual length of each unique edge
    length = mesh.edges_unique_length
    # create the graph with edge attributes for length
    g = nx.Graph()
    for edge, L in zip(edges, length):
        g.add_edge(*edge, length=L)
    
    
    # run the shortest path query using length for edge weight
    path=[]
    for each_edge in mesh.edges:
        path.append(nx.shortest_path(g, source=each_edge[0], target=each_edge[1], weight='length'))
    
    # VISUALIZE RESULT
    # make the sphere white
    #mesh.visual.face_colors = [255,255,255,255]
    # # Path3D with the path between the points
    
    path_visual = []
    
    for eachPath in path:
        path_visual.append(trimesh.load_path(mesh.vertices[eachPath]))
    
    
    # create a scene with the mesh, path, and points
    scene = trimesh.Scene([path_visual, mesh])
    
    scene.show()
    '''

scanDB()
