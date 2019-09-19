import trimesh
import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
import networkx as nx


#scan the whole database
def scanDB(path): #input the root paht
    import os
    stacks = np.array(['className','fileName','verticeNumber','faceNumber','bounding_box_volume'])
    files = os.listdir(path)
    count = 1
    for lables in files:
        try:
            lable = os.listdir(path + '/' + lables)
        except:
            continue
        for file in lable:
            if not os.path.isdir(file) and not os.path.splitext(file)[-1] == '.txt':
                f = path + '/' + lables + "/" + file  #file path
                dataForSingleFile = [lables, file]
                for eachAttributes in Meshfilter(f):
                    dataForSingleFile.append(eachAttributes)
                # print(dataForSingleFile)
                stacks = np.vstack((stacks, dataForSingleFile))
                print(count)
                count  = count+1
    return stacks  # in the end we got the data containing : className,fileName,vertice number,face number,bounding box volume

def Meshfilter(filepath):
    mesh = trimesh.load_mesh(filepath)
    vertice=mesh.vertices.shape[0]
    faces=mesh.faces.shape[0]
    Bounding_box_volume = mesh.bounding_box_oriented.volume
    if faces < 100 or vertice< 100:
        print(filepath,' ',"problem file")
    return vertice,faces,Bounding_box_volume








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
path='/Users/jack/Desktop/privateStuff/UUstuff/2019-2020/period1/MR/assignment/labeledDb/LabeledDB_new'
dataframe = scanDB(path)
pd.DataFrame(dataframe).to_csv("file.csv",header=False,index=False)



# read the csv and calculate the average bounding-box volumn

csvPath= "/Users/jack/Desktop/privateStuff/UUstuff/2019-2020/period1/MR/assignment/file.csv"
data= pd.read_csv(csvPath)
avgVol=np.average(data.iloc[:,-1])

