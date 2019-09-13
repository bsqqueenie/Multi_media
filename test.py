import trimesh
import numpy as np
import networkx as nx
offPath='/Users/jack/Desktop/privateStuff/UUstuff/2019-2020/period1/MR/assignment/benchmark/db/0/m0/m0.off'
plyPath='/Users/jack/Desktop/privateStuff/UUstuff/2019-2020/period1/MR/m0.off.ply'
mesh = trimesh.load_mesh(plyPath)
# # print(len(mesh.edges))
# mesh.show()



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
    path.append(nx.shortest_path(g,source=each_edge[0],target=each_edge[1],weight='length'))

# VISUALIZE RESULT
# make the sphere white
mesh.visual.face_colors = [255,255,255,255]
# # Path3D with the path between the points

path_visual = []

for eachPath in path:
    path_visual.append(trimesh.load_path(mesh.vertices[eachPath]))


# create a scene with the mesh, path, and points
scene = trimesh.Scene([path_visual, mesh ])

scene.show()