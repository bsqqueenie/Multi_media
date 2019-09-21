import trimesh
import numpy as np
import networkx as nx
import pandas as pd


#
# offPath='/Users/jack/Desktop/privateStuff/UUstuff/2019-2020/period1/MR/assignment/benchmark/db/16/m1693/m1693.off'
# plyPath='/Users/jack/Desktop/privateStuff/UUstuff/2019-2020/period1/MR/m0.off.ply'
# mesh = trimesh.load_mesh(offPath)
# mesh.show()

def parseCla(filePath, includeParentClass=False):
    count = 1
    f = open(filePath, "r")
    dicForClass = {}  # store all the classes and filenames
    for i in f.readlines():
        if (len(i.split()) == 3):
            label = i.split()[0]
            # print(label,count)
            if (i.split()[1] != "0") and includeParentClass == True:
                parentLable = i.split()[1]  # store the parent class too
            count = count + 1  # print the number of diff classes

        if (len(i.split()) == 1):
            i = i.strip("\n")
            try:
                dicForClass[label].append(i)
                if includeParentClass:
                    dicForClass[parentLable].append(i)
            except KeyError:
                dicForClass[label] = [i]
                if includeParentClass:
                    dicForClass[parentLable] = [i]
    return dicForClass




def Meshfilter(mesh, filepath):
    vertice = mesh.vertices.shape[0]
    faces = mesh.faces.shape[0]
    Bounding_box_volume = mesh.bounding_box_oriented.volume

    return vertice, faces, Bounding_box_volume

def scanDB2(path, Classdic):  # input the root paht
    import os
    stacks = np.array(['className', 'fileName', 'verticeNumber', 'faceNumber', 'bounding_box_volume'])
    count = 1
    sumOfVolumn = 0
    meshList = []
    with open('issue.txt','a') as txt:
        for root, dirs, files in os.walk(path):
            for eachFile in files:  # eachFile indicates the filename e.g m110.off
                if eachFile.endswith(".off") or eachFile.endswith(".ply"):  # return .off file and its last folder
                    filePath = os.path.join(root, eachFile)  # return the complete path of mesh
                    fileIndex = os.path.splitext(eachFile)[0].strip("m")
                    label = findClass(Classdic, fileIndex)

                    dataForSingleFile = [label, eachFile]

                    # print(dataForSingleFile)
                    mesh = trimesh.load_mesh(filePath)
                    meshList.append(mesh)
                    try:
                        for eachAttributes in Meshfilter(mesh, filePath):
                            dataForSingleFile.append(eachAttributes)
                        if Meshfilter(mesh, filePath)[0] < 100 or Meshfilter(mesh, filePath)[1] < 100:
                            txt.write(filePath)
                            txt.write('\n')
                    except:
                        continue
                        txt.write('————————————')
                        txt.write('\n')
                        txt.write(filePath)
                        txt.write('\n')
                        txt.write('————————————')
                        txt.write('\n')


                    sumOfVolumn = sumOfVolumn + Meshfilter(mesh, filePath)[2]
                    # print(dataForSingleFile)
                    stacks = np.vstack((stacks, dataForSingleFile))
                    print(count)
                    count = count + 1

                avgOfVolumn = sumOfVolumn / count
    return stacks, meshList, avgOfVolumn


def findClass(dictionay, fileIndex):
    for key, value in dictionay.items():
        if (fileIndex in value):
            return key
        else:
            continue



def merge2dicts(dict1, dcit2):
    new = dict.copy(dict1)
    for key, value in dcit2.items():
        if key in new.keys():
            new[key] = new[key] + dcit2[key]
        else:
            new[key] = dcit2[key]

    return new  # return merged dictionary


def normalization(meshList, avgVolumn):
    newMeshList = []
    count = 1
    for eachMesh in meshList:
        # meshName = trimesh.load_mesh(filepath)
        # create a matrix for tanslation to the [0,0,0]
        matrix = np.array([np.absolute(eachMesh.center_mass[0]), np.absolute(eachMesh.center_mass[1]),
                           np.absolute(eachMesh.center_mass[2])])
        eachMesh.apply_translation(matrix)  # move the tresh to the origin
        eachMesh.apply_scale(pow(avgVolumn / eachMesh.bounding_box_oriented.volume, 1 / 3))
        newMeshList.append(eachMesh)
        print(count)
        count = count + 1
    return newMeshList


testPath = '/Users/jack/Desktop/privateStuff/UUstuff/2019-2020/period1/MR/assignment/benchmark/classification/v1/coarse1/coarse1Test.cla'
trainPath = '/Users/jack/Desktop/privateStuff/UUstuff/2019-2020/period1/MR/assignment/benchmark/classification/v1/coarse1/coarse1Train.cla'

test = parseCla(testPath, False)
train = parseCla(trainPath, False)
new = merge2dicts(test, train)  # the merged data of test and train

#
path1 = '/Users/jack/Desktop/privateStuff/UUstuff/2019-2020/period1/MR/assignment/benchmark/db'
stacks, meshList, avgOfVolumn = scanDB2(path1, new)
# newMeshList = normalization(meshList, avgOfVolumn)
# print(newMeshList[0].bounding_box_oriented.volume, newMeshList[1].bounding_box_oriented.volume)
pd.DataFrame(stacks).to_csv("file_benchmark.csv", header=False, index=False)
