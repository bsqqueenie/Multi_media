import trimesh
import numpy as np
import networkx as nx
import pandas as pd
import subprocess
import os
# #
# offPath='/Users/jack/Desktop/privateStuff/UUstuff/2019-2020/period1/MR/assignment/benchmark/db/0/m94/m94.off'
# plyPath='/Users/jack/Desktop/privateStuff/UUstuff/2019-2020/period1/MR/m0.off.ply'
# mesh = trimesh.load_mesh(offPath)



def refineMesh(issueFileTxt,outPutPath,jarPath):  # this function is for those meshes that have less than 100 faces and vertices
    count = 1
    txt=open(issueFileTxt,'r')
    for eachPath in txt.readlines():
        inputpath = eachPath.strip('\n')
        refineFilePath = os.path.join(outPutPath,os.path.basename(inputpath))
        subprocess.call(['java', '-jar', jarPath, inputpath, refineFilePath])
        # print(count)
        count=count+1
    print("finished")



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




def Meshfilter(mesh):
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
                    
                    
                    
                    # print(dataForSingleFile)
                    mesh = trimesh.load_mesh(filePath)
                    try:
                        if Meshfilter(mesh)[0] >= 100 and Meshfilter(mesh)[1] >= 100:
                            dataForSingleFile = [label, eachFile]
                            meshList.append(mesh)
                            for eachAttributes in Meshfilter(mesh):
                                dataForSingleFile.append(eachAttributes)
                            # print(count)
                            sumOfVolumn = sumOfVolumn + Meshfilter(mesh)[2]
                            # print(dataForSingleFile)
                            stacks = np.vstack((stacks, dataForSingleFile))
                            count = count + 1
                        else:
                            txt.write(filePath)
                            txt.write('\n')
                    except:
                        
                        txt.write(filePath,'new error')
        
        
        
        
        
        
        
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
                           # print(count)
        count = count + 1
    return newMeshList


testPath = '/Users/jack/Desktop/privateStuff/UUstuff/2019-2020/period1/MR/assignment/benchmark/classification/v1/coarse1/coarse1Test.cla'
trainPath = '/Users/jack/Desktop/privateStuff/UUstuff/2019-2020/period1/MR/assignment/benchmark/classification/v1/coarse1/coarse1Train.cla'

print('----process CLA files----')
test = parseCla(testPath, False)
train = parseCla(trainPath, False)
new = merge2dicts(test, train)  # the merged data of test and train
#
#
print('----process whole benchmark----')
path1 = '/Users/jack/Desktop/privateStuff/UUstuff/2019-2020/period1/MR/assignment/benchmark/db'
stacks, meshList, avgOfVolumn = scanDB2(path1, new)
print("numberOMesh",len(meshList),len(stacks))
#

print('----process unqualified meshes----')
# # refine unqualified meshes, the refined meshed will be stored at the outputpath
issuePath='/Users/jack/Desktop/privateStuff/UUstuff/2019-2020/period1/MR/assignment/issue.txt'
outputpath='/Users/jack/Desktop/privateStuff/UUstuff/2019-2020/period1/MR/assignment/benchmark/refine/'
javaFilePath= '/Users/jack/Desktop/privateStuff/UUstuff/2019-2020/period1/MR/assignment/catmullclark.jar'
refineMesh(issuePath,outputpath,javaFilePath)


# process the refined meshes
print('----process refined meshes----')
path2 = '/Users/jack/Desktop/privateStuff/UUstuff/2019-2020/period1/MR/assignment/benchmark/refine'
stacks2, meshList2, avgOfVolumn2 = scanDB2(path2, new)

print("numberOfrefinedMesh",len(meshList2),len(stacks2))

# combine all data
print('----combine all data----')
print(len(stacks),len(stacks2))
alldata=np.vstack((stacks,stacks2))
fianlAvgVol=(avgOfVolumn+avgOfVolumn2)/2
finalMeshList=meshList+meshList2

print('----Nomalization and store data to csv----')
newMeshList = normalization(finalMeshList, fianlAvgVol)
print(finalMeshList[0].bounding_box_oriented.volume, finalMeshList[1].bounding_box_oriented.volume)
pd.DataFrame(alldata).to_csv("file_benchmark_final.csv", header=False, index=False)
