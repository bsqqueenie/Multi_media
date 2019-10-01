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


def normalization(meshList, avgVolumn=1):
    newMeshList = []
    count = 1
    for eachMesh in meshList:
        # meshName = trimesh.load_mesh(filepath)
        # create a matrix for tanslation to the [0,0,0]

        maxLengthOfSide = max(eachMesh.bounding_box_oriented.primitive.extents)
        eachMesh.apply_scale(avgVolumn/maxLengthOfSide)
        normalizationOfCoordinates = []
        for eachValue in eachMesh.center_mass:
            if eachValue <= 0:
                normalizationOfCoordinates.append(eachValue)
            else:
                normalizationOfCoordinates.append(-np.absolute(eachValue))

        eachMesh.apply_translation(normalizationOfCoordinates)  # move the tresh to the origin
        newMeshList.append(eachMesh)
        # print(count)
        count = count + 1
    return newMeshList


def scanDB2(path, Classdic):  # input the root paht, not enter the path for one specific file
    import os
    stacks = np.array(['className', 'fileName', 'verticeNumber', 'faceNumber', 'bounding_box_volume',"faceType"])
    count = 1
    sumOfVolume = 0
    meshList = []
    with open('issue_final.txt','a') as txt:
        for root, dirs, files in os.walk(path):
            for eachFile in files:  # eachFile indicates the filename e.g m110.off
                if eachFile.endswith(".off") or eachFile.endswith(".ply"):  # return .off file and its last folder
                    filePath = os.path.join(root, eachFile)  # return the complete path of mesh
                    fileIndex = os.path.splitext(eachFile)[0].strip("m")
                    label = findClass(Classdic, fileIndex)



                    # print(dataForSingleFile)
                    mesh = trimesh.load_mesh(filePath)
                    try:
                        dataForSingleFile = [label, eachFile]

                        for eachAttributes in Meshfilter(mesh):
                            dataForSingleFile.append(eachAttributes)
                        dataForSingleFile.append("onlyTriangles")
                        sumOfVolume = sumOfVolume + Meshfilter(mesh)[2]
                        # print(dataForSingleFile)
                        stacks = np.vstack((stacks, dataForSingleFile))
                        # print(count)
                        count = count + 1

                        if Meshfilter(mesh)[0] >= 100 and Meshfilter(mesh)[1] >= 100:
                            meshList.append(mesh)

                        elif Meshfilter(mesh)[0] < 100 or Meshfilter(mesh)[1] < 100:
                            txt.write(filePath)
                            txt.write('\n')
                            # print(filePath)
                            # print(mesh.bounding_box_oriented.volume,mesh.faces.shape[0],mesh.vertices.shape[0],mesh.center_mass)
                    except:

                        txt.write(filePath,'new error')

                avgVolume = sumOfVolume / count
    return stacks, meshList, avgVolume


testPath = '/Users/jack/Desktop/privateStuff/UUstuff/2019-2020/period1/MR/assignment/benchmark/classification/v1/coarse1/coarse1Test.cla'
trainPath = '/Users/jack/Desktop/privateStuff/UUstuff/2019-2020/period1/MR/assignment/benchmark/classification/v1/coarse1/coarse1Train.cla'

print('----process CLA files----')
test = parseCla(testPath, False)
train = parseCla(trainPath, False)
new = merge2dicts(test, train)  # the merged data of test and train
#
#
print('----process whole benchmark----')
path1 = '/Users/jack/Desktop/privateStuff/UUstuff/2019-2020/period1/MR/assignment/benchmark/db'  # the number of file is 1813, nr.1693 was removed beforehand
# path1="/Users/jack/Desktop/privateStuff/UUstuff/2019-2020/period1/MR/assignment/benchmark/refine/1"
stacks1, meshList1, avgOfVolumn1 = scanDB2(path1, new)
print("numberOMesh",len(meshList1),len(stacks1))
pd.DataFrame(stacks1).to_csv("file_benchmark_before_refinement.csv", header=False, index=False)
#
# print(meshList1[0].bounding_box_oriented.volume,meshList1[0].faces.shape[0],meshList1[0].vertices.shape[0])





print('----process unqualified meshes----')
# # refine unqualified meshes, the refined meshed will be stored at the outputpath
issuePath='/Users/jack/Desktop/privateStuff/UUstuff/2019-2020/period1/MR/assignment/issue_final.txt'
outputpath='/Users/jack/Desktop/privateStuff/UUstuff/2019-2020/period1/MR/assignment/benchmark/refine/1'
javaFilePath= '/Users/jack/Desktop/privateStuff/UUstuff/2019-2020/period1/MR/assignment/catmullclark.jar'
refineMesh(issuePath,outputpath,javaFilePath)


# process the refined meshes
print('----process refined meshes----')
path2 = '/Users/jack/Desktop/privateStuff/UUstuff/2019-2020/period1/MR/assignment/benchmark/refine/1'
stacks2, meshList2, avgOfVolumn2 = scanDB2(path2, new)
print("numberOMesh",len(meshList2),len(stacks2))
# print("numberOfrefinedMesh",len(meshList2),len(stacks2))
# for i in meshList2:
#     print(i.bounding_box_oriented.volume,i.faces.shape[0],i.vertices.shape[0],i.center_mass)




# combine all data
print('----combine all data----')
print(len(meshList1),len(meshList2))

# replace the old values with refined values

refinedValues = pd.DataFrame(data=stacks2[1:,0:],columns=stacks2[0,0:])
oldValues = pd.DataFrame(data=stacks1[1:,0:],columns=stacks1[0,0:])
for eachRefinedFile in refinedValues['fileName']:
    oldValues.loc[oldValues['fileName'] == eachRefinedFile] = refinedValues.loc[refinedValues['fileName'] == eachRefinedFile].values

# store the refined data to csv

pd.DataFrame(oldValues).to_csv("file_benchmark_after_refinement.csv", header=True, index=False)




finalMeshList=meshList1+meshList2
print("numberOfMesh",len(finalMeshList))

print('---- before Nomalization ----')
print(finalMeshList[5].bounding_box_oriented.volume, finalMeshList[5].faces.shape[0], finalMeshList[5].vertices.shape[0], finalMeshList[5].center_mass)

print('---- Nomalization and return the new mesh list ----')

fianlAvgVol = 1
newMeshList = normalization(finalMeshList, fianlAvgVol)
# print(finalMeshList[0].bounding_box_oriented.volume, finalMeshList[1].bounding_box_oriented.volume)

print('---- after Nomalization ----')
print(newMeshList[5].bounding_box_oriented.volume, newMeshList[5].faces.shape[0], newMeshList[5].vertices.shape[0], newMeshList[5].center_mass)

print("the amount of mesh after normalization:", len(newMeshList))

