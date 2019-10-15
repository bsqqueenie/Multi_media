import random
import subprocess
import os
import itertools
import trimesh
import numpy as np
import pandas as pd
import math
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# #
# offPath='/Users/jack/Desktop/privateStuff/UUstuff/2019-2020/period1/MR/assignment/benchmark/db/0/m94/m94.off'
# plyPath='/Users/jack/Desktop/privateStuff/UUstuff/2019-2020/period1/MR/m0.off.ply'
# mesh = trimesh.load_mesh(offPath)


# this function is for refining meshes containing vertices or faces under 100

def cleanOffMesh(cleanOFfListPath,outPutPath,jarPath,threshold = 7000):  # this function is for those meshes that have less than 100 faces and vertices
    txt=open(cleanOFfListPath,'r')
    distance = 0.005
    for eachPath in txt.readlines():
        inputpath = eachPath.strip('\n')
        refineFilePath = os.path.join(outPutPath,os.path.basename(inputpath))
        subprocess.call(['java', '-jar', jarPath, inputpath, refineFilePath, str(distance)])
        mesh_after = trimesh.load_mesh(refineFilePath)
        print(os.path.basename(inputpath), '的顶点数和面数分别为:{},{}'.format(len(mesh_after.vertices), len(mesh_after.faces)))
        while len(mesh_after.vertices)>threshold :
            print("entering looping for cleaningoff")
            subprocess.call(['java', '-jar', jarPath, refineFilePath, refineFilePath, str(distance)])
            distance = distance + 0.001
            mesh_after = trimesh.load_mesh(refineFilePath)
            mesh_after.remove_duplicate_faces()
            print(os.path.basename(inputpath),'的顶点数和面数分别为:{},{}'.format(len(mesh_after.vertices),len(mesh_after.faces)))


# this function is for refining meshes containing vertices or faces above 100
def refineMesh(issueFileTxt,outPutPath,jarPath,threshold=1000):  # this function is for those meshes that have less than 100 faces and vertices
    import subprocess
    import os
    txt=open(issueFileTxt,'r')
    for eachPath in txt.readlines():
        inputpath = eachPath.strip('\n')
        refineFilePath = os.path.join(outPutPath,os.path.basename(inputpath))
        subprocess.call(['java', '-jar', jarPath, inputpath, refineFilePath])
        mesh_after = trimesh.load_mesh(refineFilePath)
        print(os.path.basename(inputpath), '的顶点数和面数分别为:{},{}'.format(len(mesh_after.vertices), len(mesh_after.faces)))
        while len(mesh_after.vertices)<threshold or len(mesh_after.faces)<threshold  :
            print("entering looping for refining")
            subprocess.call(['java', '-jar', jarPath, refineFilePath, refineFilePath])
            mesh_after = trimesh.load_mesh(refineFilePath)
            mesh_after.remove_duplicate_faces()
            print(os.path.basename(inputpath), '的顶点数和面数分别为:{},{}'.format(len(mesh_after.vertices), len(mesh_after.faces)))







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


def scanDB2(path, Classdic, cleanMeshMode=True,min=1000,max=7000):  # input the root paht, not enter the path for one specific file
    
    stacks = np.array(['className', 'fileName', 'verticeNumber', 'faceNumber', 'bounding_box_volume',"faceType"])
    qualifiedStack = np.array(['className', 'fileName', 'verticeNumber', 'faceNumber', 'bounding_box_volume',"faceType"])
    count = 1
    meshList = [] # for problematic meshes
    with open('issue_final.txt','a') as txt:
        for root, dirs, files in os.walk(path):
            for eachFile in files:  # eachFile indicates the filename e.g m110.off
                if eachFile.endswith(".off") or eachFile.endswith(".ply"):  # return .off file and its last folder
                    filePath = os.path.join(root, eachFile)  # return the complete path of mesh
                    fileIndex = os.path.splitext(eachFile)[0].strip("m")
                    label = findClass(Classdic, fileIndex)
                    # print(dataForSingleFile)
                    mesh = trimesh.load_mesh(filePath)
                    mesh.remove_duplicate_faces()
                    
                    try:
                        dataForSingleFile = [label, eachFile]
                        
                        for eachAttributes in Meshfilter(mesh):
                            dataForSingleFile.append(eachAttributes)
                        dataForSingleFile.append("onlyTriangles")
                        # print(dataForSingleFile)
                        stacks = np.vstack((stacks, dataForSingleFile))
                        
                        if not cleanMeshMode:
                            meshList.append([eachFile, mesh])
            
                if cleanMeshMode:
                    
                    # if min <= Meshfilter(mesh)[0] < max and min <= Meshfilter(mesh)[1] < max:
                    #     meshList.append([eachFile, mesh])
                    #     qualifiedStack = np.vstack((qualifiedStack, dataForSingleFile))
                    #     txt2.write(filePath)
                    #     txt2.write('\n')
                    #
                    # elif Meshfilter(mesh)[0] > max or Meshfilter(mesh)[1] > max :
                    #     txt1.write(filePath)
                    #     txt1.write('\n')
                    
                    if Meshfilter(mesh)[0] < min or Meshfilter(mesh)[1] < min:
                        
                        txt.write(filePath)
                            txt.write('\n')
                            else:
                                meshList.append([eachFile, mesh])
                                qualifiedStack = np.vstack((qualifiedStack, dataForSingleFile))
                # print(filePath)
                # print(mesh.bounding_box_oriented.volume,mesh.faces.shape[0],mesh.vertices.shape[0],mesh.center_mass)
                
                
                except:
                    print(filePath + ' ' + "new_error")
                        txt.write(filePath + ' ' + "new_error")
                        txt.write('\n')
                    
                    print(count)
                    count = count + 1



return stacks,meshList,qualifiedStack

def translation(mesh, center):
    i = 0
    for vertex in mesh.vertices:
        mesh.vertices[i] = vertex - center
        i += 1
    return mesh


def normalization_1(meshList, avgVolumn=1):
    newMeshList = []
    count = 1
    for eachMesh in meshList:
        # meshName = trimesh.load_mesh(filepath)
        # create a matrix for tanslation to the [0,0,0]
        
        maxLengthOfSide = max(eachMesh.bounding_box_oriented.primitive.extents)
        eachMesh.apply_scale(avgVolumn / maxLengthOfSide)
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

def normalization(meshList):
    newMeshList = []
    count = 1
    for each in meshList:
        
        mesh = each[1]
        
        
        '''
            # meshName = trimesh.load_mesh(filepath)
            # create a matrix for tanslation to the [0,0,0]
            
            numOfvertives = mesh.vertices.shape[0]
            numOffaces = mesh.faces.shape[0]
            print('Original')
            print('Number of vertices and faces of the mesh:', numOfvertives, numOffaces)
            print('Barycenter:', mesh.center_mass)
            print('The size of the bounding box(length,width,height):', mesh.bounding_box_oriented.primitive.extents, "\n")
            mesh.show()
            
            '''
        
        # Centering
        ori = [0, 0, 0]
        center = sum(mesh.vertices) / mesh.vertices.shape[0]
        mesh.apply_translation(ori - center)
        center = sum(mesh.vertices) / mesh.vertices.shape[0]
        Dis = np.linalg.norm(center - ori)
        
        while (Dis >= 0.05):
            mesh.apply_translation(ori - center)  # move the mesh to the originz
            center = sum(mesh.vertices) / mesh.vertices.shape[0]
            Dis = np.linalg.norm(center - ori)
        
        '''
            print('Centering done')
            print('Barycenter:', mesh.center_mass)
            print('The size of the bounding box(length,width,height):', mesh.bounding_box_oriented.primitive.extents, "\n")
            mesh.show()
            '''
        
        
        # Alignment
        
        pca = PCA(n_components=2)
        Reduced_mesh = pca.fit_transform(mesh.vertices)
        # print(pca.components_)
        
        transform_x = trimesh.geometry.align_vectors(pca.components_[0], [1, 0, 0])
        mesh.apply_transform(transform_x)
        Reduced_mesh_newx = pca.fit_transform(mesh.vertices)
        
        transform_y = trimesh.geometry.align_vectors(pca.components_[1], [0, 1, 0])
        mesh.apply_transform(transform_y)
        Reduced_mesh_newy = pca.fit_transform(mesh.vertices)
        
        '''
            print(pca.components_)
            print('Alignment done')
            print('Barycenter:', mesh.center_mass)
            print('The size of the bounding box(length,width,height):', mesh.bounding_box_oriented.primitive.extents, "\n")
            mesh.show()
            '''
        # Flipping
        
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
        if moment_lx < moment_rx:  # right side of x axis should be the moment higher side
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
        
        '''
            print('Flipping done')
            print('Barycenter:', mesh.center_mass)
            print('The size of the bounding box(length,width,height):', mesh.bounding_box_oriented.primitive.extents)
            mesh.show()
            '''
        
        # Scaling
        
        
        maxLengthOfSide = max(mesh.bounding_box_oriented.primitive.extents)
        mesh.apply_scale(1 / maxLengthOfSide)
        '''
            print('Scaling done')
            print('Barycenter:', mesh.center_mass)
            print('The size of the bounding box(length,width,height):', mesh.bounding_box_oriented.primitive.extents, "\n")
            mesh.show()
            '''
        
        newMeshList.append([each[0],mesh]) # store filename and mesh object
        # print(count)
        count = count + 1
    
    return newMeshList


# 3.2 A
def feature_extraction(meshList): # 传入 normalize 过后的mesh
    counter = 1
    featureStacks = np.array(['fileName','surfaceArea','compactness','boundingBoxVolume','diameter','eccentricity'])
    
    # shapePropertyStacks = np.array(['A3', 'D1', 'D2', 'D3', 'D4'])
    for each in meshList:
        
        try:
            
            mesh = each[1]
            featureofEachMesh =[]
            # surface area
            surfaceArea = mesh.area    # equivalent to sum(mesh.area_faces)
            
            # compactness(with respect to a sphere)
            
            # C = A ^3 / V^2 For 3d
            
            sphereVolume = mesh.bounding_sphere.volume
            
            compactness = surfaceArea**3 / sphereVolume**2
            
            # axis - aligned bounding - box volume
            
            boundingBoxVolume = mesh.bounding_box.volume  # 是否需要用mesh.bounding_box_oriented.volume ？
            
            
            # diameter  (the diameter of minimun bounding sphere)
            
            diameter = 2*pow(3*mesh.bounding_sphere.volume  / (4*math.pi),1/3)
            
            
            # eccentricity(ratio of largest to smallest eigenvalues of covariance matrix)
            
            
            pca = PCA()
            Reduced_mesh = pca.fit_transform(mesh.vertices)
            covarianceMatrix = pca.get_covariance()
            values, vectors = np.linalg.eig(covarianceMatrix)
            eccentricity = max(values)/min(values)
            featureofEachMesh.extend([each[0],surfaceArea, compactness, boundingBoxVolume, diameter, eccentricity])
            featureStacks = np.vstack((featureStacks, featureofEachMesh))
            print(counter)
            counter = counter + 1
        # print("surface area is {}, compactness is {},axis-aligned-bounding-box volume is {},diameter is {},eccentricity is {}".format(surfaceArea,compactness,boundingBoxVolume,diameter,eccentricity))
        except:
            print(each[1], "error with this file")

return featureStacks



def shape_property(meshList, bins=8, showPlot=False):
    meshCounter = 1
    
    allFeature = []
    for each in meshList:
        mesh = each[1]
        A3holder = []
        D1holder = []
        D2holder = []
        D3holder = []
        D4holder = []
        
        for i in range(len(mesh.vertices)):
            
            
            
            # print(i)
            
            vertex = mesh.vertices[i]  # 获取中心点
            
            # A3:angle between 3 random vertices
            
            while True: # 要在去除当前选中的中心点之外的顶点中不重复选取两点
                index = random.sample(range(len(mesh.vertices)), 2)
                if (index[0] != i and index[1] != i):
                    break
            A3holder.append(angle(mesh.vertices[index[0]], vertex, mesh.vertices[index[1]]).item())
            
            
            # D1 : distance between barycenter and random vertex
            
            barycenter = mesh.center_mass.tolist()
            D1holder.append(pow(sum((barycenter - vertex) ** 2), 1 / 2).item())  # compute the distance between barycenter and the random vertex
            
            # D2: distance between 2 random vertices
            
            while True: # 要在去除当前选中的中心点之外的顶点中不重复选取两点
                index = random.sample(range(len(mesh.vertices)), 1)
                if (index[0] != i):
                    break
            v1 = vertex
            v2 = mesh.vertices[index][0]
            D2holder.append(float(pow(sum((v1 - v2) ** 2), 1 / 2)))
            
            # D3:  square root of area of triangle given by 3 random vertices
            while True:  # 要在去除当前选中的中心点之外的顶点中不重复选取两点
                index = random.sample(range(len(mesh.vertices)), 2)
                if (index[0] != i and index[1] != i):
                    break
                    areaofTriangle = area(mesh.vertices[index][0], mesh.vertices[index][1], vertex)
                    D3holder.append(float(areaofTriangle ** (1 / 2)))
                    
                    
                    # D4: cube root (pow(a,1/3)) of volume of tetrahedron formed by 4 random vertices
                    while True:  # 要在去除当前选中的中心点之外的顶点中不重复选取两点
                        index = random.sample(range(len(mesh.vertices)), 3)
                            if (index[0] != i and index[1] != i and index[2] !=i):
                                break
                                    volume = tetrahedron_calc_volume(mesh.vertices[index][0], mesh.vertices[index][1], mesh.vertices[index][2],
                                                                     vertex)
                                        D4holder.append(float(volume ** (1 / 3)))
                                            
                                            
                                            
                                            #
                                            A3counts, A3x, A3y = plt.hist(A3holder, bins=bins)  # bins needs to be defined.
                                                plt.close()
                                                    D1counts, D1x, D1y = plt.hist(D1holder, bins=bins)  # bins needs to be defined.
                                                        plt.close()
                                                            D2counts, D2x, D2y = plt.hist(D2holder, bins=bins)  # bins needs to be defined.
                                                                plt.close()
                                                                    D3counts, D3x, D3y = plt.hist(D3holder, bins=bins)  # bins needs to be defined.
                                                                        plt.close()
                                                                            D4counts, D4x, D4y = plt.hist(D4holder, bins=bins)  # bins needs to be defined.
                                                                                plt.close()
                                                                                    #
                                                                                    
                                                                                    allDataForSingleMesh = [A3counts.tolist(), D1counts.tolist(), D2counts.tolist(), D3counts.tolist(), D4counts.tolist()]
                                                                                        merged = list(itertools.chain(*allDataForSingleMesh))
                                                                                            allFeature.append([each[0],merged])
                                                                                                if meshCounter%10 == 0 or meshCounter == len(meshList):
                                                                                                    print(meshCounter,'len of stack:',' ', len(allFeature))
                                                                                                        # print(meshCounter)
                                                                                                        meshCounter = meshCounter + 1

if showPlot:
    A3counts, A3x, A3y = plt.hist(A3holder, bins=bins)  # bins needs to be defined.
        print("A3 counts", A3counts)
        plt.title("angle between 3 random vertices")
        plt.show()
        
        D1counts, D1x, D1y = plt.hist(D1holder, bins=bins)  # bins needs to be defined.
        print("D1 counts", D1counts)
        plt.title("distance between barycenter and random vertex")
        plt.show()
        
        D2counts, D2x, D2y = plt.hist(D2holder, bins=bins)  # bins needs to be defined.
        print("D2 counts", D2counts)
        plt.title("distance between 2 random vertices")
        plt.show()
        
        D3counts, D3x, D3y = plt.hist(D3holder, bins=bins)  # bins needs to be defined.
        print("D3 counts", D3counts)
        plt.title("square root of area of triangle given by 3 random vertices")
        plt.show()
        
        D4counts, D4x, D4y = plt.hist(D4holder, bins=bins)  # bins needs to be defined.
        print("D4 counts", D4counts)
        plt.title("cube root of volume of tetrahedron formed by 4 random vertices")
        plt.show()
    
    return allFeature



def area(a, b, c):  # return the area of a triangle given three points
    from numpy.linalg import norm
    return 0.5 * norm(np.cross(b - a, c - a))


# those 3 following functions are for computing volume of tetrahedron formed by 4 random vertices

def determinant_3x3(m):
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
            m[1][0] * (m[0][1] * m[2][2] - m[0][2] * m[2][1]) +
            m[2][0] * (m[0][1] * m[1][2] - m[0][2] * m[1][1]))


def subtract(a, b):
    return (a[0] - b[0],
            a[1] - b[1],
            a[2] - b[2])


def tetrahedron_calc_volume(a, b, c, d):
    return (abs(determinant_3x3((subtract(a, b),
                                 subtract(b, c),
                                 subtract(c, d),
                                 ))) / 6.0)


def angle(randomPoint1, vertex, randomPoint2):  # calculate the angle given 3 3d points
    
    b = vertex
    a = randomPoint1
    c = randomPoint2
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)


testPath = '/Users/jack/Desktop/privateStuff/UUstuff/2019-2020/period1/MR/assignment/benchmark/classification/v1/coarse1/coarse1Test.cla'
trainPath = '/Users/jack/Desktop/privateStuff/UUstuff/2019-2020/period1/MR/assignment/benchmark/classification/v1/coarse1/coarse1Train.cla'

print('----process CLA files----')
test = parseCla(testPath, False)
train = parseCla(trainPath, False)
new = merge2dicts(test, train)  # the merged data of test and train

print('----process whole benchmark----')
path1 = '/Users/jack/Desktop/privateStuff/UUstuff/2019-2020/period1/MR/assignment/benchmark/db/0'  # the number of file is 1813, nr.1693 was removed beforehand
# path1="/Users/jack/Desktop/privateStuff/UUstuff/2019-2020/period1/MR/assignment/benchmark/refine/1"
stacks1, meshList1,qualifiedStack = scanDB2(path1, new,cleanMeshMode=True)
print("the amount of prcoess of mesh : {},the amount of valid mesh : {}  ".format(len(stacks1)-1,len(meshList1)))
print('----meshed needed resample and cleanoff returned----')


# store the data to csv

pd.DataFrame(stacks1).to_csv("file_benchmark_before_refinement.csv", header=False, index=False)
#






print('----process unqualified meshes (vertices or faces under 1000)----')
# # refine unqualified meshes, the refined meshed will be stored at the outputpath
issuePath='/Users/jack/Desktop/privateStuff/UUstuff/2019-2020/period1/MR/assignment/issue_final.txt'
outputpath='/Users/jack/Desktop/privateStuff/UUstuff/2019-2020/period1/MR/assignment/benchmark/refine/1'
javaFilePath= '/Users/jack/Desktop/privateStuff/UUstuff/2019-2020/period1/MR/assignment/catmullclark.jar'
refineMesh(issuePath,outputpath,javaFilePath)


# print('----process unqualified meshes (vertices or faces above 7000)----')
# # # refine unqualified meshes, the refined meshed will be stored at the outputpath
# cleanOFfListPathtxt = '/Users/jack/Desktop/privateStuff/UUstuff/2019-2020/period1/MR/assignment/cleanOff_list.txt'
# path_cleanOff = '/Users/jack/Desktop/privateStuff/UUstuff/2019-2020/period1/MR/assignment/benchmark/refine/cleanoff'
# cleanOff_jar = '/Users/jack/Desktop/privateStuff/UUstuff/2019-2020/period1/MR/assignment/cleanoff.jar'
# cleanOffMesh(cleanOFfListPathtxt,path_cleanOff,cleanOff_jar)

#
#
# process the refined meshes
print('----process refined meshes----')
path2 = '/Users/jack/Desktop/privateStuff/UUstuff/2019-2020/period1/MR/assignment/benchmark/refine/1'
stacks2, meshList2 ,_= scanDB2(path2, new,cleanMeshMode=False)
# path3 = '/Users/jack/Desktop/privateStuff/UUstuff/2019-2020/period1/MR/assignment/benchmark/refine/cleanoff'
# stacks3, meshList3,_ = scanDB2(path3, new,cleanMeshMode=False)
# print(len(meshList3))

print("the amount of prcoess of Samll mesh : {},the amount of valid mesh {}".format(len(stacks2)-1,len(meshList2)))
# print("the amount of prcoess of Big mesh : {},the amount of valid mesh {}".format(len(stacks3)-1,len(meshList3)))





# combine all data
print('----combine all data----')

# replace the old values with refined values

refinedSmall = pd.DataFrame(data=stacks2[1:,0:],columns=stacks2[0,0:])
pd.DataFrame(refinedSmall).to_csv("refinedSmall.csv", header=False, index=False)
oldValues = pd.DataFrame(data=qualifiedStack[1:,0:],columns=qualifiedStack[0,0:])

combinedDataFrame = pd.concat([oldValues,refinedSmall])
print('combineedDF:', combinedDataFrame.shape)

# store the refined data to csv

pd.DataFrame(combinedDataFrame).to_csv("file_benchmark_after_refinement.csv", header=True, index=False)




# finalMeshList=meshList1+meshList2+meshList3
finalMeshList=meshList1+meshList2

print("the amount of valid meshes",len(finalMeshList))

print('---- before Nomalization ----')
print(finalMeshList[0][1].bounding_box_oriented.volume, finalMeshList[0][1].faces.shape[0], finalMeshList[0][1].vertices.shape[0], finalMeshList[0][1].center_mass)


print('---- Nomalization and return the new mesh list ----')
newMeshList = normalization(finalMeshList)


print('---- after Nomalization ----')
print(finalMeshList[0][1].bounding_box_oriented.volume, finalMeshList[0][1].faces.shape[0], finalMeshList[0][1].vertices.shape[0], finalMeshList[0][1].center_mass)

print("the amount of mesh after normalization:", len(newMeshList))


#feature extraction phase
print('---- global feature extraction phase ----')

globalFeature = feature_extraction(finalMeshList)
data = pd.DataFrame(data=globalFeature[1:,0:],columns=globalFeature[0,0:])
pd.DataFrame(data).to_csv("global_feature.csv", header=True, index=False)

print('---- shape property extraction phase ----')
shapeProperty = shape_property(finalMeshList)

data = pd.DataFrame(shapeProperty)
pd.DataFrame(data).to_csv("shape_features.csv", header=False, index=False)

