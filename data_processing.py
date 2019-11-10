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

#scan the whole database

def scanDB2(path, cleanMeshMode=True,
            max=7000):  # input the root paht, not enter the path for one specific file

    stacks = np.array(['className', 'fileName', 'verticeNumber', 'faceNumber', 'bounding_box_volume', "faceType"])
    qualifiedStack = np.array(
        ['className', 'fileName', 'verticeNumber', 'faceNumber', 'bounding_box_volume', "faceType"])
    count = 1
    meshList = []  # for problematic meshes
    with open('issue_final.txt', 'a') as txt1:
        for root, dirs, files in os.walk(path):
            for eachFile in files:  # eachFile indicates the filename e.g m110.off
                if eachFile.endswith(".off") or eachFile.endswith(".ply"):  # return .off file and its last folder
                    filePath = os.path.join(root, eachFile)  # return the complete path of mesh
                    label = os.path.basename(os.path.dirname(filePath))
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


                            if Meshfilter(mesh)[0] > max : #or Meshfilter(mesh)[1] > max
                                txt1.write(filePath)
                                txt1.write('\n')

                            else:
                                meshList.append([eachFile, mesh])
                                qualifiedStack = np.vstack((qualifiedStack, dataForSingleFile))



                    except:
                        print(filePath + ' ' + "new_error")
                        txt1.write(filePath + ' ' + "new_error")
                        txt1.write('\n')

                    print(count)
                    count = count + 1

    return stacks, meshList, qualifiedStack


def Meshfilter(mesh):
    vertice = mesh.vertices.shape[0]
    faces = mesh.faces.shape[0]
    Bounding_box_volume = mesh.bounding_box_oriented.volume

    return vertice, faces, Bounding_box_volume


def normalization(meshList):
    newMeshList = []
    count = 1
    for each in meshList:

        mesh = each[1]
        # Ori
        ori = [0, 0, 0]


        # Centering

        center = sum(mesh.vertices) / mesh.vertices.shape[0]
        mesh.apply_translation(ori - center)
        center = sum(mesh.vertices) / mesh.vertices.shape[0]
        Dis = np.linalg.norm(center - ori)


        while (Dis >= 0.05):
            mesh.apply_translation(ori - center)  # move the mesh to the originz
            center = sum(mesh.vertices) / mesh.vertices.shape[0]
            Dis = np.linalg.norm(center - ori)




        # Alignment

        pca = PCA(n_components=3)
        Reduced_mesh = pca.fit_transform(mesh.vertices)
        # print(pca.components_)
        a = trimesh.geometry.align_vectors(pca.components_[0], [1, 0, 0], return_angle=True)
        b = trimesh.geometry.align_vectors(pca.components_[0], [-1, 0, 0], return_angle=True)
        transform_x = (trimesh.geometry.align_vectors(pca.components_[0], [1, 0, 0]) if a[1] <= b[
            1] else trimesh.geometry.align_vectors(pca.components_[0], [-1, 0, 0]))
        mesh.apply_transform(transform_x)
        Reduced_mesh_newx = pca.fit_transform(mesh.vertices)
        c = trimesh.geometry.align_vectors(pca.components_[0], [0, 1, 0], return_angle=True)
        d = trimesh.geometry.align_vectors(pca.components_[0], [0, -1, 0], return_angle=True)

        transform_y = (trimesh.geometry.align_vectors(pca.components_[0], [0, 1, 0]) if c[1] <= d[
            1] else trimesh.geometry.align_vectors(pca.components_[0], [0, -1, 0]))
        mesh.apply_transform(transform_y)
        Reduced_mesh_newy = pca.fit_transform(mesh.vertices)





        # Flipping
        center = sum(mesh.vertices) / mesh.vertices.shape[0]
        moment_lx = 0
        moment_rx = 0
        moment_ly = 0
        moment_ry = 0
        moment_lz = 0
        moment_rz = 0
        thread = [0, 0, 1]

        for vertex in mesh.vertices:
            a = trimesh.geometry.align_vectors(vertex, thread, return_angle=True)
            if vertex[0] <= 0:
                moment_lx += np.linalg.norm(vertex - center) * math.sin(a[1])
            else:
                moment_rx += np.linalg.norm(vertex - center) * math.sin(a[1])
        if moment_lx < moment_rx:  # left side of x axis should be the moment higher side
            transform = trimesh.geometry.align_vectors([-1, 0, 0], [1, 0, 0])
            mesh.apply_transform(transform)

        for vertex in mesh.vertices:
            a = trimesh.geometry.align_vectors(vertex, thread, return_angle=True)
            if vertex[1] <= 0:
                moment_ly += np.linalg.norm(vertex - center) * math.sin(a[1])
            else:
                moment_ry += np.linalg.norm(vertex - center) * math.sin(a[1])
        if moment_ly < moment_ry:  # left side of y axis should be the moment higher side
            transform = trimesh.geometry.align_vectors([0, -1, 0], [0, 1, 0])
            mesh.apply_transform(transform)

        for vertex in mesh.vertices:
            a = trimesh.geometry.align_vectors(vertex, thread, return_angle=True)
            if vertex[2] <= 0:
                moment_lz += np.linalg.norm(vertex - center) * math.sin(a[1])
            else:
                moment_rz += np.linalg.norm(vertex - center) * math.sin(a[1])

        if moment_lz < moment_rz:  # right side of z axis should be the moment higher side
            transform = trimesh.geometry.align_vectors([0, 0, -1], [0, 0, 1])
            mesh.apply_transform(transform)

        for vertex in mesh.vertices:
            if vertex[1] <= center[1]:
                moment_ly += np.linalg.norm(vertex - ori)
            else:
                moment_ry += np.linalg.norm(vertex - ori)
        if moment_ly < moment_ry:  # right side of y axis should be the moment higher side
            transform = trimesh.geometry.align_vectors([0, -1, 0], [0, 1, 0])
            mesh.apply_transform(transform)

        # Scaling

        maxLengthOfSide = max(mesh.bounding_box_oriented.primitive.extents)
        mesh.apply_scale(1 / maxLengthOfSide)




        newMeshList.append([each[0],mesh]) # store filename and mesh object
        print(count)
        count = count + 1

    return newMeshList


# 3.2 A
def feature_extraction(meshList): # 传入 normalize 过后的mesh
    counter = 1
    featureStacks=[]
    columns = ['fileName','surfaceArea','compactness','boundingBoxVolume','diameter','eccentricity']
    B_discriptor = ["A3", "D1", "D2", "D3", "D4"]
    for i in range(5):
        count = 1
        while count < 9:
            columns.append('{}_{}'.format(B_discriptor[i], count))
            count = count + 1
        count = 1

    # after this the header of dataframe was obtained.

    for each in meshList:

        try:
            mesh = each[1]
            featureofEachMesh = []
            # surface area
            surfaceArea = mesh.area  # equivalent to sum(mesh.area_faces)

            # compactness(with respect to a sphere)

            # C = A ^3 / V^2 For 3d

            meshVol = TriangleMeshVolume(mesh)

            compactness = surfaceArea ** 3 / meshVol ** 2

            # axis - aligned bounding - box volume

            boundingBoxVolume = mesh.bounding_box.volume  # 是否需要用mesh.bounding_box_oriented.volume ？

            # diameter  (the diameter of minimun bounding sphere)

            diameter = 2 * pow(3 * mesh.bounding_sphere.volume / (4 * math.pi), 1 / 3)

            # eccentricity(ratio of largest to smallest eigenvalues of covariance matrix)

            pca = PCA()
            Reduced_mesh = pca.fit_transform(mesh.vertices)
            covarianceMatrix = pca.get_covariance()
            values, vectors = np.linalg.eig(covarianceMatrix)
            eccentricity = max(values) / min(values)
            featureofEachMesh.extend([each[0], surfaceArea, compactness, boundingBoxVolume, diameter, eccentricity])
            shape_property(mesh, 8, False)
            final = featureofEachMesh + shape_property(mesh,8,False)
            featureStacks.append(final)
            # print(counter)
            counter = counter + 1
        except:
            print(each[0], "error with this file")

    return featureStacks,columns

def shape_property(mesh, bins=8, showPlot=False):

    A3holder = []
    D1holder = []
    D2holder = []
    D3holder = []
    D4holder = []
    barycenter = sum(mesh.vertices) / len(mesh.vertices)
    for i in range(len(mesh.vertices)):

        # print(i)

        vertex = mesh.vertices[i]  # 获取中心点



        # A3:angle between 3 random vertices

        while True:  # 要在去除当前选中的中心点之外的顶点中不重复选取两点
            index = random.sample(range(len(mesh.vertices)), 2)
            if (index[0] != i and index[1] != i):
                break
        A3holder.append(angle(mesh.vertices[index[0]], vertex, mesh.vertices[index[1]]).item())

        # D1 : distance between barycenter and random vertex


        D1holder.append(pow(sum((barycenter - vertex) ** 2),
                            1 / 2).item())  # compute the distance between barycenter and the random vertex

        # D2: distance between 2 random vertices

        while True:  # 要在去除当前选中的中心点之外的顶点中不重复选取两点
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
            if (index[0] != i and index[1] != i and index[2] != i):
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

    allDataForSingleMesh = [A3counts.tolist(), D1counts.tolist(), D2counts.tolist(), D3counts.tolist(),
                            D4counts.tolist()]
    merged = list(itertools.chain(*allDataForSingleMesh))
    # print(meshCounter)

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

    return merged


def SignedTriangleVolume(v1, v2, v3):
    return v1.dot(np.cross(v2,v3)) / 6.0


def TriangleMeshVolume(mesh):

    volume =0

    for eahcTri in mesh.triangles:
        # print(eahcTri[0],eahcTri[1],eahcTri[2])
        volume = volume + SignedTriangleVolume(eahcTri[0],eahcTri[1],eahcTri[2])

    return abs(volume)



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


# this function is for refining meshes containing vertices or faces under 100

def cleanOffMesh(cleanOFfListPath,outPutPath,jarPath,threshold = 7000):  # this function is for those meshes that have less than 100 faces and vertices
    txt=open(cleanOFfListPath,'r')
    distance = 0.005
    refineFilePath = None
    for eachPath in txt.readlines():
        inputpath = eachPath.strip('\n')
        refineFilePath = os.path.join(outPutPath,os.path.basename(inputpath))
        subprocess.call(['java', '-jar', jarPath, inputpath, refineFilePath, str(distance)])
        mesh_after = trimesh.load_mesh(refineFilePath)
        mesh_after.remove_duplicate_faces()

        # print(os.path.basename(inputpath), 'vertices and faces:{},{}'.format(len(mesh_after.vertices), len(mesh_after.faces)))
        while mesh_after.vertices.shape[0] < 1000 or  mesh_after.faces.shape[0] < 1000:
                # print("invalid processing, going back to original mesh")
                distance = distance - 0.001
                subprocess.call(['java', '-jar', jarPath, inputpath, refineFilePath, str(distance)])
                mesh_after = trimesh.load_mesh(refineFilePath)
                mesh_after.remove_duplicate_faces()
                # print(os.path.basename(inputpath),'vertices and faces:{},{}'.format(len(mesh_after.vertices),len(mesh_after.faces)))

        while mesh_after.vertices.shape[0] > threshold and  mesh_after.faces.shape[0] > threshold:
                # print("entering looping for cleaningoff")
                subprocess.call(['java', '-jar', jarPath, refineFilePath, refineFilePath, str(distance)])
                distance = distance + 0.001
                mesh_after = trimesh.load_mesh(refineFilePath)
                mesh_after.remove_duplicate_faces()
                # print(os.path.basename(inputpath),'vertices and faces:{},{}'.format(len(mesh_after.vertices),len(mesh_after.faces)))
    return refineFilePath



def readNewMesh(path, max=7000):  # input the root paht, not enter the path for one specific file

    mesh = trimesh.load_mesh(path)
    mesh.remove_duplicate_faces()

    if Meshfilter(mesh)[0] <= max:
        nomalizedMesh = normalization([[os.path.basename(path), mesh]])
        featureStacks, columnsName = feature_extraction(nomalizedMesh)

    if Meshfilter(mesh)[0] > max:
        with open(cleanMesh, 'w') as txt1:
            txt1.write(path)
        refineMeshPath = cleanOffMesh(cleanMesh, refinedPath, cleanOff_jar)
        mesh  = trimesh.load_mesh(refineMeshPath)
        nomalizedMesh = normalization([[os.path.basename(path), mesh]])
        featureStacks, columnsName = feature_extraction(nomalizedMesh)
    # print(featureStacks)
    return featureStacks


# set the path

DSpath = 'DataSet/LabeledDB'
# location where stores all the refined meshes
refinedPath = 'DataSet/RefinedMeshes'
cleanOff_jar = 'cleanoff.jar'
cleanMesh= 'refined_mesh.txt'
cleanOFFListPathtxt = 'issue_final.txt'


def processDB(dataBasePath,needRefinedMeshPath,cleanOffTool):
    print('----process whole benchmark----')

    stacks1, meshList1, qualifiedStack = scanDB2(dataBasePath, cleanMeshMode=True)
    # store the data to csv
    pd.DataFrame(stacks1).to_csv("csvFiles/small_before_refinement.csv", header=False, index=False)

    print('----process poorly-sampled meshes (vertices above 7000)----')

    # # refine unqualified meshes, the refined meshed will be stored at the outputpath

    path_cleanOff = refinedPath

    cleanOffMesh(needRefinedMeshPath, path_cleanOff, cleanOffTool)
    #
    #
    #
    # # process the refined meshes

    path3 = refinedPath
    stacks3, meshList3, _ = scanDB2(path3, cleanMeshMode=False)

    # combine all data
    print('----combine all data----')

    # replace the old values with refined values

    refinedSmall = pd.DataFrame(stacks3)
    oldValues = pd.DataFrame(qualifiedStack)

    combinedDataFrame = pd.concat([oldValues, refinedSmall])
    print('combineedDF:', combinedDataFrame.shape)

    # store the refined data to csv

    pd.DataFrame(combinedDataFrame).to_csv("csvFiles/small_after_refinement.csv", header=True, index=False)

    finalMeshList = meshList1 + meshList3

    print("the amount of valid meshes", len(finalMeshList))

    print('---- Nomalization and return the new mesh list ----')
    newMeshList = normalization(finalMeshList)

    print("the amount of mesh after normalization:", len(newMeshList))

    # feature extraction phase
    print('---- feature extraction phase ----')

    feature, header = feature_extraction(finalMeshList)
    data = pd.DataFrame(feature, columns=header)
    # pd.DataFrame(data).to_csv("feature.csv", header=True, index=False)

    path2 = 'csvFiles/small_before_refinement.csv'

    all_feature = data
    class_table = pd.read_csv(path2)
    className = []

    for i in all_feature['fileName']:
        className.append(class_table.loc[class_table['fileName'] == i].iloc[:, 0].values[0])

    all_feature.insert(loc=0, column="className", value=className)
    pd.DataFrame(all_feature).to_csv("csvFiles/LPSB_features_final.csv", header=True, index=False)
    print("the whole database has been processed and features have been extracted")



# process the whole LPSB

# processDB(DSpath,cleanOFFListPathtxt,cleanOff_jar)



# process one mesh file in order to extract the features
# path = '/Users/jack/Desktop/personalProjects/Multi_media/DataSet/LabeledDB/Ant/81.off'
# print(readNewMesh(path))