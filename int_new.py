# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'int.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

import trimesh
import math
import numpy as np
import os
import pandas as pd
from sklearn.decomposition import PCA
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QGraphicsView, QGraphicsScene, QApplication, QMessageBox
from annoy import AnnoyIndex
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import preprocessing
import scipy

meshlist = []
Dis_list = []


def Normalization(path):
    ori = [0, 0, 0]
    try:
        mesh = trimesh.load_mesh(path)
    except:
        print("The file doesn't exist")
        return

    # Ori
    # print('特征值：{}\n特征向量：{}'.format(pca.explained_variance_, pca.components_),"\n")
    numOfvertives = mesh.vertices.shape[0]
    numOffaces = mesh.faces.shape[0]
    '''
    print('Original')
    print('Number of vertices and faces of the mesh:', numOfvertives, numOffaces)
    print('Barycenter:', sum(mesh.vertices) / mesh.vertices.shape[0])
    print('The size of the bounding box(length,width,height):', mesh.bounding_box_oriented.primitive.extents, "\n")
    mesh.show()
    '''

    # Centering

    center = sum(mesh.vertices) / mesh.vertices.shape[0]
    mesh.apply_translation(ori - center)
    center = sum(mesh.vertices) / mesh.vertices.shape[0]
    Dis = np.linalg.norm(center - ori)
    '''
    print("New center:", center)
    print("Dis:", Dis)
    '''

    while (Dis >= 0.05):
        mesh.apply_translation(ori - center)  # move the mesh to the originz
        center = sum(mesh.vertices) / mesh.vertices.shape[0]
        Dis = np.linalg.norm(center - ori)
        '''
        print("New center:", center)
        print("Dis:", Dis)
        '''
    '''
    print('Centering done')
    print('Barycenter:', sum(mesh.vertices) / mesh.vertices.shape[0])
    print('The size of the bounding box(length,width,height):', mesh.bounding_box_oriented.primitive.extents, "\n")
    mesh.show()
    '''

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

    '''
    transform_z = trimesh.geometry.align_vectors(pca.components_[2], [0, 0, 1])
    mesh.apply_transform(transform_z)
    Reduced_mesh_newy = pca.fit_transform(mesh.vertices)
    '''

    '''
    print('Alignment done')
    print('Barycenter:', sum(mesh.vertices) / mesh.vertices.shape[0])
    print('The size of the bounding box(length,width,height):', mesh.bounding_box_oriented.primitive.extents, "\n")
    mesh.show()
    '''

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

    '''
    print('Flipping done')
    print('Barycenter:', sum(mesh.vertices) / mesh.vertices.shape[0])
    print('The size of the bounding box(length,width,height):', mesh.bounding_box_oriented.primitive.extents)
    mesh.show()
    '''
    # Scaling

    maxLengthOfSide = max(mesh.bounding_box_oriented.primitive.extents)
    mesh.apply_scale(1 / maxLengthOfSide)
    '''
    print('Scaling done')
    print('Barycenter:', sum(mesh.vertices) / mesh.vertices.shape[0])
    print('The size of the bounding box(length,width,height):', mesh.bounding_box_oriented.primitive.extents, "\n")
    mesh.show()
    '''
    return mesh


def querying(filename, direname):
    global meshlist, Dis_list
    Dis_list = []
    filename_list = []
    dislist = [0]

    data = pd.read_csv("csvFiles/features_final.csv")

    norm_data = (data.iloc[:, 2:] - data.iloc[:, 2:].min()) / (data.iloc[:, 2:].max() - data.iloc[:, 2:].min())

    # norm_data = data.iloc[:,1:] / data.iloc[:,1:].max(axis=0)
    # norm_data = (data.iloc[:,1:] - data.iloc[:,1:].mean()) / (data.iloc[:,1:].std())
    new_data = pd.concat([data.iloc[:, 0:2], norm_data], 1)
    row = new_data.shape[0]
    print(new_data.head(10))
    Target = new_data.loc[new_data["fileName"] == filename]  # set target model

    Target_global = Target.iloc[:, 2:7]
    Target_h1 = Target.iloc[:, 7:15].apply(lambda x: x / 8)
    Target_h2 = Target.iloc[:, 15:23].apply(lambda x: x / 8)
    Target_h3 = Target.iloc[:, 23:31].apply(lambda x: x / 8)
    Target_h4 = Target.iloc[:, 31:39].apply(lambda x: x / 8)
    Target_h5 = Target.iloc[:, 39:47].apply(lambda x: x / 8)
    for i in range(row):
        Com_global = new_data.iloc[i, 2:7]
        Com_h1 = new_data.iloc[i, 7:15].apply(lambda x: x / 8)
        Com_h2 = new_data.iloc[i, 15:23].apply(lambda x: x / 8)
        Com_h3 = new_data.iloc[i, 23:31].apply(lambda x: x / 8)
        Com_h4 = new_data.iloc[i, 31:39].apply(lambda x: x / 8)
        Com_h5 = new_data.iloc[i, 39:47].apply(lambda x: x / 8)

        Dis_h1 = scipy.spatial.distance.cosine(Target_h1.values[0], Com_h1.values)  # Earth mover's distance
        Dis_h2 = scipy.spatial.distance.cosine(Target_h2.values[0], Com_h2.values)
        Dis_h3 = scipy.spatial.distance.cosine(Target_h3.values[0], Com_h3.values)
        Dis_h4 = scipy.spatial.distance.cosine(Target_h4.values[0], Com_h4.values)
        Dis_h5 = scipy.spatial.distance.cosine(Target_h5.values[0], Com_h5.values)

        Dis_global = np.linalg.norm(Target_global - Com_global)  # Euclidean distance
        Dis = (Dis_h1 + Dis_h2 + Dis_h3 + Dis_h4 + Dis_h5 + Dis_global) / 6
        # Dis = scipy.spatial.distance.cosine(Target, Com) #cosine distance
        dislist.append(Dis)

    del (dislist[0])

    new_data.insert(0, "Distance", dislist)
    new_data = new_data.sort_values(by="Distance")
    print(new_data.head(10))
    for i in range(5):  # save the target file itself and its 5 top matching
        filename_list.append(new_data.iloc[i, 2])
    for j in range(5):  # show meshes
        file = new_data.loc[new_data["fileName"] == filename_list[j]]
        number = float(filename_list[j][0:-4])
        direname1 = file.iloc[:, 1].values[0]
        Dis = file.iloc[:, 0].values[0]
        Dis_list.append(Dis)
        mesh = Normalization(
            'DataSet/LabeledDB/' + '/' + str(
                direname1) + '/' + str(int(number)) + '.off')

        meshlist.append(mesh)
    print(Dis_list)
    return meshlist


class Ui_MainWindow(QtWidgets.QWidget):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 548)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setObjectName("pushButton")
        self.verticalLayout_2.addWidget(self.pushButton)
        self.pushButton_7 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_7.setObjectName("pushButton_7")
        self.verticalLayout_2.addWidget(self.pushButton_7)
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.verticalLayout_2.addWidget(self.pushButton_2)
        self.pushButton_8 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_8.setObjectName("pushButton_8")
        self.verticalLayout_2.addWidget(self.pushButton_8)
        self.verticalLayout_6.addLayout(self.verticalLayout_2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setObjectName("pushButton_5")
        self.verticalLayout.addWidget(self.pushButton_5)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_3.addWidget(self.label_2)
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setObjectName("pushButton_4")
        self.verticalLayout_3.addWidget(self.pushButton_4)
        self.horizontalLayout.addLayout(self.verticalLayout_3)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_4.addWidget(self.label_3)
        self.pushButton_6 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_6.setObjectName("pushButton_6")
        self.verticalLayout_4.addWidget(self.pushButton_6)
        self.horizontalLayout.addLayout(self.verticalLayout_4)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_5.addWidget(self.label_4)
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setObjectName("pushButton_3")
        self.verticalLayout_5.addWidget(self.pushButton_3)
        self.horizontalLayout.addLayout(self.verticalLayout_5)
        self.verticalLayout_6.addLayout(self.horizontalLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.pushButton.clicked.connect(self.openfile)
        self.pushButton_2.clicked.connect(self.query)
        self.pushButton_7.clicked.connect(self.showmesh)
        self.pushButton_5.clicked.connect(self.showmesh1)
        self.pushButton_4.clicked.connect(self.showmesh2)
        self.pushButton_6.clicked.connect(self.showmesh3)
        self.pushButton_3.clicked.connect(self.showmesh4)
        self.pushButton_8.clicked.connect(self.t_SNE)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "File select"))
        self.pushButton_7.setText(_translate("MainWindow", "show mesh"))
        self.pushButton_2.setText(_translate("MainWindow", "Query"))
        self.pushButton_8.setText(_translate("MainWindow", "t-SNE"))
        self.label.setText(_translate("MainWindow", " "))
        self.pushButton_5.setText(_translate("MainWindow", "show mesh1"))
        self.label_2.setText(_translate("MainWindow", " "))
        self.pushButton_4.setText(_translate("MainWindow", "show mesh2"))
        self.label_3.setText(_translate("MainWindow", " "))
        self.pushButton_6.setText(_translate("MainWindow", "show mesh3"))
        self.label_4.setText(_translate("MainWindow", " "))
        self.pushButton_3.setText(_translate("MainWindow", "show mesh4"))

    def openfile(self):
        global meshlist
        meshlist = []
        _translate = QtCore.QCoreApplication.translate
        filename, filetype = QFileDialog.getOpenFileName(self, "选取文件", "./",
                                                         "OFF Files (*.off);;PLY Files (*.ply)")
        [dirname, bfilename] = os.path.split(filename)
        print(bfilename)
        meshlist = querying(bfilename, dirname)
        QMessageBox.warning(self,
                            "错误",
                            "Success",
                            QMessageBox.Yes)

    def query(self):
        global Dis_list
        _translate = QtCore.QCoreApplication.translate
        self.label.setText(_translate("MainWindow", "Dis=" + str(Dis_list[1])))
        self.label_2.setText(_translate("MainWindow", "Dis=" + str(Dis_list[2])))
        self.label_3.setText(_translate("MainWindow", "Dis=" + str(Dis_list[3])))
        self.label_4.setText(_translate("MainWindow", "Dis=" + str(Dis_list[4])))

    def showmesh(self):
        global meshlist
        meshlist[0].show()

    def showmesh1(self):
        global meshlist
        meshlist[1].show()

    def showmesh2(self):
        global meshlist
        meshlist[2].show()

    def showmesh3(self):
        global meshlist
        meshlist[3].show()

    def showmesh4(self):
        global meshlist
        meshlist[4].show()

    def t_SNE(self):
        # load the feature
        path = 'csvFiles/LPSB_features_final.csv'
        data = pd.read_csv(path).iloc[:,1:]

        # load the labels and binarize the labels
        path2 = 'csvFiles/small_before_refinement.csv'
        data2 = pd.read_csv(path2)
        classList = []
        for i in pd.read_csv(path)['className'].values:
            classList.append(i)
        lb = preprocessing.LabelBinarizer()
        lb.fit(classList)
        labels = lb.transform(classList)
        # print(feature.shape)
        # print(labels)

        def annQuery(csvData, classSet,
                     queryMesh=100):  # in the end, the queryMesh should be changed to a mesh file or the feature of the mesh
            f = 45
            t = AnnoyIndex(f, 'euclidean')  # Length of item vector that will be indexed

            # nomalize feature data
            features = (csvData.iloc[:, 1:] - csvData.iloc[:, 1:].min()) / (
                    csvData.iloc[:, 1:].max() - csvData.iloc[:, 1:].min())
            for i in range(features.shape[0]):
                v = features.iloc[i]
                t.add_item(i, v)

            t.build(100)  # N trees
            t.save('test.ann')

            u = AnnoyIndex(f, 'euclidean')
            u.load('test.ann')  # super fast, will just mmap the file

            u.get_item_vector(0)

            queryindex = queryMesh
            print("quering mesh:", csvData.iloc[queryindex, 0])
            print("N nearest neighbors", u.get_nns_by_item(queryindex, 10))  # will find the N nearest neighbors

            # a.get_nns_by_vector(v, n, search_k=-1, include_distances=False) # can query by the factor

            real_labels = []
            meshFile = []
            for i in u.get_nns_by_item(queryindex, 10):
                real_labels.append(classSet[i])
                meshFile.append(csvData.iloc[i, 0])

            print(real_labels)
            print(meshFile)

        # annQuery(data, classList, 1789)

        def T_SNE(csvData, label):
            new_labels = []
            for i in label:
                new_labels.append(i.tolist().index(1))
            #   print(classList[0:len(label)])
            #   print(new_labels)
            feature = (csvData.iloc[:, 1:] - csvData.iloc[:, 1:].min()) / (
                    csvData.iloc[:, 1:].max() - csvData.iloc[:, 1:].min())
            features_embedded = TSNE(n_components=2, perplexity=10).fit_transform(feature)
            return features_embedded, new_labels

        features_embedded, labelsIndex = T_SNE(data, labels)
        # draw the plot

        vis_x = features_embedded[:, 0]
        vis_y = features_embedded[:, 1]

        fig, ax = plt.subplots()
        sc = plt.scatter(vis_x, vis_y, c=labelsIndex, cmap=plt.cm.get_cmap("jet", 53))
        annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"),
                            arrowprops=dict(arrowstyle="->"))
        # annotText = "{}, {}".format(data.iloc[0,0],classList[0])
        # annot.set_text(annotText)
        annot.set_visible(False)

        def update_annot(ind):

            pos = sc.get_offsets()[ind["ind"][0]]
            annot.xy = pos
            for i in range(len(vis_x)):
                if vis_x[i] == pos[0] and vis_y[i] == pos[1]:
                    rowindex = i
                continue
            text = "{}, {}".format(data.iloc[rowindex, 0], classList[rowindex])
            annot.set_text(text)
            annot.get_bbox_patch().set_alpha(0.4)

        def hover(event):
            vis = annot.get_visible()
            if event.inaxes == ax:
                cont, ind = sc.contains(event)
                if cont:
                    update_annot(ind)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if vis:
                        annot.set_visible(False)
                        fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", hover)
        # plt.colorbar(ticks=range(53))
        plt.rcParams["figure.figsize"] = 10, 10
        plt.show()

