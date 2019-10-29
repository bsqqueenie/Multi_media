# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'int.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!
import trimesh
import tkinter as tk
from tkinter import filedialog
import numpy as np
import os
import pandas as pd
from pandas import Series
import networkx as nx
from sklearn.decomposition import PCA
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QGraphicsView, QGraphicsScene, QApplication, QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

meshlist = []
class Figure_Canvas(FigureCanvas):

    # 通过继承FigureCanvas类，使得该类既是一个PyQt5的Qwidget，又是一个matplotlib的FigureCanvas，连接pyqt5与matplot
    def __init__(self, parent=None, width=6, height=2.3):
        fig = plt.figure(figsize=(width, height), dpi=100)
        FigureCanvas.__init__(self, fig) # 初始化父类
        self.setParent(parent)
        self.ax = fig.add_subplot(111) #调用figure下面的add_subplot方法，类似于matplotlib.pyplot下面的subplot方法

    def p1(self, Item=None, type=None):
        Item.show()

def Normalization(path):
    ori = [0,0,0]
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

    pca = PCA(n_components=2)
    Reduced_mesh = pca.fit_transform(mesh.vertices)
    #print(pca.components_)

    transform_x = trimesh.geometry.align_vectors(pca.components_[0], [1, 0, 0])
    mesh.apply_transform(transform_x)
    Reduced_mesh_newx = pca.fit_transform(mesh.vertices)

    transform_y = trimesh.geometry.align_vectors(pca.components_[1], [0, 1, 0])
    mesh.apply_transform(transform_y)
    Reduced_mesh_newy = pca.fit_transform(mesh.vertices)
    #print(pca.components_)
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

    for vertex in mesh.vertices:
        if vertex[0] <= center[0]:
            moment_lx += np.linalg.norm(vertex - ori)
        else:
            moment_rx += np.linalg.norm(vertex - ori)
    if moment_lx < moment_rx:  # right side of x axis should be the moment higher side
        transform = trimesh.geometry.align_vectors([-1, 0, 0], [1, 0, 0])
        mesh.apply_transform(transform)

    for vertex in mesh.vertices:
        if vertex[1] <= center[1]:
            moment_ly += np.linalg.norm(vertex - ori)
        else:
            moment_ry += np.linalg.norm(vertex - ori)
    if moment_ly < moment_ry:  # right side of y axis should be the moment higher side
        transform = trimesh.geometry.align_vectors([0, -1, 0], [0, 1, 0])
        mesh.apply_transform(transform)

    for vertex in mesh.vertices:
        if vertex[2] <= center[2]:
            moment_lz += np.linalg.norm(vertex - ori)
        else:
            moment_rz += np.linalg.norm(vertex - ori)
    if moment_lz < moment_rz:  # right side of z axis should be the moment higher side
        transform = trimesh.geometry.align_vectors([0, 0, -1], [0, 0, 1])
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

def querying(filename):
    global meshlist
    filename_list = []
    dislist = [0]
    data = pd.read_csv("/Users/darkqian/PycharmProjects/MR/Multi_meadia/feature/all_feature.csv")

    norm_data = (data.iloc[:,1:] - data.iloc[:,1:].min()) / (data.iloc[:,1:].max() - data.iloc[:,1:].min())

    #norm_data = data.iloc[:,1:] / data.iloc[:,1:].max(axis=0)
    #norm_data = (data.iloc[:,1:] - data.iloc[:,1:].mean()) / (data.iloc[:,1:].std())
    new_data = pd.concat([data.iloc[:,0], norm_data], 1)

    row = new_data.shape[0]
    Target = new_data.loc[new_data["fileName"]==filename] #set target model
    Target = Target.iloc[:,1:]

    for i in range(row):

        Com = new_data.iloc[i, 1:]
        #Dis = scipy.stats.wasserstein_distance(Target, Com) #Earth mover's distance
        Dis = np.linalg.norm(Target - Com) # Euclidean distance
        #Dis = scipy.spatial.distance.cosine(Target, Com) #cosine distance
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
        mesh = Normalization('/Users/darkqian/PycharmProjects/MR/benchmark/db/'+ str(int(direname1)) + '/m' + str(int(number)) +'/m' + str(int(number)) +'.off')
        meshlist.append(mesh)

    return meshlist

class Ui_MainWindow(QtWidgets.QWidget):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 574)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout_2.addWidget(self.pushButton_2)
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout_2.addWidget(self.pushButton)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setObjectName("graphicsView")
        self.verticalLayout.addWidget(self.graphicsView)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.graphicsView_2 = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.horizontalLayout.addWidget(self.graphicsView_2)
        self.graphicsView_3 = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView_3.setObjectName("graphicsView_3")
        self.horizontalLayout.addWidget(self.graphicsView_3)
        self.graphicsView_4 = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView_4.setObjectName("graphicsView_4")
        self.horizontalLayout.addWidget(self.graphicsView_4)
        self.graphicsView_5 = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView_5.setObjectName("graphicsView_5")
        self.horizontalLayout.addWidget(self.graphicsView_5)
        self.verticalLayout.addLayout(self.horizontalLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.pushButton.clicked.connect(self.openfile)
        self.pushButton_2.clicked.connect(self.query)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton_2.setText(_translate("MainWindow", "Query"))
        self.pushButton.setText(_translate("MainWindow", "File select"))

    def openfile(self):
        global meshlist
        _translate = QtCore.QCoreApplication.translate
        filename, filetype = QFileDialog.getOpenFileName(self, "选取文件", "./",
                                                         "OFF Files (*.off);;PLY Files (*.ply)")
        [dirname, bfilename] = os.path.split(filename)
        print(bfilename)
        meshlist = querying(bfilename)

        dr = Figure_Canvas()
        dr.p1(meshlist[0])  # 画图
        graphicscene = QtWidgets.QGraphicsScene()
        graphicscene.addWidget(dr)  # 把图形放到QGraphicsScene中，注意：图形是作为一个QWidget放到QGraphicsScene中的
        self.graphicsView.setScene(graphicscene)  # 第五步，把QGraphicsScene放入QGraphicsView
        self.graphicsView.show()

    def query(self):

        dr = Figure_Canvas()
        dr.p1(meshlist[1])  # 画图
        graphicscene = QtWidgets.QGraphicsScene()
        graphicscene.addWidget(dr)  # 把图形放到QGraphicsScene中，注意：图形是作为一个QWidget放到QGraphicsScene中的
        self.graphicsView_2.setScene(graphicscene)  # 第五步，把QGraphicsScene放入QGraphicsView
        self.graphicsView_2.show()

        r = Figure_Canvas()
        dr.p1(meshlist[2])  # 画图
        graphicscene = QtWidgets.QGraphicsScene()
        graphicscene.addWidget(dr)  # 把图形放到QGraphicsScene中，注意：图形是作为一个QWidget放到QGraphicsScene中的
        self.graphicsView_3.setScene(graphicscene)  # 第五步，把QGraphicsScene放入QGraphicsView
        self.graphicsView_3.show()

        r = Figure_Canvas()
        dr.p1(meshlist[3])  # 画图
        graphicscene = QtWidgets.QGraphicsScene()
        graphicscene.addWidget(dr)  # 把图形放到QGraphicsScene中，注意：图形是作为一个QWidget放到QGraphicsScene中的
        self.graphicsView_4.setScene(graphicscene)  # 第五步，把QGraphicsScene放入QGraphicsView
        self.graphicsView_4.show()

        r = Figure_Canvas()
        dr.p1(meshlist[4])  # 画图
        graphicscene = QtWidgets.QGraphicsScene()
        graphicscene.addWidget(dr)  # 把图形放到QGraphicsScene中，注意：图形是作为一个QWidget放到QGraphicsScene中的
        self.graphicsView_5.setScene(graphicscene)  # 第五步，把QGraphicsScene放入QGraphicsView
        self.graphicsView_5.show()
