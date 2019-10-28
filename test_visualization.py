import pandas as pd
from annoy import AnnoyIndex
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import preprocessing


# step 5


# load the feature
path='/Users/jack/Desktop/privateStuff/UUstuff/2019-2020/period1/MR/assignment/all_feature.csv'
data = pd.read_csv(path)


# load the labels and binarize the labels
path2='/Users/jack/Desktop/privateStuff/UUstuff/2019-2020/period1/MR/assignment/file_benchmark_before_refinement.csv'
data2 = pd.read_csv(path2)
classList = []
for i in data.iloc[:,0]:
  className = data2.loc[data2['fileName'] == i].iloc[0,0]
  classList.append(className)

lb = preprocessing.LabelBinarizer()
lb.fit(classList)
labels = lb.transform(classList)

# print(feature.shape)
# print(labels)


def annQuery(csvData, classSet, queryMesh=100): # in the end, the queryMesh should be changed to a mesh file or the feature of the mesh
  f = 45
  t = AnnoyIndex(f, 'euclidean')  # Length of item vector that will be indexed

  # nomalize feature data
  features = (csvData.iloc[:, 1:] - csvData.iloc[:, 1:].min()) / (csvData.iloc[:, 1:].max() - csvData.iloc[:, 1:].min())
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

def T_SNE(csvData,label):
  new_labels=[]
  for i in label:
    new_labels.append(i.tolist().index(1))
#   print(classList[0:len(label)])
#   print(new_labels)
  feature = (csvData.iloc[:, 1:] - csvData.iloc[:, 1:].min()) / (csvData.iloc[:, 1:].max() - csvData.iloc[:, 1:].min())
  features_embedded = TSNE(n_components=2).fit_transform(feature)
  return features_embedded,new_labels






features_embedded,labelsIndex = T_SNE(data, labels)
# draw the plot

vis_x = features_embedded[:, 0]
vis_y = features_embedded[:, 1]

fig, ax = plt.subplots()
sc=plt.scatter(vis_x, vis_y, c=labelsIndex, cmap=plt.cm.get_cmap("jet", 53))
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
plt.colorbar(ticks=range(53))
plt.rcParams["figure.figsize"] = 10, 10
plt.show()



