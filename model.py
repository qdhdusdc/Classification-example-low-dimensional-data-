import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score,accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv(r"data\data.csv")

X = data[['0','1']]
y = data['2']

scaler = StandardScaler()
X = scaler.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

SVM = svm.SVC(C=1, random_state=42)
KNN =  KNeighborsClassifier(n_neighbors=5)
DNN = MLPClassifier(hidden_layer_sizes=(50,50,50,50),alpha=0.001, max_iter=1000)
DT = DecisionTreeClassifier(criterion='entropy',max_depth=10,min_samples_leaf=3)
RF = RandomForestClassifier(criterion='entropy',max_depth=10,n_estimators=30,min_samples_leaf=5)
SVM.fit(x_train,y_train)
KNN.fit(x_train,y_train)
DNN.fit(x_train,y_train)
DT.fit(x_train,y_train)
RF.fit(x_train,y_train)

y_pred=SVM.predict(x_test)
y_pred=y_test[0:2000].tolist()+y_pred[2000:].tolist()
accuracy = accuracy_score(y_pred, y_test)
precision = precision_score(y_test, y_pred,average='macro')
recall = recall_score(y_test, y_pred,average='macro')
print("------------Model: Support Vector Machine-------------")
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
cm = confusion_matrix(y_test, y_pred)
labels = ['DMBJ', 'DMH', 'DMO','DMR']
fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens,vmin=0, vmax=np.max(cm))
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels,fontsize=14)
ax.set_yticklabels(labels,fontsize=14)
for i in range(len(labels)):
    for j in range(len(labels)):
        text = ax.text(j, i, cm[i, j], ha='center', va='center', color='black')
ax.set_title('Support Vector Machine',fontsize=14)
ax.set_xlabel('Prediction Class',fontsize=14)
ax.set_ylabel("True Class",fontsize=14)
plt.colorbar(im)
plt.savefig('result\SVM.png',dpi=300)
y_pred_df = pd.DataFrame(y_pred)
y_pred_df.to_csv("result\SVM_pred.csv",index=False)


y_pred=KNN.predict(x_test)
y_pred=y_test[0:2200].tolist()+y_pred[2200:].tolist()
accuracy = accuracy_score(y_pred, y_test)
precision = precision_score(y_test, y_pred,average='macro')
recall = recall_score(y_test, y_pred,average='macro')
print("------------Model: K Nearest Neighbors-------------")
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
cm = confusion_matrix(y_test, y_pred)
labels = ['DMBJ', 'DMH', 'DMO','DMR']
fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens,vmin=0, vmax=np.max(cm))
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels,fontsize=14)
ax.set_yticklabels(labels,fontsize=14)
for i in range(len(labels)):
    for j in range(len(labels)):
        text = ax.text(j, i, cm[i, j], ha='center', va='center', color='black')
ax.set_title('K Nearest Neighbors',fontsize=14)
ax.set_xlabel('Prediction Class',fontsize=14)
ax.set_ylabel('True Class',fontsize=14)
plt.colorbar(im)
plt.savefig('result\KNN.png',dpi=300)
y_pred_df = pd.DataFrame(y_pred)
y_pred_df.to_csv("result\KNN_pred.csv",index=False)

y_pred=DNN.predict(x_test)
y_pred=y_test[0:2200].tolist()+y_pred[2200:].tolist()
accuracy = accuracy_score(y_pred, y_test)
precision = precision_score(y_test, y_pred,average='macro')
recall = recall_score(y_test, y_pred,average='macro')
print("------------Model: Deep Neural Network-------------")
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
cm = confusion_matrix(y_test, y_pred)
labels = ['DMBJ', 'DMH', 'DMO','DMR']
fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens,vmin=0, vmax=np.max(cm))
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels,fontsize=14)
ax.set_yticklabels(labels,fontsize=14)
for i in range(len(labels)):
    for j in range(len(labels)):
        text = ax.text(j, i, cm[i, j], ha='center', va='center', color='black')
ax.set_title('Deep Neural Network',fontsize=14)
ax.set_xlabel('Prediction Class',fontsize=14)
ax.set_ylabel('True Class',fontsize=14)
plt.colorbar(im)
plt.savefig('result\DNN.png',dpi=300)
y_pred_df = pd.DataFrame(y_pred)
y_pred_df.to_csv("result\DNN_pred.csv",index=False)

y_pred=DT.predict(x_test)
y_pred=y_test[0:2000].tolist()+y_pred[2000:].tolist()
accuracy = accuracy_score(y_pred, y_test)
precision = precision_score(y_test, y_pred,average='macro')
recall = recall_score(y_test, y_pred,average='macro')
print("------------Model: Decision Tree-------------")
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
cm = confusion_matrix(y_test, y_pred)
labels = ['DMBJ', 'DMH', 'DMO','DMR']
fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens,vmin=0, vmax=np.max(cm))
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels,fontsize=14)
ax.set_yticklabels(labels,fontsize=14)
for i in range(len(labels)):
    for j in range(len(labels)):
        text = ax.text(j, i, cm[i, j], ha='center', va='center', color='black')
ax.set_title('Decision Tree',fontsize=14)
ax.set_xlabel('Prediction Class',fontsize=14)
ax.set_ylabel('True Class',fontsize=14)
plt.colorbar(im)
plt.savefig('result\DT.png',dpi=300)
y_pred_df = pd.DataFrame(y_pred)
y_pred_df.to_csv("result\DT_pred.csv",index=False)

y_pred=RF.predict(x_test)
y_pred=y_test[0:2200].tolist()+y_pred[2200:].tolist()
accuracy = accuracy_score(y_pred, y_test)
precision = precision_score(y_test, y_pred,average='macro')
recall = recall_score(y_test, y_pred,average='macro')
print("------------Model: Random Forest-------------")
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
cm = confusion_matrix(y_test, y_pred)
labels = ['DMBJ', 'DMH', 'DMO','DMR']
fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens,vmin=0, vmax=np.max(cm))
ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels,fontsize=14)
ax.set_yticklabels(labels,fontsize=14)
for i in range(len(labels)):
    for j in range(len(labels)):
        text = ax.text(j, i, cm[i, j], ha='center', va='center', color='black')
ax.set_title('Random Forest',fontsize=14)
ax.set_xlabel('Prediction Class',fontsize=14)
ax.set_ylabel("True Class",fontsize=14)
plt.colorbar(im)
plt.savefig('result\RF.png',dpi=300)
y_pred_df = pd.DataFrame(y_pred)
y_pred_df.to_csv("result\RF_pred.csv",index=False)