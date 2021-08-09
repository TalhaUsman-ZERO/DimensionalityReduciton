# Principal Component Analysis (PCA)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer

# Importing the dataset
dataset = pd.read_csv('UNSW_NB15_traintest-dataset.csv')

X = dataset.iloc[:, 1:43].values

labelencoder_attack=LabelEncoder()
y = labelencoder_attack.fit_transform(dataset.iloc[:, -2].values)

#labeling categorical features for independent variables
labelencoder_proto=LabelEncoder()
X[:,1]=labelencoder_proto.fit_transform(X[:,1])

labelencoder_service=LabelEncoder()
X[:,2]=labelencoder_service.fit_transform(X[:,2])


labelencoder_state=LabelEncoder()
X[:,3]=labelencoder_state.fit_transform(X[:,3])

ct=ColumnTransformer([("proto,sevice,state",OneHotEncoder(),[1,2,3])],remainder="passthrough")
X=ct.fit_transform(X)
X=X.toarray()


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components =2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_


# Training the Random Forest Classification model on the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('blue', 'grey','orange','black','white','yellow','brown','red','purple','magenta')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('blue', 'grey','orange','black','white','yellow','brown','red','purple','magenta'))(i), label = j)
plt.title('0:Analysis  1:Backdoor  2:DOS  3:Exploits  4:Fuzzers  5:Generic  6:Normal  7:Reconnainssance  8:Shellcode  9:Worms')
plt.suptitle('Random Forest Classification (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('blue', 'grey','orange','black','white','yellow','brown','red','purple','magenta')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('blue', 'grey','orange','black','white','yellow','brown','red','purple','magenta'))(i), label = j)
plt.title('0:Analysis  1:Backdoor  2:DOS  3:Exploits  4:Fuzzers  5:Generic  6:Normal  7:Reconnainssance  8:Shellcode  9:Worms')
plt.suptitle('Random Forest Classification (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

