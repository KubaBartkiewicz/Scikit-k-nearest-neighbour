import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

# Load data
train = pd.read_csv('D:\Scikit k-nearest neighbour\data/train.csv', header=None)
trainLabel = pd.read_csv('D:\Scikit k-nearest neighbour\data/trainLabels.csv', header=None)
test = pd.read_csv('D:\Scikit k-nearest neighbour\data/test.csv', header=None)

# Data details
print('Train shape: ' + str(train.shape))
print('Test shape: ' + str(test.shape))
print('TrainLabel shape: ' + str(trainLabel.shape))
print(train.head())
print(train.tail())
print(train.info())
print(train.describe())

# Split data
X, y = train, np.ravel(trainLabel)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Data features correlation map
f, ax = plt.subplots(figsize=[14, 7])
sns.heatmap(pd.DataFrame(np.c_[X, trainLabel, X.loc[:, 9]]).corr(),
            annot=True, linewidths=.5, fmt='0.2f', ax=ax, annot_kws={"size": 5})
plt.yticks(rotation=0)
plt.title('Data features correlations')
plt.show()

# kNN parameters
k_range = np.arange(1, 21)     # Range of checked k for kNN
cv = 5                         # Number of cross validation groups
val_accuracy = {'test': [], 'cross_val': []}

# Loop over different values of k
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    val_accuracy['test'].append(knn.score(X_test, y_test))
    val_accuracy['cross_val'].append(np.mean(cross_val_score(knn, X, y, cv=cv)))

# Plot of kNN accuracy for cross validation and standard validation on test data set
plt.style.use("ggplot")
plt.plot(k_range, val_accuracy['test'], label='Accuracy on test data')
plt.plot(k_range, val_accuracy['cross_val'], label='Cross Validation Accuracy')
plt.legend()
plt.title('Different validation methods')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(k_range)
plt.show()


# Feature scaling
std = StandardScaler()
X_std = std.fit_transform(X)
norm = Normalizer()
X_norm = norm.fit_transform(X)
scaling_val_accuracy = {'std': [], 'norm': []}
bestKnn = None
bestAcc = 0.0
data_processing_method = None

# Loop over different values of k
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    a1 = np.mean(cross_val_score(knn, X_std, y, cv=cv))
    scaling_val_accuracy['std'].append(a1)
    a2 = np.mean(cross_val_score(knn, X_norm, y, cv=cv))
    scaling_val_accuracy['norm'].append(a2)

    # saving best score parameters
    if a1 > bestAcc:
        bestAcc = a1
        bestKnn = knn
        data_processing_method = 'Standardized'
    if a2 > bestAcc:
        bestAcc = a2
        bestKnn = knn
        data_processing_method = 'Normalized'

# Plot of kNN accuracy for different feature scaling methods
plt.style.use("ggplot")
plt.plot(k_range, val_accuracy['cross_val'], label='CV Accuracy without scaling')
plt.plot(k_range, scaling_val_accuracy['std'], label='CV Accuracy with std')
plt.plot(k_range, scaling_val_accuracy['norm'], label='CV Accuracy with norm')
plt.legend()
plt.title('Feature scaling methods')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(k_range)
plt.show()


# PCA - principal component analysis
pca_range = np.arange(1, 40)
best_pca = 0
pca_accuracy = []

# Loop over different values of k and different number of PCA components
for i in pca_range:
    pca = PCA(n_components=i)
    X_pca = pca.fit_transform(X)
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        a1 = np.mean(cross_val_score(knn, X_pca, y, cv=cv))
        if a1 > bestAcc:
            bestAcc = a1
            bestKnn = knn
            data_processing_method = 'PCA'
            best_pca = i

# Needed for plot
# Calculating accuracy of different k values for best number of components
for k in k_range:
    pca = PCA(n_components=best_pca)
    X_pca = pca.fit_transform(X)
    knn = KNeighborsClassifier(n_neighbors=k)
    pca_accuracy.append(np.mean(cross_val_score(knn, X_pca, y, cv=cv)))

# Plot of accuracy after pca and raw data
plt.style.use("ggplot")
plt.plot(k_range, pca_accuracy, label='CV Accuracy with pca')
plt.plot(k_range, val_accuracy['cross_val'], label='CV Accuracy without scaling')
plt.legend()
plt.title(f'PCA with {best_pca}-components and raw data')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(k_range)
plt.show()


# GMM - gaussian mixture model
lowest_bic = np.infty
n_components_range = range(1, 7)
cv_types = ['spherical', 'tied', 'diag', 'full']
gmm_values = []

# Choosing best fitting model using GMM
for cv_type in cv_types:
    for n_components in n_components_range:
        gmm = GaussianMixture(n_components=n_components, covariance_type=cv_type)
        gmm.fit(X)
        gmm_values.append(gmm.aic(X))
        if gmm_values[-1] < lowest_bic:
            lowest_bic = gmm_values[-1]
            best_gmm = gmm

best_gmm.fit(X)
gmm_train = best_gmm.predict_proba(X)
gmm_val_accuracy = {'none': [], 'gmm': []}

# Loop over different k values for best GMM model
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    a1 = np.mean(cross_val_score(knn, gmm_train, y, cv=cv))
    gmm_val_accuracy['gmm'].append(a1)
    if a1 > bestAcc:
        bestAcc = a1
        bestKnn = knn
        data_processing_method = 'GMM'

plt.style.use("ggplot")
plt.plot(k_range, gmm_val_accuracy['gmm'], label='CV Accuracy with GMM')
plt.plot(k_range, val_accuracy['cross_val'], label='CV Accuracy without scaling')
plt.legend()
plt.title('GMM and Raw data')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(k_range)
plt.show()


# Best method
print("Best k: ", bestKnn)
print("Method: ", data_processing_method)
print("Accuracy: ", bestAcc)


# Predict test set labels
best_gmm.fit(test)
gmm_test = best_gmm.predict_proba(test)
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(gmm_train, y)
solution = pd.DataFrame(knn.predict(gmm_test))
solution.columns = ['Solution']
solution['Id'] = np.arange(1, solution.shape[0]+1)
solution = solution[['Id', 'Solution']]

solution.to_csv('data/solution.csv', index=False)
