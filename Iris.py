import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

#Load Iris Dataset
iris = sns.load_dataset('iris')

#Print First five elements
print(iris.head())
#Print info of the datasets
print(iris.info())
#Print info description for the datasets:
print(iris.describe())

#Split data into training and testing:
iris_index = np.arange(iris.shape[0])
training_data = int(np.floor(0.75 * iris.shape[0]))
iris_training_index = np.random.choice(iris_index, size = training_data, replace = False)
iris_testing_index = np.delete(iris_index, iris_training_index)
iris_training = iris.iloc[iris_training_index]
iris_testing = iris.iloc[iris_testing_index]

#Shuffling:
iris_index = np.arange(iris.shape[0])
np.random.shuffle(iris_index)
training_data = int(np.floor(0.75 * iris.shape[0]))
iris_training = iris.iloc[:training_data]
iris_test = iris.iloc[training_data:]

#Print Training and testing data Info:
print(iris_training.info())
print(iris_testing.info())

#Import Sklearn Package:
from sklearn.model_selection import train_test_split
x = iris.drop('species', axis = 1)
y = iris['species']
x_train, x_test, y_train , y_test = train_test_split(x, y, test_size = 0.25, random_state = 20, stratify = y)

#Print Info:
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_train.shape)

#KNN:
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
k_range = list(range(1,26))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    scores.append(accuracy_score(y_test, y_pred))

plt.plot(k_range, scores)
plt.xlabel('K value')
plt.ylabel('Accuracy score')
plt.title('Accuracy score for KNN')
plt.show()

#Logistic Regression:
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(x_train, y_train)
y_pred = log.predict(x_test)
print("the Accuracy of logistic regression is:" , accuracy_score(y_test, y_pred))

#Data Visualization:
data_viz = sns.pairplot(iris, hue = 'species', markers = '+')
plt.show()

#Violin Plot Visualization:
viz = sns.violinplot(y = 'species', x = 'sepal_length', data = iris, inner = 'quartile')
plt.show()
viz = sns.violinplot(y = 'species', x = 'sepal_width', data = iris, inner = 'quartile')
plt.show()
viz= sns.violinplot(y = 'species', x = 'petal_length', data = iris, inner = 'quartile')
plt.show()
viz = sns.violinplot(y = 'species', x = 'petal_width', data = iris, inner = 'quartile')
plt.show()

#Choosing KNN to Model Iris Species Prediction:
#Since on the plot it shows that at k = 8, it has the highest accuracy score:
knn = KNeighborsClassifier(n_neighbors= 8)
knn.fit(x, y)
print(knn.predict([[6,3,4,2]]))

