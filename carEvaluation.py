import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pandas as pd
import seaborn as sns

names = ["buying","maint", "doors", "persons", "lug_boot", "safety", "class_values"]
cars = pd.read_csv("car.data", names = names)

#creating mapping

buying_label = {v : k for k, v in enumerate(set(cars["buying"]))}
maint_label = {v : k for k, v in enumerate(set(cars["maint"]))}
doors_label = {v : k for k, v in enumerate(set(cars["doors"]))}
persons_label = {v : k for k, v in enumerate(set(cars["persons"]))}
lug_boot_label = {v : k for k, v in enumerate(set(cars["lug_boot"]))}
safety_label = {v : k for k, v in enumerate(set(cars["safety"]))}
class_label = {v : k for k, v in enumerate(set(cars["class_values"]))}

cars_encode = cars.copy()
#print(cars_encode.head())

cars_encode["buying"] = cars_encode["buying"].map(buying_label)
cars_encode["maint"] = cars_encode["maint"].map(maint_label)
cars_encode["doors"] = cars_encode["doors"].map(doors_label)
cars_encode["persons"] = cars_encode["persons"].map(persons_label)
cars_encode["lug_boot"] = cars_encode["lug_boot"].map(lug_boot_label)
cars_encode["safety"] = cars_encode["safety"].map(safety_label)
cars_encode["class_values"] = cars_encode["class_values"].map(class_label)

plt.figure(figsize = (10, 6))
sns.heatmap(cars_encode.corr(), annot = True)
plt.show()
#plt.show()

#print(cars.describe())
#print(cars_encode.describe())
#print(cars_encode.head())

#print(cars_encode)

x = cars_encode[["buying", "maint", "doors", "persons", "lug_boot", "safety"]]
y = cars_encode["class_values"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 7)

models = []
names = []
results = []

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

models.append(("Logistic Reg", LogisticRegression()))
models.append(("Dec. Tree", DecisionTreeClassifier()))
models.append(("Bayes", GaussianNB()))
models.append(("MulltiBayes", MultinomialNB()))
models.append(("KNN", KNeighborsClassifier()))

for name, model in models:
    kfold = StratifiedKFold(n_splits = 10, random_state = 1, shuffle = True)
    cv_results = cross_val_score(model, x_train, y_train, cv = kfold, scoring = 'accuracy')
    results.append(cv_results)
    names.append(name)
    print(f'{name}: {cv_results.mean()} ({cv_results.std()})')

for name, model in models:
    fitted_model = model.fit(x_train, y_train)
    print(f"Accuracy score for {name}: {accuracy_score(y_test, fitted_model.predict(x_test))}")

#compare algorithms
plt.boxplot(results, labels = names)
plt.title("Algorithm Comparison")
plt.show()


