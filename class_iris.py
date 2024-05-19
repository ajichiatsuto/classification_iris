from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import mglearn
import matplotlib.pyplot as plt

# iris_datasetはディクショナリによく似たBunchクラスのオブジェクト
iris_dataset = load_iris()
# print("keys of iris_dataset: \n{}".format(iris_dataset.keys()))
# print(iris_dataset["feature_names"])

# データセットの分割
# random_stateは乱数生成器のシードを固定するためのパラメータ
X_train, X_test, y_train, y_test = train_test_split(iris_dataset["data"], iris_dataset["target"], random_state=0)

# X_trainのデータからDataFrameを作る
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)

# データフレームからscatter matrixを作成し、y_trainに従って色を付ける(cで設定)
# hist_kwdsはヒストグラムのビンの数を設定
# sは点のサイズ、alphaは透明度、cmapはカラーマップ
# pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker="o", hist_kwds={"bins": 20}, s=60, alpha=.8, cmap=mglearn.cm3)
# plt.show()

# knnオブジェクトの生成
knn = KNeighborsClassifier(n_neighbors=1)

# 訓練データセットを用いてモデルを構築
knn.fit(X_train, y_train)

# 予測を行う
y_pred = knn.predict(X_test)
print("Test set predictions: \n{}".format(y_pred))
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))