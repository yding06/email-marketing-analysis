import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import resample

df = pd.read_csv('age22_campaign_pred.tsv', delimiter = '\t')
df = df.drop(df.columns[0],axis = 1)
df = df.drop('state',1)
# down_sample
df_majority = df[df.open==0]
df_minority = df[df.open==1]
df_majority_downsampled = resample(df_majority,
                                 replace=False,    # sample without replacement
                                 n_samples=41331,     # to match minority class
                                 random_state=123)
df_downsampled = pd.concat([df_majority_downsampled, df_minority])

print df_downsampled.open.value_counts()
# One hot code
df_downsampled = pd.get_dummies(df_downsampled)

# Cross-validation
df_downsampled = df_downsampled.values
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.1, random_state=0)
y = df_downsampled[0::,1]
X = df_downsampled[0::,2::]
acc_list = {}

for train_index, test_index in sss.split(X,y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
# Train KNN model
    n_neighbor = [5,6,7,8,9,10]
    for k in n_neighbor:
        knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None,
                                   n_jobs=1, n_neighbors=k, p=2, weights='uniform')
        knn.fit(X_train, y_train)
        train_prediction = knn.predict(X_test)
        prob = knn.predict_proba(X_test)
        acc = accuracy_score(y_test, train_prediction)
        if k in acc_list:
            acc_list[k] += acc
        else:
            acc_list[k] = acc
print(acc_list)
print prob
# Accuracy
for i in acc_list:
    a = acc_list[i] / 10
    print("KNN: When k = ", i, "the accuracy is", a)
