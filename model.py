import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import time as time

data = pd.read_csv(r"C:\Users\aarus\coding\ud120-projects\breast_cancer_NB\NB_bc_atmpt3\data.csv")
testData = pd.read_csv(r"C:\Users\aarus\coding\ud120-projects\breast_cancer_NB\NB_bc_atmpt3\testData.csv")

print(data.head(10))

y = data.diagnosis
cancerous_features = ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness']
x = data[cancerous_features]

features_test = testData[cancerous_features]
label_test = testData.diagnosis

clf = GaussianNB()

clf.fit(x, y)
print(clf.score(features_test, label_test))

testPrediction = np.array([[9.683,19.34,61.05,285.7,0.08491]])
pred_diag = clf.predict(testPrediction)

if pred_diag == 0:
    print("This data suggests NOT having breast cancer.")
else:
    print("This data suggests HAVING breast cancer. See an oncologist!")

