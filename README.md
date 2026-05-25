# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. open colab notebook.
2.Enter the python code.
3.run the program.
4.program run successfully.
5.take screenshot for the output.

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by:PRIYADHARSAHAN U
RegisterNumber:212225220075 
*/
```
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


data_dict = {
    "CustomerID":[1,2,3,4,5,6,7,8,9,10],
    "Gender":["Male","Male","Female","Female","Female",
              "Female","Female","Male","Male","Female"],
    "Age":[19,21,20,23,31,22,35,23,64,30],
    "Annual Income (k$)":[15,16,17,18,19,20,21,22,23,24],
    "Spending Score (1-100)":[39,81,6,77,40,76,6,94,3,72]
}

data = pd.DataFrame(data_dict)


data.to_csv(r"C:\Users\acer\Downloads\Mall_Customers.csv", index=False)


data = pd.read_csv(r"C:\Users\acer\Downloads\Mall_Customers.csv")

print(data.head())
print(data.isnull().sum())


wcss=[]

for i in range(1,6):
    kmeans = KMeans(
        n_clusters=i,
        init='k-means++',
        n_init=10,
        random_state=42
    )

    kmeans.fit(data.iloc[:,3:])
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(7,5))
plt.plot(range(1,6),wcss,marker='o')
plt.xlabel("No. of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()


km = KMeans(
    n_clusters=3,
    init='k-means++',
    n_init=10,
    random_state=42
)

y_pred = km.fit_predict(data.iloc[:,3:])

data["cluster"]=y_pred


plt.figure(figsize=(8,6))

for i in range(3):
    cluster=data[data["cluster"]==i]

    plt.scatter(
        cluster["Annual Income (k$)"],
        cluster["Spending Score (1-100)"],
        label=f"Cluster {i+1}"
    )


plt.scatter(
    km.cluster_centers_[:,0],
    km.cluster_centers_[:,1],
    s=200,
    marker='X',
    label="Centroids"
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segments")
plt.legend()
plt.show()
```

## Output:
```
```
<img width="476" height="784" alt="Screenshot 2026-05-25 180523" src="https://github.com/user-attachments/assets/b08291f7-4159-41f1-80b0-6656f4991ace" />



## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
