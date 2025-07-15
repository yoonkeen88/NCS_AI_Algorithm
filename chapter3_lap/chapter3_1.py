from sklearn import datasets
d = datasets.load_iris()
print(d.data.shape)
print(d.target.shape)
print(d.feature_names)
print(d.target_names)
print(d.DESCR)  # Print the description of the dataset