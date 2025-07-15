from sklearn.datasets import load_breast_cancer

wdbc=load_breast_cancer()
print(wdbc.DESCR)
print('특징 이름=',list(wdbc.feature_names))
print('부류 이름=',list(wdbc.target_names))