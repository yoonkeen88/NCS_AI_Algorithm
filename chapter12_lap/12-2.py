from sklearn.datasets import load_breast_cancer
from sklearn import tree
import pydotplus

wdbc=load_breast_cancer()

decision_tree=tree.DecisionTreeClassifier(max_depth=4,random_state=1)
dt=decision_tree.fit(wdbc.data,wdbc.target) # 결정 트리 학습

res=dt.predict(wdbc.data)
print('결정 트리의 정확률=',sum(res==wdbc.target)/len(res)) # 예측

dot=tree.export_graphviz(dt,out_file=None,feature_names=wdbc.feature_names,class_names=wdbc.target_names,filled=True,node_ids=True,rounded=True)
graph=pydotplus.graph_from_dot_data(dot)

import os
os.environ['PATH']=os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"

graph.write_png('tree.png') # 결정 트리를 위한 그림 저장

x_test=wdbc.data[0:1]
path=dt.decision_path(x_test)
path_seq=path.toarray()[0]

for n,value in enumerate(path_seq):
    node=graph.get_node(str(n))[0]
    if value==0:
        node.set_fillcolor('white')
    else:
        node.set_fillcolor('green') # 의사결정 경로를 green 색으로 표시

graph.write_png('tree_with_path.png') # 의사결정 경로를 포함한 그림 저장