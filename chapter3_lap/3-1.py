from sklearn import datasets

d=datasets.load_iris() # iris 데이터셋을 읽고
print(d.DESCR) # 내용을 출력
for i in range(0,len(d.data)): # 샘플을 순서대로 출력
    print(i+1,d.data[i],d.target[i])
from sklearn import svm

s=svm.SVC(gamma=0.1,C=10) # svm 분류 모델 SVC 객체 생성 -> 아마 gamma와 C는 하이퍼파라미터 
# gamma는 커널 함수의 영향을 조절하고, C는 오분류에 대한 패널티를 조절
# gamma가 작을수록 결정 경계가 부드럽고, C가 크면 오분류에 대한 패널티가 커짐
# 따라서 C와 gamma를 적절히 조정하여 모델의 성능을 최적화할 수 있음
# SVC 는 Support Vector Classification의 약자로, SVM을 사용한 분류 모델임
# SVM은 Support Vector Machine의 약자로, 분류와 회귀 분석에 사용

s.fit(d.data,d.target) # iris 데이터로 학습

new_d=[[6.4,3.2,6.0,2.5],[7.1,3.1,4.7,1.35]] # 101번째와 51번째 샘플을 변형하여 새로운 데이터 생성
res=s.predict(new_d)
print("새로운 2개 샘플의 부류는", res)