import os

# 원본 IMDB 데이터 읽기
directory='chapter8_lap/data/aclImdb/aclImdb/test'
x=[];y=[]
for c in ['neg','pos']:
    curr=os.path.join(directory,c)
    for fname in os.listdir(curr):
        if fname[-4:]=='.txt':
            f=open(os.path.join(curr,fname),encoding='utf8')
            x.append(f.read())
            y.append(c)
            
print("첫 번째 댓글:",x[0]); print("첫 번째 평가:",y[0])
print("마지막 댓글:",x[-1]); print("마지막 평가:",y[-1])