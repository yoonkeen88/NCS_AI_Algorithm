from math import sqrt
from queue import Queue

start='806524731'
goal='123456780'

n=int(sqrt(len(goal))) # 보드 크기

def print_puzzle(p):
    for i in range(n):
        print(p[n*i:n*(i+1)])

def Move(p):
    i=p.index('0') # 빈 칸 위치
    child=[]
    if not (i%n==0): # 좌변이 아니면
        child.append(p[:i-1]+p[i]+p[i-1]+p[i+1:])
    if not (i%n==n-1): # 우변이 아니면
        child.append(p[:i]+p[i+1]+p[i]+p[i+2:])
    if i>=n: # 상변이 아니면
        child.append(p[:i-n]+p[i]+p[i-n+1:i]+p[i-n]+p[i+1:])
    if i<n*n-n: # 하변이 아니면
        child.append(p[:i]+p[i+n]+p[i+1:i+n]+p[i]+p[i+n+1:])
    return child

print_puzzle(start)
Q=Queue()
Q.put(start) # 루트 노드를 삽입
V=[start]
while not Q.empty():
    node=Q.get()
    if node==goal:
        print(len(V),"개 노드를 방문하고 답을 찾았다.",)
        break
    else:
        child=Move(node)
        for j in range(len(child)):
            if child[j] not in V:
                Q.put(child[j])
                V.append(child[j])