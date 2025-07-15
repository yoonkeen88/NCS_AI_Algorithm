from math import sqrt
from queue import Queue

start='806524731'
goal='123456780'

n=int(sqrt(len(goal))) # 보드 크기

class Node():
    def __init__(self,state,path):
        self.state=state
        self.path=path

def print_puzzle(state):
    for i in range(n):
        print(state[n*i:n*(i+1)])

def Move(node):
    p=node.state
    i=p.index('0') # 빈 칸 위치
    child=[]
    if not (i%n==0): # 좌변이 아니면
        q=p[:i-1]+p[i]+p[i-1]+p[i+1:]
        child.append(Node(q,node.path+'L'))
    if not (i%n==n-1): # 우변이 아니면
        q=p[:i]+p[i+1]+p[i]+p[i+2:]
        child.append(Node(q,node.path+'R'))
    if i>=n: # 상변이 아니면
        q=p[:i-n]+p[i]+p[i-n+1:i]+p[i-n]+p[i+1:]
        child.append(Node(q,node.path+'U'))
    if i<n*n-n: # 하변이 아니면
        q=p[:i]+p[i+n]+p[i+1:i+n]+p[i]+p[i+n+1:]
        child.append(Node(q,node.path+'D'))
    return child

print_puzzle(start)
Q=Queue()
root=Node(start,'-') # 루트 노드
Q.put(root)
V=[root.state]
while not Q.empty():
    node=Q.get()
    if node.state==goal:
        print(len(V),'개 노드를 살피고 찾았다!')
        break
    else:
        child=Move(node)
        for j in range(len(child)):
            if child[j].state not in V:
                Q.put(child[j])
                V.append(child[j].state)

print(node.path,"(",len(node.path)-1,")")