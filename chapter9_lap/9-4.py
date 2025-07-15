import gym
import numpy as np

env=gym.make('FrozenLake-v0',is_slippery=False) # 환경 생성
Q=np.zeros([env.observation_space.n,env.action_space.n]) # Q 배열 초기화

rho=0.90 # 학습률
lamda=0.99 # 할인율
eps=1.0 # 엡시론
eps_decay=0.999 # 삭감 비율

n_episode=3000
length_episode=100

# 최적 행동 가치 함수 찾기(탐사와 탐험의 균형 추구)
for i in range(n_episode):
    s=env.reset() # 새로운 에피소드 시작
    for j in range(length_episode):
        r=np.random.random()
        eps=max(0.01,eps*eps_decay) # 엡시론을 조금씩 줄여나감
        if(r<eps): # eps 비율만큼 임의 선택
            a=np.random.randint(0,env.action_space.n)
        else:
            argmaxs=np.argwhere(Q[s,:]==np.amax(Q[s,:])).flatten().tolist()
            a=np.random.choice(argmaxs)
        s1,r,done,_=env.step(a)
        Q[s,a]=Q[s,a]+rho*(r+lamda*np.max(Q[s1,:])-Q[s,a])
        s=s1
        if done:
            break

np.set_printoptions(precision=2)
print(Q)