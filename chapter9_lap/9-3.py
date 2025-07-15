import gym
import numpy as np

env=gym.make('FrozenLake-v1',is_slippery=False, render_mode = 'human') # 환경 생성
Q=np.zeros([env.observation_space.n,env.action_space.n]) # Q 배열 초기화

rho=0.8 # 학습률
lamda=0.99 # 할인율

n_episode=2000
length_episode=100

# 최적 행동 가치 함수 찾기
for i in range(n_episode):
    # s=env.reset() # 새로운 에피소드 시작                                                                                                           │
    s, _ = env.reset() # 새로운 에피소드 시작
    for j in range(length_episode):
        argmaxs=np.argwhere(Q[s,:]==np.amax(Q[s,:])).flatten().tolist()
        a=np.random.choice(argmaxs)
        # s1,r,done,_=env.step(a)  # 최신 Gym은 step 함수가 5개의 값을 반환함
        s1,r,done,_,_=env.step(a)
        Q[s,a]=Q[s,a]+rho*(r+lamda*np.max(Q[s1,:])-Q[s,a]) # 식 (9.18)
        s=s1
        if done:
            break

np.set_printoptions(precision=2)
print(Q)
print("최적 행동 가치 함수 Q(s,a):")

print(len(Q))  # 상태의 개수
print(len(Q[0]))  # 행동의 개수
# Q 가 의미하는 것: 각 상태 s 에서 각 행동 a 를 취했을 때의 Q 값
# Q[s,a] 는 상태 s 에서 행동 a 를 취했을 때의 Q 값
