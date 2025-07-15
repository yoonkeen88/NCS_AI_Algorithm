import gym
# 환경 불러오기
env = gym.make("FrozenLake-v1", is_slippery=False, 
            #    render_mode='human' 이거하면 pygame 창이 뜸
               ) # gym 라이브러리가 업데이트 되면서 v0 -> v1 으로 바뀜 
print(env.observation_space) # Discrete(16) - 4x4 셀
print(env.action_space) # Discrete(4) - 상하좌우 4가지 행동

n_trial = 100  # 에피소드 길이

# 에피소드 수집
obs, info = env.reset()  # 최신 Gym은 reset도 반환값이 2개
episode = []

for i in range(n_trial):
    action = env.action_space.sample()  # 랜덤 행동
    obs, reward, terminated, truncated, info = env.step(action) # step 함수는 5개의 값을 반환함
    # obs: 현재 상태(셀 번호), reward: 보상, terminated: 에피소드 종료 여부, truncated: 시간 초과 여부, info: 추가 정보
    print(f"Action: {action}, Reward: {reward}, Obs: {obs}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")
    done = terminated or truncated  # 종료 조건
    episode.append([action, reward, obs])
    env.render()
    if done:
        break

for i in episode:
    print(i)  # 각 행동, 보상, 상태 출력
# print(episode)
# 출력은 [action, reward, obs] 형태로 되어 있음 obs 는 셀 번호
env.close()

