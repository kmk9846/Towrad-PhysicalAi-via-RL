import gym
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import collections
import matplotlib.pyplot as plt
from gym.wrappers.monitoring.video_recorder import VideoRecorder

# Hyperparameters
GAMMA = 0.99
LR = 0.001
BATCH_SIZE = 32
MEMORY_SIZE = 2000
EPSILON_START = 1.0
EPSILON_END = 0.001
EPSILON_DECAY = 0.999
TARGET_UPDATE = 10
NUM_EPISODES = 10000

# Cartpole Environment
env = gym.make("CartPole-v1", render_mode="rgb_array")
env = gym.wrappers.TimeLimit(env.unwrapped, max_episode_steps=500)
env.env.x_threshold = 4.8

class Q_Network(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Q_Network, self).__init__()
        self.fc1 = nn.Linear(input_dim, 24)
        self.fc2 = nn.Linear(24, 24)
        self.out_layer = nn.Linear(24, output_dim)
        nn.init.uniform_(self.out_layer.weight, -1e-3, 1e-3)
        nn.init.uniform_(self.out_layer.bias, -1e-3, 1e-3)

    def forward(self, x):
        l1 = F.relu(self.fc1(x))
        l2 = F.relu(self.fc2(l1))
        return self.out_layer(l2)

class ReplayBuffer:
    def __init__(self, capacity):
        self.replay_buffer = collections.deque(maxlen=capacity)

    def push(self, transition):
        self.replay_buffer.append(transition)

    def random_sample(self, batch_size):
        transition_list = random.sample(self.replay_buffer, batch_size)

        states, actions, next_states, rewards, dones = zip(*transition_list)
        state_batch = torch.cat(states)
        action_batch = torch.stack(actions)
        next_state_batch = torch.cat(next_states)
        reward_batch = torch.stack(rewards)
        done_batch = torch.stack(dones)

        return state_batch, action_batch, next_state_batch, reward_batch, done_batch

    def __len__(self):
        return len(self.replay_buffer)

class EpsilonGreedyPolicy:
    def __init__(self, start, end):
        self.epsilon = start
        self.end = end

    def epsilon_update(self):
        if self.epsilon > self.end:
            self.epsilon *= EPSILON_DECAY

    def select_action(self, network, state, num_actions):
        if random.random() < self.epsilon:
            return random.randrange(num_actions)
        return network(state).argmax().item()

def plot_episode_rewards(episode_rewards, filename="./DQN.png"):
    group_size = 10
    group_avg_rewards = []
    group_episodes = []
    total_episodes = len(episode_rewards)

    num_groups = total_episodes // group_size
    for i in range(num_groups):
        group = episode_rewards[i * group_size:(i + 1) * group_size]
        avg_reward = sum(group) / group_size
        group_avg_rewards.append(avg_reward)
        group_episodes.append((i + 1) * group_size)

    plt.figure(figsize=(10, 5))
    plt.plot(group_episodes, group_avg_rewards, label="Average Reward", color="blue", alpha=0.7)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.xlim((0, total_episodes))
    plt.ylim((0, 550))
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def record_video(model, env, filename):
    video_recorder = VideoRecorder(env, filename)
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    done =False

    while not done:
        action = model(state).argmax().item()
        state, reward, fail, success, info = env.step(action)
        done = fail or success
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        video_recorder.capture_frame()

    env.close()
    video_recorder.close()
    print(f"Video saved as {filename}")

    # DQN 으로 Cartpole 학습
def DQN_train(env):
    n_states = env.observation_space.shape[0]  # CartPole state의 크기 4x1
    n_actions = env.action_space.n  # CartPole에서 가능한 action 개수(왼쪽, 오른쪽)

    behavior_network = Q_Network(n_states, n_actions)  # behavior_network 설정
    target_network = Q_Network(n_states, n_actions)  # target_network 설정
    # behavior_network로부터 parameter 복사
    target_network.load_state_dict(behavior_network.state_dict())
    target_network.eval()  # 학습이 아니라 evaluation에만 사용

    optimizer = optim.Adam(behavior_network.parameters(), lr=LR)
    # replay_buffe를 MEMORY_SIZE 크기로 설정
    replay_buffer = ReplayBuffer(MEMORY_SIZE)

    policy = EpsilonGreedyPolicy(EPSILON_START, EPSILON_END)
    episode_rewards = []

    success_threshhold = 450
    complete_step = 0  # 연속 성공횟수 기록

    for episode in range(NUM_EPISODES):
        state, info = env.reset()  # CartPole state 초기화
        total_reward = 0
        done = False

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action = policy.select_action(behavior_network, state_tensor, n_actions)
            next_state, reward, fail, success, info = env.step(action)

            done = fail or success
            total_reward += reward
            # 성공시 보상 0.1, 실패시 보상 -1
            reward = 0.1 if not done or total_reward == 500 else -1

            action_tensor = torch.tensor(action, dtype=torch.float32)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            reward_tensor = torch.tensor(reward, dtype=torch.float32)
            done_tensor = torch.tensor(done, dtype=torch.float32)
            transition = (state_tensor, action_tensor, next_state_tensor, reward_tensor, done_tensor)
            # tensor 로 변경 후, replay_buffer에 append
            replay_buffer.push(transition)
            state = next_state

        episode_rewards.append(total_reward)
        # DQN 으로 Cartpole 학습
        # transition이 1000개 이상이면, parameter update
        if len(replay_buffer) > 1000:
            policy.epsilon_update()  # epsilon 값 decay

            # 에피소드 진행중 10개의 mini_batch로 나누어 학습
            for mini_batch in range(10):
                # replay_buffert에서 transition batch 로 가져오기
                state_batch, action_batch, next_state_batch, reward_batch, done_batch = replay_buffer.random_sample(BATCH_SIZE)

                # transition에서 실제로 실행한 action에 대한 Q-value return
                action_batch = action_batch.long().unsqueeze(1)
                select_action_Q_value_batch = behavior_network(state_batch).gather(1, action_batch).squeeze()

                # target_network에서 다음 state에 대한 Q-value 중 최대값 return
                target_network_value_batch = target_network(next_state_batch).detach()
                max_target_network_value_batch = target_network_value_batch.max(1)[0]

                # target_Q_value 구하기
                target_Q_value_batch = reward_batch + GAMMA * max_target_network_value_batch * (1- done_batch)

                # 최종 loss 구하기
                loss = F.mse_loss(select_action_Q_value_batch, target_Q_value_batch)
                # parameter update 진행
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # TARGET_UPDATE 주기마다 target_network update
        if episode % TARGET_UPDATE == 0:
            target_network.load_state_dict(behavior_network.state_dict())

        # 25번의 episode 마다 reward를 확인 후, 연속 10번 성공시, 학습 종료
        if episode % 25 == 0:
            print(f"Episode {episode}, Reward {total_reward},Buffer {len(replay_buffer)}, Epsilon {policy.epsilon: .4f}")
            if total_reward >= success_threshhold:
                complete_step += 1
            else:
                complete_step = 0
            if complete_step >= 10:
                break

    plot_episode_rewards(episode_rewards)
    env.close()
    return behavior_network

if __name__ == '__main__':
    DQN_Network = DQN_train(env)
    record_video(DQN_Network, env, "./Cartpole_DQN.mp4")
