import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from gym.wrappers.monitoring.video_recorder import VideoRecorder

# Hyperparameters
GAMMA = 0.99
LR = 0.001
NUM_EPISODES = 10000

# Cartpole Environment
env = gym.make("CartPole-v1", render_mode="rgb_array")
env = gym.wrappers.TimeLimit(env.unwrapped, max_episode_steps=500)
env.env.x_threshold = 4.8


# Actor-Critic Model
class TD_ActorCritic(nn.Module):
    def __init__(self, input_dim, action_output_dim, value_output_dim):
        super(TD_ActorCritic, self).__init__()
        self.actor_fc = nn.Linear(input_dim, 24)
        self.actor_out_layer = nn.Linear(24, action_output_dim)

        self.critic_fc1 = nn.Linear(input_dim, 24)
        self.critic_out_layer = nn.Linear(24, value_output_dim)

        nn.init.uniform_(self.actor_out_layer.weight, -1e-3, 1e-3)
        nn.init.uniform_(self.actor_out_layer.bias, -1e-3, 1e-3)
        nn.init.uniform_(self.critic_out_layer.weight, -1e-3, 1e-3)
        nn.init.uniform_(self.critic_out_layer.bias, -1e-3, 1e-3)

    def pi_prob(self, x):
        l1 = torch.tanh(self.actor_fc(x))
        l2 = self.actor_out_layer(l1)
        pi_prob = F.softmax(l2, dim=-1)
        return pi_prob

    def value(self, x):
        l1 = torch.tanh(self.critic_fc1(x))
        value = self.critic_out_layer(l1)
        return value


def plot_episode_rewards(episode_rewards, filename="./TD-ACTOR_CRITIC.png"):
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
        print("Episodes {:3d}-{:3d}: Average Reward = {:5.2f}".format(i * group_size + 1, (i + 1) * group_size,
                                                                      avg_reward))

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


# Record Video of Trained Model
def record_video(model, env, filename):
    video_recorder = VideoRecorder(env, filename)

    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    done = False

    while not done:
        action = model.pi_prob(state).argmax().item()
        state, reward, fail, success, info = env.step(action)
        done = fail or success
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        video_recorder.capture_frame()

    env.close()
    video_recorder.close()
    print(f"Video saved as {filename}")

class TransitionData:
    def __init__(self):
        self.transition_list = []

    def queue(self, transition):
        self.transition_list.append(transition)

    def dequeue(self):
        states, actions, next_states, rewards, dones = zip(*self.transition_list)
        state_batch = torch.cat(states)
        action_batch = torch.stack(actions)
        next_state_batch = torch.cat(next_states)
        reward_batch = torch.stack(rewards)
        done_batch = torch.stack(dones)

        return state_batch, action_batch, next_state_batch, reward_batch, done_batch

    def reset(self):
        self.transition_list = []

# Training Function
def TD_ActorCritic_train(env):
    n_states = env.observation_space.shape[0] # CartPole state의 크기 4x1
    n_actions = env.action_space.n # CartPole에서 가능한 action 개수(왼쪽, 오른쪽)
    output_value_dim =1 # 네트워크에서 추출되는 value 값 크기 (scalar)
    batch_size =10 # 학습 주기

    # TD_ActorCritic network 불러오기
    actor_critic_network = TD_ActorCritic(n_states, n_actions, output_value_dim)

    optimizer = optim.Adam(actor_critic_network.parameters(), lr=LR)
    episode_rewards = []

    transition_list = TransitionData()

    success_threshhold = 450
    complete_step = 0

    for episode in range(NUM_EPISODES):
        state, info = env.reset()
        total_reward = 0

        done = False
        while not done:
            for i in range(batch_size):
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                pi_prob = actor_critic_network.pi_prob(state_tensor)
                pi_prob_dist = torch.distributions.Categorical(pi_prob)
                action = pi_prob_dist.sample().item()
                next_state, reward, fail, success, info = env.step(action)

                done = fail or success
                total_reward += reward
                reward = 0.1 if not done or total_reward == 500 else -1

                action_tensor = torch.tensor(action, dtype=torch.float32)
                next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                reward_tensor = torch.tensor(reward, dtype=torch.float32)
                done_tensor = torch.tensor(done, dtype=torch.float32)

                transition = (state_tensor, action_tensor, next_state_tensor, reward_tensor, done_tensor)
                transition_list.queue(transition)
                state = next_state
                if done:
                    break

            # 학습 시작
            states_batch, actions_batch, next_states_batch, rewards_batch, dones_batch = transition_list.dequeue()

            policy_batch = actor_critic_network.pi_prob(states_batch)
            action_batch = actions_batch.long().unsqueeze(1)
            action_prob_batch = policy_batch.gather(1, action_batch).squeeze(1)

            next_state_value_batch = actor_critic_network.value(next_states_batch).squeeze(1)
            TD_target_batch = rewards_batch + GAMMA * next_state_value_batch * (1 - dones_batch)

            state_value_batch = actor_critic_network.value(states_batch).squeeze(1)
            delta_batch = TD_target_batch - state_value_batch

            actor_loss_batch = -torch.log(action_prob_batch) * delta_batch.detach()
            critic_loss = F.mse_loss(TD_target_batch.detach(), state_value_batch)

            loss = actor_loss_batch.mean() + critic_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            transition_list.reset()

        episode_rewards.append(total_reward)

        if episode % 25 == 0:
            print(
                f"Episode {episode}, Reward {total_reward}")
            if total_reward >= success_threshhold:
                complete_step += 1
            else:
                complete_step = 0

            if complete_step >= 10:
                break

    plot_episode_rewards(episode_rewards)
    env.close()
    return actor_critic_network


if __name__ == '__main__':
    TD_ActorCritic_Network = TD_ActorCritic_train(env)
    record_video(TD_ActorCritic_Network, env, './TDActor_critic.mp4')
