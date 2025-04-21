import matplotlib.pyplot as plt
import random

class GridWorld:
    def __init__(self, rows, cols, start_state, obstacles, goal):
        self.rows = rows
        self.cols = cols
        self.obstacles = obstacles
        self.goal = goal
        self.start_state = start_state
        self.actions = ['a1', 'a2', 'a3', 'a4']
        self.arrows = {'a1': '↑', 'a3': '↓', 'a4': '←', 'a2': '→'}

    def get_reward(self, state):
        if state == self.goal:
            return 10
        elif state in self.obstacles:
            return  -10
        else:
            return 0

    def step(self, state, action):
        i, j = state
        next_state = (0, 0)
        done = False
        if action == 'a1':
            next_state = (i - 1, j)
        elif action == 'a2':
            next_state = (i, j + 1)
        elif action == 'a3':
            next_state = (i + 1, j)
        elif action == 'a4':
            next_state = (i, j - 1)

        reward = self.get_reward(next_state)
        if next_state == self.goal:
            done = True

        return next_state, reward, done

class Agent:
    def __init__(self, alpha, gamma):
        self.Q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = 0

    def init_Q_table(self, rows, cols, actions):
        for i in range(rows):
            for j in range(cols):
                q_value_dict = {}
                for a in actions:
                    q_value_dict[a] = 0
                self.Q_table[(i, j)] = q_value_dict

    def OptimalAction(self, state):
        actions = self.Q_table[state]
        # 최대 Q 값 찾기
        max_q_value = max(actions.values())
        
        # 최대 Q 값을 가진 모든 행동들 찾기
        optimal_actions_list = [action for action, q_value in actions.items() if q_value == max_q_value]
        optimal_action = random.choice(optimal_actions_list)

        # 최적 행동들 중에서 무작위로 하나 선택
        return optimal_action

    def epsilon_greedy(self, state, epsilon):
        """ε-탐욕 정책을 사용하여 행동 선택"""
        actions = self.Q_table[state]
        optimal_action = self.OptimalAction(state)
        radom_actions = {key: value for key, value in actions.items() if key != optimal_action}
        if random.uniform(0, 1) < epsilon:
            return random.choice(list(radom_actions.keys()))  # 랜덤 행동 (탐험)
        else:
            return optimal_action  # 최적 행동 (탐색)

    def update_Q_table(self, state, action, reward, next_state, virtual_action):
        # SARSA 업데이트
        self.Q_table[state][action] += (
                self.alpha * (
                    reward + self.gamma * self.Q_table[next_state][virtual_action] - self.Q_table[state][action]))

def get_optimal_path(env, agent):
    optimal_path = [env.start_state]
    state = env.start_state

    while state != env.goal:
        optimal_action = agent.OptimalAction(state)
        state, _, done = env.step(state, optimal_action)
        if state[0] < 0 or state[1] < 0 or state[0] >= env.rows or state[1] >= env.cols:
            plot_q_table(env, agent, None)
            break
        if done:
            break
        optimal_path.append(state)

    return optimal_path

def plot_q_table(env, agent, optimal_path):
    fig, ax = plt.subplots(figsize=(14, 14))
    for i in range(env.rows + 1):
        ax.plot([-0.5, env.cols - 0.5], [i - 0.5, i - 0.5], 'k')
    for j in range(env.cols + 1):
        ax.plot([j - 0.5, j - 0.5], [-0.5, env.rows - 0.5], 'k')

    for i in range(env.rows):
        for j in range(env.cols):
            state = (i, j)
            if state in env.obstacles:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="gray"))
            elif state == env.goal:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="green"))
            elif state == env.start_state:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="yellow"))

            ax.text(j, i - 0.4, f"a1:{agent.Q_table[state]['a1']:.2f}",
                    ha='center', va='center', fontsize=12, color='black')
            ax.text(j, i + 0.4, f"a3:{agent.Q_table[state]['a3']:.2f}",
                    ha='center', va='center', fontsize=12, color='black')
            ax.text(j - 0.3, i, f"a4:{agent.Q_table[state]['a4']:.2f}",
                    ha='center', va='center', fontsize=12, color='black')
            ax.text(j + 0.3, i, f"a2:{agent.Q_table[state]['a2']:.2f}",
                    ha='center', va='center', fontsize=12, color='black')

            if state in env.obstacles or state != env.goal:
                optimal_action = agent.OptimalAction(state)
                if optimal_path is not None:
                    if state in optimal_path:
                        arrow_color = 'red'
                        arrow = env.arrows[optimal_action]
                        ax.text(j, i, arrow,
                                ha='center', va='center', fontsize=60, color=arrow_color, fontweight='bold')

    ax.set_xticks(range(env.cols))
    ax.set_yticks(range(env.rows))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim(-0.5, env.cols - 0.5)
    ax.set_ylim(env.rows - 0.5, -0.5)
    ax.grid(True)
    plt.savefig('SARSA.png')
    plt.close()

def SARSA_Train(env, agent, epsilon, num_episodes):
    agent.init_Q_table(env.rows, env.cols, env.actions)
    agent.epsilon = epsilon
    env = env
    for episode in range(num_episodes):
        state = env.start_state
        if agent.epsilon <= 0.1:
            agent.epsilon = 0.1
        else:
            agent.epsilon -= 0.001

        while True:
            action = agent.epsilon_greedy(state, agent.epsilon)
            next_state, reward, done = env.step(state, action)

            if next_state[0] < 0 or next_state[0] > env.rows-1 or next_state[1] <0 or next_state[1] > env.cols-1:
                next_state = state

            virtual_action = agent.epsilon_greedy(next_state, agent.epsilon)
            agent.update_Q_table(state, action, reward, next_state, virtual_action)

            if done:
                break
            state = next_state


if __name__ == '__main__':
    env = GridWorld(5, 5, (0, 1), [(2, 2), (2, 3), (3, 2), (3, 3)], (4, 2))
    agent = Agent(0.01, 0.9)
    SARSA_Train(env, agent, 1.0, 1000)
    optimal_path = get_optimal_path(env, agent)
    plot_q_table(env, agent, optimal_path)

