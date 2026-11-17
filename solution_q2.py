import gymnasium as gym

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode="human")

# Your code for Q2.2 which Executes Random Policy until 1000 episodes

explore_env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)

state_count = explore_env.observation_space.n
action_count = explore_env.action_space.n

transition_counts = []
reward_sums = []
for state in range(state_count):
    transition_counts.append([])
    reward_sums.append([])
    for action in range(action_count):
        transition_counts[state].append([0] * state_count)
        reward_sums[state].append([0.0] * state_count)

visit_counts = []
for state in range(state_count):
    visit_counts.append([0] * action_count)

for episode in range(1000):
    cur_state, info = explore_env.reset()

    for step in range(100):
        action = explore_env.action_space.sample()
        next_state, reward, terminated, truncated, info = explore_env.step(action)

        transition_counts[cur_state][action][next_state] += 1
        reward_sums[cur_state][action][next_state] += reward
        visit_counts[cur_state][action] += 1

        cur_state = next_state
        if terminated or truncated:
            break

explore_env.close()

# T and R
T = []
R = []
for state in range(state_count):
    T.append([])
    R.append([])
    for action in range(action_count):
        T[state].append([0.0] * state_count)
        R[state].append([0.0] * state_count)
        if visit_counts[state][action] == 0:
            continue
        for next_state in range(state_count):
            T[state][action][next_state] = transition_counts[state][action][next_state] / visit_counts[state][action]
            if transition_counts[state][action][next_state] > 0:
                R[state][action][next_state] = reward_sums[state][action][next_state] / transition_counts[state][action][next_state]

# Your code for Q2.3 which implements Value Iteration

gamma = 0.99
theta = 0.0001
values = [0.0] * state_count

while True:
    biggest_change = 0
    for state in range(state_count):
        old_value = values[state]
        best_value = 0
        for action in range(action_count):
            if(visit_counts[state][action] == 0):
                continue
            action_value = 0
            for next_state in range(state_count):
                action_value += T[state][action][next_state] * (R[state][action][next_state] + gamma * values[next_state])
            if action_value > best_value:
                best_value = action_value
        values[state] = best_value
        biggest_change = max(biggest_change, abs(old_value - values[state]))
    if biggest_change < theta:
        break

print("The solution of Q2.3 is:")
print(f"Value function: {values}")
print()

#Your code for Q2.4 which implements Policy Extraction

# 0 left, 1 down, 2 right, 3 up
policy = []
for state in range(state_count):
    best_action = 0
    best_action_value = 0
    for action in range(action_count):
        if(visit_counts[state][action] == 0):
            continue
        action_value = 0
        for next_state in range(state_count):
                action_value += T[state][action][next_state] * (R[state][action][next_state] + gamma * values[next_state])
        if action_value > best_action_value:
            best_action_value = action_value
            best_action = action
    policy.append(best_action)

print("The solution of Q2.4 is:")
print(f"Policy: {policy}")
print()

# Your code for Q2.5 which executes the optimal policy

cur_state, info = env.reset()

for step in range(100):
    action = policy[cur_state]
    cur_state, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        print(f"Reward = {reward}")
        cur_state, info = env.reset()

env.close()
