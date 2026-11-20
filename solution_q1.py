import gymnasium as gym
import random

env = gym.make('Blackjack-v1', natural=False, sab=False, render_mode="human")

###YOUR Q-LEARNING CODE BEGINS

train_env = gym.make('Blackjack-v1', natural=False, sab=False)

# 0 stick, 1 hit
q_values = []
for player_sum in range(32):
    q_values.append([])
    for dealer_card in range(11):
        q_values[player_sum].append([])
        for usable_ace in range(2):
            q_values[player_sum][dealer_card].append([0.0, 0.0])

def choose_action(cur_state, epsilon):
    player_sum, dealer_card, usable_ace = cur_state
    if(random.random() < epsilon):
        return train_env.action_space.sample()

    if q_values[player_sum][dealer_card][usable_ace][0] >= q_values[player_sum][dealer_card][usable_ace][1]:
        return 0
    return 1

alpha = 0.1
gama = 0.9
epsilon = 1.0
min_epsilon = 0.05
num_episodes = 20000

win_count = 0
loss_count = 0
draw_count = 0
for episode in range(num_episodes):
    cur_state, info = train_env.reset()
    done = False

    while not done:
        player_sum, dealer_card, usable_ace = cur_state
        action = choose_action(cur_state, epsilon)
        next_state, reward, terminated, truncated, info = train_env.step(action)

        old_q = q_values[player_sum][dealer_card][usable_ace][action]
        if terminated or truncated:
            target_q = reward
        else:
            next_sum, next_dealer, next_ace = next_state
            next_q = max(q_values[next_sum][next_dealer][next_ace])
            target_q = reward + gama * next_q

        q_values[player_sum][dealer_card][usable_ace][action] = old_q + alpha * (target_q - old_q)
        cur_state = next_state
        done = terminated or truncated

    epsilon = epsilon * 0.999
    if epsilon < min_epsilon:
        epsilon = min_epsilon

    if reward > 0:
        win_count += 1
        print(f"Episode {episode + 1}: won")
    elif reward < 0:
        loss_count += 1
        print(f"Episode {episode + 1}: lost")
    else:
        draw_count += 1
        print(f"Episode {episode + 1}: draw")

print("The solution of Q1 is:")
print(f"Total win rate = {win_count / num_episodes}")
print(f"Wins = {win_count}")
print(f"Losses = {loss_count}")
print(f"Draws = {draw_count}")
print()

#next part
for episode in range(5):
    cur_state, info = env.reset()
    done = False
    while not done:
        action = choose_action(cur_state, 0)
        cur_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    if reward > 0:
        print(f"Test episode {episode + 1}: won")
    elif reward < 0:
        print(f"Test episode {episode + 1}: lost")
    else:
        print(f"Test episode {episode + 1}: draw")

train_env.close()
env.close()


