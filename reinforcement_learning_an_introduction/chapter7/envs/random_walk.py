import string

import gymnasium as gym
import matplotlib.pyplot as plt
from gymnasium import spaces
from pygame.gfxdraw import filled_circle


class RandomWalkEnv(gym.Env):
    def __init__(self, num_non_terminal_states=5, start_state = 2, rewards=None):
        super(RandomWalkEnv, self).__init__()
        # Define action and observation space (gym.spaces objects)
        self.action_space = spaces.Discrete(2)  # Actions: 0 (left), 1 (right)
        self.observation_space = spaces.Discrete(num_non_terminal_states)  # States: A, B, C, D, E and T

        self.num_non_terminal_states = num_non_terminal_states

        # Initialize state
        self.start_state = start_state
        self.state = start_state  # Start state (e.g. C = index 2)
        self.terminal_state = num_non_terminal_states  # last state is terminal

        self.rewards = rewards or {}

    def reset(self, *, seed=None, options=None):
        self.state = self.start_state  # Reset to start state
        # observation, info
        return self.state, {}

    def step(self, action):
        curr_state = self.state
        # Action: 0 = left, 1 = right
        if action == 0:
            if self.state == 0:
                self.state = self.terminal_state
            else:
                self.state = self.state - 1
        else:
            if self.state != self.terminal_state:
                self.state = self.state + 1

        reward = self.rewards.get((curr_state, self.state), 0)
        terminated = self.state == self.terminal_state

        # observation, reward, terminated, truncated, info
        return self.state, reward, terminated, False, {}

    def render(self):
        fig, ax = plt.subplots(figsize=(self.num_non_terminal_states + 1, 1))
        ax.set_xlim(-1, self.num_non_terminal_states + 1)
        ax.set_ylim(-1, 1)
        # ax.set_xticks(list(string.ascii_letters[:self.num_non_terminal_states])+['T'])
        ax.set_xticks(range(self.num_non_terminal_states + 1))
        ax.set_yticks([])


        # Plot arrows
        for state in range(self.num_non_terminal_states):
            ax.arrow(state + 0.3, 0, 0.4, 0, head_width=0.2, head_length=0.1, fc='black', ec='black')
            ax.arrow(state - 0.3, 0, -0.4, 0, head_width=0.2, head_length=0.1, fc='black', ec='black')

        # Plot states
        for state in range(self.num_non_terminal_states):
            if state == self.state:
                t = ax.text(state, 0, 'X', ha='center', va='center', fontsize=20, color='red')
                t.set_bbox(dict(facecolor='red', alpha=0.1))
            else:
                ax.text(state, 0, 'O', ha='center', va='center', fontsize=20, color='blue')

        # Plot terminal state
        ax.text(self.terminal_state, 0, 'T', ha='center', va='center', fontsize=20, color='green')
        ax.text(-1, 0, 'T', ha='center', va='center', fontsize=20, color='green')

        # Plot rewards
        for (state, next_state), reward in self.rewards.items():
            x = (state + next_state) / 2
            if state == 0 and next_state == self.terminal_state:
                x = -0.5

            t = ax.text(x, 0.3, f'{reward:+}', ha='center', va='center', fontsize=11)


        # remove borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        plt.show()

if __name__ == '__main__':
    # # Create the environment
    env = RandomWalkEnv(num_non_terminal_states=9, start_state=2)

    env.reset()
    env.render()