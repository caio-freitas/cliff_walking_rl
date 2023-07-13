from abc import ABC, abstractmethod
import numpy as np

class Agent(ABC):
    def __init__(self, env):
        self.env = env
        self.state = env.reset()
        self.total_reward = 0.0
    
    @abstractmethod
    def step(self):
        raise NotImplementedError

    def reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0


class RandomAgent(Agent):
    def step(self):
        self.state, reward, done, _ = self.env.step(self.env.action_space.sample())
        self.total_reward += reward
        if done:
            self.reset()
        return done
    
class SARSAAgent(Agent):
    def __init__(self, env, gamma=0.99, alpha=0.01, epsilon=0.8):
        super().__init__(env)
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        # discretize observation and action space
        print("Observation space:", env.observation_space)
        print("Action space:", env.action_space)
        
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))
        
        print("Q shape:", self.Q.shape)
        self.action = self.pick_action(self.state)

    def update_q(self, state, action, reward, next_state, next_action):
        state = self._get_int_value(state)
        next_state = self._get_int_value(next_state)
        action = self._get_int_value(action)
        next_action = self._get_int_value(next_action)

        # self.Q = self.Q + self.alpha * (reward + self.gamma * self.Q[next_state][next_action] - self.Q[state][action])
        self.Q[state][action] = self.Q[state][action] + self.alpha * (reward + self.gamma * self.Q[next_state][next_action] - self.Q[state][action])
    def _get_int_value(self, var):
        try:
            return var[0]
        except:
            return var

    def pick_action(self, state, update_q=False):
        state = self._get_int_value(state)
        if np.random.random() < self.epsilon and not update_q:
            self.epsilon *= 0.9999
            # return random tuple of actions
            print("Exploring...")
            return self.env.action_space.sample()

        max_action = np.where(self.Q[state] == np.max(self.Q[state]))

        # pick random action where the value is the max
        max_action = np.random.choice(max_action[0])

        # print("max_action:", max_action)

        return max_action
    

    def step(self):
        # convert action to Discrete
        self.action = self._get_int_value(self.action)

        prev_action = self.action
        print("action:", self.action)
        (next_state, reward, done, info, p) = self.env.step(self.action)
        print("reward:", reward)
        self.action = self.pick_action(next_state)
        self.update_q(self.state, prev_action, reward, next_state, self.action)
        self.state = next_state
        self.total_reward += reward
        # print(f"total_reward: {self.total_reward}")
        return done
    
    def save(self, filename):
        np.save(filename, self.Q)

    def load(self, filename):
        self.Q = np.load(filename)

    def reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0


class QLearningAgent(Agent):
    def __init__(self, env, gamma=0.99, alpha=0.01, epsilon=0.8):
        super().__init__(env)
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        # discretize observation and action space
        print("Observation space:", env.observation_space)
        print("Action space:", env.action_space)
        
        self.Q = np.zeros((env.observation_space.n, env.action_space.n))
        
        print("Q shape:", self.Q.shape)
        self.action = self.pick_action(self.state)

    def update_q(self, state, action, reward, next_state, next_action):
        state = self._get_int_value(state)
        next_state = self._get_int_value(next_state)
        action = self._get_int_value(action)
        next_action = self._get_int_value(next_action)

        # self.Q = self.Q + self.alpha * (reward + self.gamma * self.Q[next_state][next_action] - self.Q[state][action])
        self.Q[state][action] = self.Q[state][action] + self.alpha * (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action])
    def _get_int_value(self, var):
        try:
            return var[0]
        except:
            return var

    def pick_action(self, state, update_q=False):
        state = self._get_int_value(state)
        if np.random.random() < self.epsilon and not update_q:
            self.epsilon *= 0.9999
            # return random tuple of actions
            print("Exploring...")
            return self.env.action_space.sample()

        max_action = np.where(self.Q[state] == np.max(self.Q[state]))

        # pick random action where the value is the max
        max_action = np.random.choice(max_action[0])

        # print("max_action:", max_action)

        return max_action
    

    def step(self):
        # convert action to Discrete
        self.action = self._get_int_value(self.action)

        prev_action = self.action
        print("action:", self.action)
        (next_state, reward, done, info, p) = self.env.step(self.action)
        print("reward:", reward)
        self.action = self.pick_action(next_state)
        self.update_q(self.state, prev_action, reward, next_state, self.action)
        self.state = next_state
        self.total_reward += reward
        # print(f"total_reward: {self.total_reward}")
        return done
    
    def save(self, filename):
        np.save(filename, self.Q)

    def load(self, filename):
        self.Q = np.load(filename)

    def reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0
