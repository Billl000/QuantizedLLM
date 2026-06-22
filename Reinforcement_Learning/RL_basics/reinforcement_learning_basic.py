import gymnasium as gym
#import random
import numpy as np
import matplotlib.pyplot as plt
import pickle

class Agent:
    
    def run(self, steps, is_training=True, render=False):
        env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, render_mode="human" if render else None)
        
        if (is_training):
            q = np.ones((env.observation_space.n, env.action_space.n)) #init a 64x4 array
        else:
            f = open("frozenlake_8x8.pkl", "rb")
            q = pickle.load(f)
            f.close()
        
        lr = 0.1
        gamma = 0.99
        
        epsilon = 1
        epsilon_decay_rate = 1 / (steps * 0.8)
        rng = np.random.default_rng()
        
        reward_per_episode = np.zeros(steps)
        for i in range(steps):
            state = env.reset()[0]
            terminated = False
            truncated = False
    
            while(not terminated and not truncated):
                if (is_training and rng.random() < epsilon):
                    action = env.action_space.sample() #0 = left, 1 = down, 2 = right, 3 = up
                else:
                    action = np.argmax(q[state, :])
                
                new_state, reward, terminated, truncated, _ = env.step(action)
                
                if is_training:
                    q[state, action] = q[state, action] + lr * (reward + gamma * np.max(q[new_state, :]) - q[state, action])
                
                state = new_state

            epsilon = max(0.001, epsilon - epsilon_decay_rate)
        
            
                
            if (reward == 1):
                reward_per_episode[i] = 1
        env.close()
        
        sum_rewards = np.zeros(steps)
        for t in range(steps):
            sum_rewards[t] = np.sum(reward_per_episode[max(0, t-100):(t+1)])
        plt.plot(sum_rewards)
        plt.savefig("frozenlake_8x8.png")
        
        if is_training:
            f = open("frozenlake_8x8.pkl", "wb")
            pickle.dump(q, f)
            f.close()
            
agent = Agent()
#agent.run(15000, is_training=True, render=False)

agent.run(1000, is_training=False, render=True)