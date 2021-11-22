from DQNAgent import DQNAgent
import gym


env_name = "CartPole-v1"
env = gym.make(env_name)
print("Observation space: ", env.observation_space)
print("Action space: ", env.action_space)

agent = DQNAgent(env)
state = env.reset()

num_eps = 2000

for ep in range(num_eps):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        agent.train(state, action, next_state, reward, done)
        env.render()
        total_reward += reward
        state = next_state

    print("Episode: {}, total_reward: {:.2f}".format(ep, total_reward))