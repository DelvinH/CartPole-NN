
import gym


env = gym.make('CartPole-v0')
env.reset()
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        #print(observation)
        action = env.action_space.sample() # 1 moves right, two moves left
        print(action)
        observation, reward, done, info = env.step(action)
        #print(done)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()