from flappy_bird_env import FlappyBirdEnv
from dqn_agent import DQNAgent
import numpy as np
import time

def train():
    env = FlappyBirdEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size)
    episodes = 1000
    target_update_frequency = 10
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done:
            # 渲染环境
            env.render()
            
            # 选择动作
            action = agent.act(state)
            
            # 执行动作
            next_state, reward, done, _, _ = env.step(action)
            
            # 存储经验
            agent.remember(state, action, reward, next_state, done)
            
            # 训练智能体
            agent.replay()
            
            state = next_state
            total_reward += reward
            steps += 1
            
            # 控制游戏速度
            time.sleep(0.01)
        
        # 更新目标网络
        if episode % target_update_frequency == 0:
            agent.update_target_model()
        
        print(f"Episode: {episode + 1}, Score: {env.score}, Total Reward: {total_reward:.2f}, Steps: {steps}")
        
        # 保存模型
        if (episode + 1) % 100 == 0:
            agent.save(f"flappy_bird_model_{episode + 1}.pth")
    
    env.close()

if __name__ == "__main__":
    train() 