import pygame
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class FlappyBirdEnv(gym.Env):
    def __init__(self):
        super(FlappyBirdEnv, self).__init__()
        
        # 游戏窗口大小
        self.WINDOW_WIDTH = 400
        self.WINDOW_HEIGHT = 600
        
        # 动作空间：0 = 不跳，1 = 跳
        self.action_space = spaces.Discrete(2)
        
        # 观察空间：鸟的位置(y)，速度，最近的管道位置和高度
        self.observation_space = spaces.Box(
            low=np.array([0, -10, 0, 0, 0]),
            high=np.array([self.WINDOW_HEIGHT, 10, self.WINDOW_WIDTH, self.WINDOW_HEIGHT, self.WINDOW_HEIGHT]),
            dtype=np.float32
        )
        
        # 初始化 Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        pygame.display.set_caption('Flappy Bird RL')
        
        # 游戏参数
        self.gravity = 0.5
        self.jump_strength = -10
        self.pipe_speed = 3
        self.pipe_gap = 150
        self.pipe_frequency = 1500  # 毫秒
        self.pipe_width = 50
        self.bird_radius = 20
        
        self.reset()

    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # 初始化鸟的位置和速度
        self.bird_y = self.WINDOW_HEIGHT // 2
        self.bird_velocity = 0
        self.bird_x = 50  # 小鸟水平位置保持不动
        
        # 初始化管道
        self.pipes = []
        self.last_pipe = pygame.time.get_ticks()
        
        # 初始化分数
        self.score = 0
        
        return self._get_observation(), {}

    def step(self, action):
        # 初始化奖励与终止标志
        reward = 0.1  # 存活奖励
        done = False

        # 处理动作
        if action == 1:
            self.bird_velocity = self.jump_strength
        
        # 更新鸟的位置
        self.bird_velocity += self.gravity
        self.bird_y += self.bird_velocity
        
        # 检查是否撞到地面或天花板
        if self.bird_y - self.bird_radius < 0 or self.bird_y + self.bird_radius > self.WINDOW_HEIGHT:
            done = True
            reward = -1
        
        # 生成新管道
        current_time = pygame.time.get_ticks()
        if current_time - self.last_pipe > self.pipe_frequency:
            pipe_height = random.randint(100, self.WINDOW_HEIGHT - 100)
            self.pipes.append({
                'x': self.WINDOW_WIDTH,
                'top_height': pipe_height,
                'bottom_y': pipe_height + self.pipe_gap
            })
            self.last_pipe = current_time
        
        # 更新管道位置并检测是否已通过管道
        for pipe in self.pipes[:]:
            pipe['x'] -= self.pipe_speed

            # 当管道完全离开屏幕左侧时移除
            if pipe['x'] < -self.pipe_width:
                self.pipes.remove(pipe)

            # 判断小鸟是否成功穿过管道（只计算一次）
            if (not pipe.get('passed', False)) and pipe['x'] + self.pipe_width < self.bird_x:
                pipe['passed'] = True
                self.score += 1
                reward += 1  # 通过管道奖励
        
        # 碰撞检测（使用矩形碰撞，提升精度）
        bird_rect = pygame.Rect(self.bird_x - self.bird_radius,
                                 self.bird_y - self.bird_radius,
                                 self.bird_radius * 2,
                                 self.bird_radius * 2)

        for pipe in self.pipes:
            top_rect = pygame.Rect(pipe['x'], 0, self.pipe_width, pipe['top_height'])
            bottom_rect = pygame.Rect(pipe['x'], pipe['bottom_y'], self.pipe_width,
                                      self.WINDOW_HEIGHT - pipe['bottom_y'])

            if bird_rect.colliderect(top_rect) or bird_rect.colliderect(bottom_rect):
                done = True
                reward = -1
                break
        
        return self._get_observation(), reward, done, False, {}

    def _get_observation(self):
        # 找到最近的管道
        nearest_pipe = None
        nearest_pipe_x = float('inf')
        
        for pipe in self.pipes:
            if 0 < pipe['x'] < nearest_pipe_x:
                nearest_pipe = pipe
                nearest_pipe_x = pipe['x']
        
        if nearest_pipe is None:
            nearest_pipe_x = self.WINDOW_WIDTH
            nearest_pipe_top = self.WINDOW_HEIGHT // 2
        
        return np.array([
            self.bird_y,
            self.bird_velocity,
            nearest_pipe_x,
            nearest_pipe['top_height'] if nearest_pipe else self.WINDOW_HEIGHT // 2,
            nearest_pipe['bottom_y'] if nearest_pipe else self.WINDOW_HEIGHT // 2 + self.pipe_gap
        ], dtype=np.float32)

    def render(self):
        # 处理 Pygame 事件，避免窗口"未响应"
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                import sys
                sys.exit()

        self.screen.fill((135, 206, 235))  # 天蓝色背景
        
        # 绘制鸟（黄色球）
        pygame.draw.circle(self.screen, (255, 255, 0), (self.bird_x, int(self.bird_y)), self.bird_radius)
        
        # 绘制管道
        for pipe in self.pipes:
            # 上管道
            pygame.draw.rect(self.screen, (0, 255, 0),
                           (pipe['x'], 0, self.pipe_width, pipe['top_height']))
            # 下管道
            pygame.draw.rect(self.screen, (0, 255, 0),
                           (pipe['x'], pipe['bottom_y'], self.pipe_width,
                            self.WINDOW_HEIGHT - pipe['bottom_y']))
        
        pygame.display.flip()

    def close(self):
        pygame.quit() 