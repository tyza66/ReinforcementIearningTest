# ReinforcementIearningTest
### 强化学习测试

这个项目使用深度强化学习（DQN）来训练一个智能体玩 Flappy Bird 游戏。

## 环境要求

- Python 3.8+
- PyTorch
- Pygame
- NumPy
- Gymnasium

## 安装

1. 克隆仓库：
```bash
git clone <repository-url>
cd flappy-bird-rl
```

2. 使用 uv 安装依赖：
```bash
uv pip install -e .
```

## 使用方法

运行训练脚本：
```bash
python train.py
```

训练过程中，模型会每 100 个回合自动保存一次。保存的模型文件格式为 `flappy_bird_model_X.pth`，其中 X 是回合数。

## 项目结构

- `flappy_bird_env.py`: Flappy Bird 游戏环境的实现
- `dqn_agent.py`: DQN 智能体的实现
- `train.py`: 训练脚本
- `pyproject.toml`: 项目依赖管理文件

## 实现细节

- 使用 PyGame 实现游戏环境
- 使用 PyTorch 实现 DQN 算法
- 状态空间包含：鸟的位置、速度、最近的管道位置和高度
- 动作空间：跳跃或不跳跃
- 奖励设计：
  - 存活奖励：+0.1
  - 碰撞惩罚：-1
  - 通过管道奖励：+1

## 训练过程

训练过程中会显示以下信息：
- 当前回合数
- 得分
- 总奖励
- 步数

模型会随着训练逐渐学习如何玩 Flappy Bird 游戏，探索率（epsilon）会随着训练逐渐降低。
