# --- test_logger.py (修正版) ---

import argparse
import numpy as np
from exp_environments import *
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.logger import configure # 导入日志配置工具

def get_args():
    parser = argparse.ArgumentParser()
    # --- 保留了所有必要的参数 ---
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--q', type=float, default=0.5) # 设置为 0.5
    parser.add_argument('--num_clients', type=int, default=100)
    parser.add_argument('--subsample_rate', type=float, default=0.1)
    parser.add_argument('--num_attacker', type=int, default=20)
    parser.add_argument('--num_class', type=int, default=10)
    parser.add_argument('--fl_epoch', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--dataset', type=str, default='MNIST') # 设置为 MNIST
    parser.add_argument('--dummy_batch_size', type=int, default=16)
    parser.add_argument('--attack', type=str, default='EB') # 设置为 EB
    args = parser.parse_args()
    return args

# --- 开始测试 ---
args = get_args()
env = FL_mnist(args)

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# 1. 配置一个新的日志记录器
# 它会把日志同时输出到您的终端(stdout)和TensorBoard文件夹
new_logger = configure(folder="CIFAR10-lmp-q0.1/", format_strings=["stdout", "tensorboard"])

# 2. 创建模型实例，注意 tensorboard_log=None
model = TD3("MlpPolicy", env, buffer_size=1000,
            policy_kwargs={"net_arch" : [256,128]}, tensorboard_log=None,
            verbose=1, gamma=0.99, action_noise=action_noise, learning_rate=1e-5, train_freq=(3, 'step'), batch_size=64)

# 3. 将新的日志记录器应用到模型上
model.set_logger(new_logger)

print('--- 开始测试日志功能，将只运行5轮 ---')
# 我们只训练5轮来快速验证日志表格是否出现
model.learn(total_timesteps=5, log_interval=1)
print('--- 测试结束 ---')