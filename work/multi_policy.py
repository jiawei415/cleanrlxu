# https://github.com/facebookresearch/torchbeast/blob/master/torchbeast/core/environment.py

import numpy as np
from collections import deque
import gym
from gym import spaces
import cv2
cv2.ocl.setUseOpenCL(False)


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip       = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True, dict_space_key=None):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        if action >= self.env.action_space.n:
            action = 0
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0

class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]

def wrap_atari(env, max_episode_steps=None):
    assert 'NoFrameskip' in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)

    assert max_episode_steps is None

    return env

def wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=False, scale=False):
    """Configure environment for DeepMind-style Atari.
    """
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env


class ImageToPyTorch(gym.ObservationWrapper):
    """
    Image shape to channels x weight x height
    """

    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(old_shape[-1], old_shape[0], old_shape[1]),
            dtype=np.uint8,
        )

    def observation(self, observation):
        return np.transpose(observation, axes=(2, 0, 1))

def wrap_pytorch(env):
    return ImageToPyTorch(env)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import argparse
from distutils.util import strtobool
import numpy as np
import gym
from gym.wrappers import TimeLimit, Monitor
# import pybullet_envs
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import random
import pickle
import os
import math
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PPO agent')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--envs-name', nargs='+', default=['Tennis', 'Pong', 'Tennis', 'Pong'],
                        help='the id of the gym environment')
    parser.add_argument('--envs-ckpt', nargs='+', default=['20210510160835', '20210425165437','20210510160835','20210425165437'],
                        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4, # 0.00025
                        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1, # 2 for breakout 1.5 for tennis and pong
                        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=10000000,
                        help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--prod-mode', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='weather to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
                        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help="the entity (team) of wandb's project")
    parser.add_argument('--ckpt', type=str, default=None,
                        help="load checkpoint path")

    # Algorithm specific arguments
    parser.add_argument('--n-minibatch', type=int, default=4,
                        help='the number of mini batch')
    parser.add_argument('--num-envs', type=int, default=4,
                        help='the number of parallel game environment')
    parser.add_argument('--num-steps', type=int, default=128,
                        help='the number of steps per game environment')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='the discount factor gamma')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='the lambda for the general advantage estimation')
    parser.add_argument('--ent-coef', type=float, default=0.01,
                        help="coefficient of the entropy")
    parser.add_argument('--fea-coef', type=float, default=0.2,
                        help="coefficient of the feature loss")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='the maximum norm for the gradient clipping')
    parser.add_argument('--clip-coef', type=float, default=0.1,
                        help="the surrogate clipping coefficient")
    parser.add_argument('--update-epochs', type=int, default=4,
                         help="the K epochs to update the policy")
    parser.add_argument('--kle-stop', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                         help='If toggled, the policy updates will be early stopped w.r.t target-kl')
    parser.add_argument('--kle-rollback', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                         help='If toggled, the policy updates will roll back to previous policy if KL exceeds target-kl')
    parser.add_argument('--target-kl', type=float, default=0.03,
                         help='the target-kl variable that is referred by --kl')
    parser.add_argument('--gae', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                         help='Use GAE for advantage computation')
    parser.add_argument('--norm-adv', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                          help="Toggles advantages normalization")
    parser.add_argument('--anneal-lr', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                          help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument('--clip-vloss', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                          help='Toggles wheter or not to use a clipped loss for the value function, as per the paper.')

    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())



class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        super(VecPyTorch, self).__init__(venv)
        self.device = device

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info

now_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
experiment_name = f"multi_policy_{now_time}"
writer = SummaryWriter(f"results/{experiment_name}/logs")
writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
        '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))
ckpt_save_path = f"results/{experiment_name}/checkpoints"
os.makedirs(ckpt_save_path, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic

def make_env(gym_id, seed, idx):
    if "NoFrameskip-v4" not in gym_id:
        gym_id = gym_id + "NoFrameskip-v4"
    def thunk():
        env = gym.make(gym_id)
        env = wrap_atari(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if args.capture_video:
            if idx == 0:
                env = Monitor(env, f'videos/{experiment_name}')
        env = wrap_pytorch(
            wrap_deepmind(
                env,
                clip_rewards=True,
                frame_stack=True,
                scale=False,
            )
        )
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk

envs_name = args.envs_name # ['Tennis', 'Pong', 'Tennis', 'Pong']
envs = VecPyTorch(DummyVecEnv([make_env(game_name, args.seed+i, i) for i, game_name in enumerate(envs_name)]), device)
source_envs = [VecPyTorch(DummyVecEnv([make_env(game_name, args.seed+i, i)]), device) for i, game_name in enumerate(envs_name)]
source_ckpts = args.envs_ckpt # ['20210510160835', '20210425165437','20210510160835','20210425165437']
source_paths = [f"results/{game_name}NoFrameskip-v4_mppo_{ckpt}/checkpoints/best_model.pt" for game_name, ckpt in zip(envs_name, source_ckpts)]
assert isinstance(envs.action_space, Discrete), "only discrete action space is supported"
print(f"envs name: {envs_name}")

args.num_envs = len(envs_name)
args.batch_size = int(args.num_envs * args.num_steps)
args.minibatch_size = int(args.batch_size // args.n_minibatch)

class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Teacher(nn.Module):
    def __init__(self, envs, frames=4):
        super(Teacher, self).__init__()
        self.network = nn.Sequential(
            Scale(1/255),
            layer_init(nn.Conv2d(frames, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(3136, 512)),
            nn.ReLU()
        )
        # self.actor = layer_init(nn.Linear(512, envs.action_space.n), std=0.01)
        self.actor = layer_init(nn.Linear(512, 18), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)
        self.n = envs.action_space.n

    def get_action(self, x, action=None):
        x = self.network(x)
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        logp = probs.log_prob(action)
        entropy = probs.entropy()
        return action, logp, entropy, x

    def get_value(self, x):
        x = self.network(x)
        return self.critic(x)

class Student(nn.Module):
    def __init__(self, envs, frames=4):
        super(Student, self).__init__()
        self.network = nn.Sequential(
            Scale(1/255),
            layer_init(nn.Conv2d(frames, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(3136, 1024)),
            nn.ReLU(),
            layer_init(nn.Linear(1024, 512)),
            nn.ReLU()
        )
        # self.actor = layer_init(nn.Linear(512, envs.action_space.n), std=0.01)
        self.fc = layer_init(nn.Linear(512, 512))
        self.actor = layer_init(nn.Linear(512, 18), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)
        self.n = envs.action_space.n

    def get_action(self, x, action=None):
        x = self.network(x)
        hidden = self.fc(x)
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        logp = probs.log_prob(action)
        entropy = probs.entropy()
        return action, logp, entropy, hidden

    def get_value(self, x):
        x = self.network(x)
        return self.critic(x)

teacher = [Teacher(env).to(device) for env in source_envs]
source_model = [torch.load(source_path) for source_path in source_paths]
for i in range(args.num_envs):
    teacher[i].load_state_dict(source_model[i])
    teacher[i].eval()

student = Student(envs).to(device)
optimizer = optim.Adam(student.parameters(), lr=args.learning_rate, eps=1e-5)
loss_fn = torch.nn.MSELoss()
if args.anneal_lr:
    lr = lambda f: f * args.learning_rate

global_step = 0
ckpt = args.ckpt
if ckpt:
    ckpt_load_path = f"results/{game_name}_policy_{ckpt}/checkpoints/best_ckpt.pkl"
    with open(ckpt_load_path, 'rb') as f:
        checkpoint = pickle.load(f)
    print(f"load checkpoint path: {ckpt_load_path}")
    # checkpoint = torch.load(ckpt_load_path)
    student.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    global_step = checkpoint['global_step']
    print(f"load checkpoint done")

args.num_envs = len(envs_name)
obs = torch.zeros((args.num_steps, args.num_envs) + envs.observation_space.shape).to(device)
actions = torch.zeros((args.num_steps, args.num_envs)).to(device)
teacher_actions = torch.zeros((args.num_steps, args.num_envs)).to(device)
teacher_logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)

next_obs = envs.reset()
next_done = torch.zeros(args.num_envs).to(device)
num_updates = args.total_timesteps // args.batch_size
last_reward = [float("-inf")] * args.num_envs
training_step = 0
for update in range(1, num_updates+1):
    # Annealing the rate if instructed to do so.
    if args.anneal_lr:
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = lr(frac)
        optimizer.param_groups[0]['lr'] = lrnow

    for step in range(0, args.num_steps):
        global_step += 1 * args.num_envs
        obs[step] = next_obs
        with torch.no_grad():
            action, _, _, _ = student.get_action(obs[step])
            teacher_action = [teacher[i].get_action(obs[step][i].unsqueeze(dim=0))[0] for i in range(args.num_envs)]
            teacher_logprob = [teacher[i].get_action(obs[step][i].unsqueeze(dim=0))[1] for i in range(args.num_envs)]
        actions[step] = action
        teacher_actions[step] = torch.cat(teacher_action)
        teacher_logprobs[step] = torch.cat(teacher_logprob)
        next_obs, rs, ds, infos = envs.step(action)

        for i, info in enumerate(infos):
            if 'episode' in info.keys():
                game_name = envs_name[i]
                episode_reward = info['episode']['r']
                print(f"game: {game_name}, global_step: {global_step}, episode_reward: {episode_reward}")
                if episode_reward > last_reward[i]:
                    checkpoint = {
                            "net": student.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "global_step": global_step
                            }
                    with open(ckpt_save_path + f"/best_ckpt_{game_name}.pkl", 'wb') as f:
                        pickle.dump(checkpoint, f)
                    f.close()
                    torch.save(checkpoint['net'], ckpt_save_path + f"/best_model_{game_name}.pt")
                    print(f"save best checkpoint!")
                    print(f'game: {game_name}, last_reward: {last_reward[i]}, episode_reward: {episode_reward}')
                    last_reward[i] = episode_reward
                writer.add_scalar(f"charts/episode_reward/{game_name}", episode_reward, global_step)
                # break

    b_obs = obs.reshape((-1,)+envs.observation_space.shape)
    b_actions = actions.reshape((-1,))
    b_teacher_actions = teacher_actions.reshape((-1,))
    b_teacher_logprobs = teacher_logprobs.reshape(-1)

    inds = np.arange(args.batch_size,)
    for i_epoch_pi in range(args.update_epochs):
        np.random.shuffle(inds)
        training_step += 1
        for start in range(0, args.batch_size, args.minibatch_size):
            end = start + args.minibatch_size
            minibatch_ind = inds[start:end]
            _, studentlogproba, entropy, hidden_student = student.get_action(b_obs[minibatch_ind], b_actions.long()[minibatch_ind])

            # Policy loss
            # entropy_loss = entropy.mean()
            pg_loss = torch.exp(b_teacher_logprobs[minibatch_ind]) * studentlogproba
            pg_loss = pg_loss.mean()
            # feature loss
            # feature_loss = loss_fn(hidden_student, hidden_teacher)

            loss = pg_loss # - args.ent_coef * entropy_loss + args.fea_coef * feature_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)
            optimizer.step()

        if training_step % 2000 == 0:
            checkpoint = {
                    "net": student.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "global_step": global_step
                    }
            with open(ckpt_save_path + f"/ckpt_{global_step}.pkl", 'wb') as f:
                pickle.dump(checkpoint, f)
            f.close()
            torch.save(checkpoint['net'], ckpt_save_path + f"/model_{global_step}.pt")
            print(f"save checkpoint at {global_step}!")

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]['lr'], global_step)
    # writer.add_scalar("losses/feature_loss", feature_loss.item(), global_step)
    writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
    # writer.add_scalar("losses/entropy_loss", entropy.mean().item(), global_step)

checkpoint = {
        "net": student.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": global_step
        }
with open(ckpt_save_path + f"/ckpt_{global_step}.pkl", 'wb') as f:
    pickle.dump(checkpoint, f)
torch.save(checkpoint['net'], ckpt_save_path + f"/model_{global_step}.pt")
envs.close()
writer.close()