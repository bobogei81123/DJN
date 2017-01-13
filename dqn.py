from enum import Enum
from collections import namedtuple as NamedTuple
import numpy as np
from nn import NN
from training_data import TrainingData
from os import path
from typing import Any, Tuple, List
from termcolor import colored
import tensorflow as tf
from datetime import datetime

Action = int
Reward = float
Observation = np.ndarray

# Experience := (obs_now, action, reward, obs_next)
Experience = NamedTuple('Experience', ['obs', 'action', 'reward', 'obs_next'])

def make_simple_summary(tag, value):
    return tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])

class DQN:

    def __init__(self,
        # About enviroment
                 env,
                 *,
                 max_step: int,
        # About RL
                 gamma: float,
                 epsilon: float=1.0,
                 epsilon_min: float=0.05,
                 epsilon_decay: float=0.05,
                 update_episode_n: int,
        # About history
                 history_max_n: int,
        # About NN
                 nnet_epoch: int,
                 eta: float,
                 eta_decay: float = 1.0,
                 batch_size: int = 20,
        # About Validation
                 episode_per_validation: int=5,
        # Misc
                 save_dir: str='hao123'
                 ) -> None:
        '''
        env: The enviroment to play
        max_step:

        gamma:
        epsilon:
        epsilon_min:
        epsilon_decay:

        update_episode_n: Update NN per episode
        history_max_n: Maximum size of history
        nnet_epoch: Epoch to run each NN update
        eta: Learning rate
        eta_decay: Learning rate decay per episode
        batch_size:
        save_dir: Dir name to save model (save at {save_dir}/datestring)
        '''

        self.env = env
        self.save_dir = path.join(save_dir, datetime.now().strftime('%Y%m%d%H%M%S'))
        self.model_save_dir = path.join(self.save_dir, 'model')
        self.summary_save_dir = path.join(self.save_dir, 'summary')

        self.obs_shape = env.observation_space.shape
        self.action_n = env.action_space.n
        self.max_step = max_step

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.update_episode_n = update_episode_n

        self.history_max_n = history_max_n

        self.nnet_epoch = nnet_epoch
        self.eta = eta
        self.eta_decay = eta_decay
        self.batch_size = batch_size

        self.episode_per_validation = episode_per_validation

        self.best_reward = -1e9

        self.history = [] # type: List[Experience]

        self.nn = NN(self.obs_shape, self.action_n, batch_size)

        self.summary = tf.summary.FileWriter(self.summary_save_dir)
        self.global_step = 0

    def get_action(self,
                   obs,
                   use_epsilon: bool=True) -> Action:

        if use_epsilon:
            return self._epsilon_greedy(obs, self.epsilon)
        else:
            return self._epsilon_greedy(obs, self.epsilon_min)

    def _epsilon_greedy(self,
                        obs,
                        eps: float) -> Action:
        r = np.random.random()

        if r < eps:
            return np.random.randint(self.action_n)
        else:
            return self._best_action(obs)

    def _best_action(self,
                     obs) -> Action:
        act_value, _ = self.nn.feed(obs)
        return act_value

    def add_summary(self, tag, value, global_step=None):
        if global_step is None: global_step = self.global_step
        self.summary.add_summary(
            make_simple_summary(tag, value),
            global_step=global_step,
        )


    def start_training(self,
                       reward_target=1e9) -> None:
        iter_n = 0
        while True:
            #Training
            iter_n += 1
            print(colored(f'[Iter #{iter_n}]', 'green'))
            print(colored('[Generating Episode]', 'cyan'), f'ε = {self.epsilon:.4f}')

            reward_avg = 0.
            for _ in range(self.update_episode_n):
                reward, history = self.start_episode(True)
                reward_avg += reward
                self.history.extend(history)

            reward_avg /= self.update_episode_n
            self.add_summary('reward', reward_avg)
            self.add_summary('eta', self.eta)
            self.add_summary('epsilon', self.epsilon)
            print(f'Played {self.update_episode_n} episodes,',
                  colored(f'R_avg = {reward_avg:.4f}', 'red', attrs=['bold']))
            self.update()

            # Testing
            if iter_n % self.episode_per_validation == 0:
                print(colored(f'[Validation #{iter_n}]', 'blue', attrs=['bold']))

                reward_avg = 0.
                test_n = 10
                for i in range(test_n):
                    reward_avg += self.start_episode(False)[0]
                reward_avg /= test_n
                self.add_summary('reward_val', reward_avg)
                print(colored('[Validation result]', 'blue', attrs=['bold']),
                      colored(f'R_avg = {reward_avg:.4f}', 'red', attrs=['bold']))

                self.best_reward = max(self.best_reward, reward_avg)

                # Return if the target is reached
                if reward_avg >= reward_target:
                    return

            self.summary.flush()

            self.epsilon = max(self.epsilon_min,
                               self.epsilon - self.epsilon_decay)
            self.eta *= self.eta_decay


    def start_episode(self, use_epsilon: bool=False) -> Tuple[Reward, List[Experience]]:
        history = []
        obs = self.env.reset()

        reward_tot = 0.

        for step in range(self.max_step):
            # Small hack to decide if training or not
            if use_epsilon:
                self.global_step += 1

            action = self.get_action(obs, use_epsilon)
            obs_next, reward, done, info = self.env.step(action)
            history.append(Experience(obs, action, reward, None if done else obs_next))
            reward_tot += reward
            if done:
                break
            obs = obs_next

        return (reward_tot, history)

    def update(self) -> None:
        history_n = len(self.history)

        # If history_n > history_max_n, delete the oldest
        if history_n > self.history_max_n:
            d = history_n - self.history_max_n
            self.history = self.history[d:]

        obs, action, reward, obs_next = zip(*self.history)
        obs_arr = np.array(obs)
        action_arr = np.array(action)

        # Calculate Q(s) ≈ max Q(s,a | theta') using obs_next
        Qsa = np.array([
            o if o is not None else np.zeros(self.obs_shape)
            for o in obs_next
        ])
        Q_next = np.max(self.nn.feed(Qsa)[1], axis=1)

        for i, x in enumerate(obs_next):
            if x is None:
                Q_next[i] = 0.

        target_arr = np.array(reward) + self.gamma*Q_next

        n = len(action)

        data = TrainingData(obs_arr, action_arr, target_arr, self.batch_size)
        # real_epoch = int(self.nnet_epoch * self.history_max_n / len(inp_arr))
        self.nn.train(data, self.nnet_epoch, self.eta)


if __name__ == '__main__':
    from envs.eat_bullet.eat_bullet import EatBulletEnv

    env = EatBulletEnv(grid_size=(10, 10), food_n=10)
    dqn = DQN(env=env,
              max_step=200,

              gamma=0.97,
              epsilon=1.0,
              # epsilon_min,
              epsilon_decay=0.02,
              update_episode_n=20,

              history_max_n=50000,
              nnet_epoch=5,
              eta=2e-4,
              eta_decay=0.999,
              batch_size=20,
              save_dir='eatbullet'
              )

    dqn.start_training()



