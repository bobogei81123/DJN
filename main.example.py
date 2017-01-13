from dqn import DQN
from envs.eat_bullet.eat_bullet import EatBulletEnv

if __name__ == '__main__':

    raise Exception('Please copy main.example.py to main.py and don\'t change this file')

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



