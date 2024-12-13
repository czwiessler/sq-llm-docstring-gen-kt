from __future__ import print_function
import argparse
import logging
import os

# Iroko imports
import dc_gym
import dc_gym.utils as dc_utils

# configure logging
log = logging.getLogger(__name__)


# set up paths
cwd = os.getcwd()
lib_dir = os.path.dirname(dc_gym.__file__)
INPUT_DIR = lib_dir + '/inputs'
OUTPUT_DIR = cwd + '/results'

PARSER = argparse.ArgumentParser()
PARSER.add_argument('--env', '-e', dest='env',
                    default='iroko', help='The platform to run.')
PARSER.add_argument('--topo', '-to', dest='topo',
                    default='dumbbell', help='The topology to operate on.')
PARSER.add_argument('--episodes', '-t', dest='episodes',
                    type=int, default=2,
                    help='total number of episodes to train rl agent')
PARSER.add_argument('--output', dest='output_dir', default=OUTPUT_DIR,
                    help='Folder which contains all the collected metrics.')
PARSER.add_argument('--transport', dest='transport', default="udp",
                    help='Choose the transport protocol of the hosts.')
PARSER.add_argument('--agent', '-a', dest='agent', default="PG",
                    help='must be string of either: PPO, DDPG, PG,'
                         ' DCTCP, TCP_NV, PCC, or TCP', type=str.lower)
ARGS = PARSER.parse_args()


def test_run(input_dir, output_dir, env, topo):
    # Assemble a configuration dictionary for the environment
    env_config = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "env": env,
        "topo": topo,
        "agent": ARGS.agent,
        "transport": ARGS.transport,
        "episodes": ARGS.episodes,
        "tf_index": 0
    }
    dc_env = dc_utils.EnvFactory.create(env_config)
    for episodes in range(ARGS.episodes):
        dc_env.reset()
        done = False
        while not done:
            action = dc_env.action_space.sample()
            obs, reward, done, info = dc_env.step(action)

    log.info("Generator Finished. Simulation over. Clearing dc_env...")
    dc_env.close()


def clean():
    log.info("Removing all traces of Mininet")
    os.system('sudo mn -c')
    os.system("sudo killall -9 goben")
    os.system("sudo killall -9 node_control")


def init():
    output_dir = ARGS.output_dir + "/" + ARGS.agent
    dc_utils.check_dir(output_dir)
    test_run(INPUT_DIR, output_dir, ARGS.env, ARGS.topo)


if __name__ == '__main__':
    logging.basicConfig(format="%(levelname)s:%(message)s",
                        level=logging.INFO)
    init()
