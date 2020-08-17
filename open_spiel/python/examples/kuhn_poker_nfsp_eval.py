# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NFSP agents trained on Kuhn Poker."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
import tensorflow.compat.v1 as tf

from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import nfsp
import glob
import os
import numpy as np
from open_spiel.python.algorithms import random_agent


FLAGS = flags.FLAGS

flags.DEFINE_integer("num_train_episodes", int(3e6),
                     "Number of training episodes.")
flags.DEFINE_integer("eval_every", 100,
                     "Episode frequency at which the agents are evaluated.")
flags.DEFINE_list("hidden_layers_sizes", [
    128,
], "Number of hidden units in the avg-net and Q-net.")
flags.DEFINE_integer("replay_buffer_capacity", int(2e5),
                     "Size of the replay buffer.")
flags.DEFINE_integer("reservoir_buffer_capacity", int(2e6),
                     "Size of the reservoir buffer.")
flags.DEFINE_float("anticipatory_param_agent0", 0,
                   "Prob of using the rl best response as episode policy for agent0.")
flags.DEFINE_float("anticipatory_param_agent1", 0,
                   "Prob of using the rl best response as episode policy for agent1.")
flags.DEFINE_string("experiment_name", "kuhn_poker_0.1_7_27", "Experiment name")
flags.DEFINE_string("load_path", "/home/jxu8/Code_update/open_spiel/sessions_nfsp/", "Path to load the session")

flags.DEFINE_string("save_path", "/home/jxu8/Code_update/open_spiel/evaluation_data/eval_kp_nfsp_0.1_7_27/", "Path to load the session")


class NFSPPolicies(policy.Policy):
  """Joint policy to be evaluated."""

  def __init__(self, env, nfsp_policies, mode):
    game = env.game
    player_ids = [0, 1]
    super(NFSPPolicies, self).__init__(game, player_ids)
    self._policies = nfsp_policies
    self._mode = mode
    self._obs = {"info_state": [None, None], "legal_actions": [None, None]}

  def action_probabilities(self, state, player_id=None):
    cur_player = state.current_player()
    legal_actions = state.legal_actions(cur_player)

    self._obs["current_player"] = cur_player
    self._obs["info_state"][cur_player] = (
        state.information_state_tensor(cur_player))
    self._obs["legal_actions"][cur_player] = legal_actions

    info_state = rl_environment.TimeStep(
        observations=self._obs, rewards=None, discounts=None, step_type=None)

    with self._policies[cur_player].temp_mode_as(self._mode):
      p = self._policies[cur_player].step(info_state, is_evaluation=True).probs
    prob_dict = {action: p[action] for action in legal_actions}
    return prob_dict

def eval_against_random_agent1(env, trained_agents, random_agents, num_episodes):
  """Evaluates `trained_agents` against `random_agents` for `num_episodes`."""
  wins = np.zeros(3)
  cur_agents = [trained_agents[0], random_agents[1]]
  for _ in range(num_episodes):
      time_step = env.reset()
      while not time_step.last():
        player_id = time_step.observations["current_player"]
        agent_output = cur_agents[player_id].step(time_step, is_evaluation=True)
        time_step = env.step([agent_output.action])
      if time_step.rewards[0] > 0:
        wins[0] += 1
      elif time_step.rewards[1] > 0:
        wins[1] += 1
      else:
        wins[2] += 1
  return wins / num_episodes

def eval_against_random_agent0(env, trained_agents, random_agents, num_episodes):
  """Evaluates `trained_agents` against `random_agents` for `num_episodes`."""
  wins = np.zeros(3)
  cur_agents = [random_agents[0], trained_agents[1]]
  for _ in range(num_episodes):
      time_step = env.reset()
      while not time_step.last():
        player_id = time_step.observations["current_player"]
        agent_output = cur_agents[player_id].step(time_step, is_evaluation=True)
        time_step = env.step([agent_output.action])
      if time_step.rewards[0] > 0:
        wins[0] += 1
      elif time_step.rewards[1] > 0:
        wins[1] += 1
      else:
        wins[2] += 1
  return wins / num_episodes

def eval_against_trained_agents(env, trained_agents, num_episodes):
  wins = np.zeros(3)
  for _ in range(num_episodes):
    time_step = env.reset()
    while not time_step.last():
      player_id = time_step.observations["current_player"]
      if env.is_turn_based:
        agents_output = trained_agents[player_id].step(time_step, is_evaluation=True)
        action_list = [agents_output.action]
      else:
        agents_output = [agent.step(time_step, is_evaluation=True) for agent in trained_agents]
        action_list = [agent_output.action for agent_output in agents_output]
      time_step = env.step(action_list)
    if time_step.rewards[0] > 0:
      wins[0] += 1
    elif time_step.rewards[1] > 0:
      wins[1] += 1
    else:
      wins[2] += 1
  return wins / num_episodes

def eval_between_random_agents(env, random_agents, num_episodes):
    wins = np.zeros(3)
    rewards = np.zeros(2)
    for _ in range(num_episodes):
        time_step = env.reset()
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            if env.is_turn_based:
                agents_output = random_agents[player_id].step(time_step, is_evaluation=True)
                action_list = [agents_output.action]
            else:
                agents_output = [agent.step(time_step, is_evaluation=True) for agent in random_agents]
                action_list = [agent_output.action for agent_output in agents_output]
            time_step = env.step(action_list)
        rewards[0] += time_step.rewards[0]
        rewards[1] += time_step.rewards[1]
        if time_step.rewards[0] > 0:
            wins[0] += 1
        elif time_step.rewards[1] > 0:
            wins[1] += 1
        else:
            wins[2] += 1
    return wins / num_episodes, rewards / num_episodes

def main(unused_argv):
  game = "kuhn_poker"
  num_players = 2
  load_path = FLAGS.load_path + FLAGS.experiment_name
  env_configs = {"players": num_players}
  env = rl_environment.Environment(game, **env_configs)
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]

  hidden_layers_sizes = [int(l) for l in FLAGS.hidden_layers_sizes]
  kwargs = {
      "replay_buffer_capacity": FLAGS.replay_buffer_capacity,
      "epsilon_decay_duration": FLAGS.num_train_episodes,
      "epsilon_start": 0.06,
      "epsilon_end": 0.001,
  }

  random_agents = [
      random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
      for idx in range(num_players)
  ]

  model_dirs = sorted(glob.glob(load_path + "/episode-*"), key=lambda x: int(os.path.split(x)[1][8:]))
  for counter, dir in enumerate(model_dirs, 1):
      if counter % 1 == 0:
          tf.reset_default_graph()
          with tf.Session() as sess:
            # pylint: disable=g-complex-comprehension
            agent0 = nfsp.NFSP(sess, 0, info_state_size, num_actions, hidden_layers_sizes,
                          FLAGS.reservoir_buffer_capacity, FLAGS.anticipatory_param_agent0,
                          **kwargs)
            agent1 = nfsp.NFSP(sess, 1, info_state_size, num_actions, hidden_layers_sizes,
                          FLAGS.reservoir_buffer_capacity, FLAGS.anticipatory_param_agent1,
                          **kwargs)

            saver = tf.train.Saver()
            saver.restore(sess, dir + "/trained_model-10000")
            #expl_policies_avg = NFSPPolicies(env, [agent0, agent1], nfsp.MODE.average_policy)
            #expl_list, expl = exploitability.exploitability(env.game, expl_policies_avg)
            # f3 = open(FLAGS.save_path + "exploitability_list.txt", "a")
            # f3.write(str(expl_list[0]) + ' ' + str(expl_list[1]) + '\n')
            # f4 = open(FLAGS.save_path + "exploitability_avg.txt", "a")
            # f4.write(str(expl) + '\n')

            # logging.info("Episode: %s, Exploitability AVG %s", counter*10000, expl)
            # logging.info("_____________________________________________")

            # win_rates_against_random_agent1 = eval_against_random_agent1(env, [agent0, agent1], random_agents, 1000)
            # win_rates_against_random_agent0 = eval_against_random_agent0(env, [agent0, agent1], random_agents, 1000)
            # win_rates_against_trained_agents = eval_against_trained_agents(env, [agent0, agent1], 1000)
            win_rates_between_random_agents, avg_utility_between_random_agents = eval_between_random_agents(env, random_agents, 1000)

            f1 = open(FLAGS.save_path + "win_rates/eta_0/win_rates_between_random_agents.txt", "a")
            f1.write(str(win_rates_between_random_agents[0]) + ' ' + str(win_rates_between_random_agents[1]) + ' ' + str(win_rates_between_random_agents[2]) + '\n')
            f2 = open(FLAGS.save_path + "avg_utility/eta_0/avg_utility_between_random_agents.txt", "a")
            f2.write(str(avg_utility_between_random_agents[0]) + ' ' + str(avg_utility_between_random_agents[1]) + '\n')

            logging.info("Episode: %s", counter*10000)
            logging.info("Trained_agent0 vs Random_agent1: %s", avg_utility_between_random_agents)


            # f1 = open(FLAGS.save_path + "win_rates_against_random_agent1.txt", "a")
            # f1.write(str(win_rates_against_random_agent1[0]) + ' ' + str(win_rates_against_random_agent1[1]) + ' ' + str(win_rates_against_random_agent1[2]) + '\n')
            # f2 = open(FLAGS.save_path + "win_rates_against_random_agent0.txt", "a")
            # f2.write(str(win_rates_against_random_agent0[0]) + ' ' + str(win_rates_against_random_agent0[1]) + ' ' + str(win_rates_against_random_agent0[2]) + '\n')
            # f3 = open(FLAGS.save_path + "win_rates_against_eachother.txt", "a")
            # f3.write(str(win_rates_against_trained_agents[0]) + ' ' + str(win_rates_against_trained_agents[1]) + ' ' + str(win_rates_against_trained_agents[2]) + '\n')

            # logging.info("Episode: %s", counter*10000)
            # logging.info("Trained_agent0 vs Random_agent1: %s", win_rates_against_random_agent1)
            # logging.info("Random_agent0 vs Trained_agent1: %s", win_rates_against_random_agent0)
            # logging.info("Trained_agent0 vs Trained_agent1 %s", win_rates_against_trained_agents)
            # logging.info("_____________________________________________")



if __name__ == "__main__":
  app.run(main)
