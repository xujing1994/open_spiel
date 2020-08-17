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
flags.DEFINE_float("anticipatory_param", 0.1,
                   "Prob of using the rl best response as episode policy.")
flags.DEFINE_integer("save_every", 100,
                     "Episode frequency at which the agents are evaluated.")

counter = np.zeros(1)

class NFSPPolicies(policy.Policy):
  """Joint policy to be evaluated."""

  def __init__(self, env, nfsp_policies, mode):
    game = env.game
    player_ids = [0, 1]
    super(NFSPPolicies, self).__init__(game, player_ids)
    self._policies = nfsp_policies
    self._mode = mode
    self._obs = {"info_state": [None, None], "legal_actions": [None, None]}
    counter[0] += 1


  def action_probabilities(self, state, player_id=None):
    cur_player = state.current_player()
    legal_actions = state.legal_actions(cur_player)

    self._obs["current_player"] = cur_player
    self._obs["info_state"][cur_player] = (
        state.information_state_tensor(cur_player)) # tic_tac_toe: observation_tensor
    self._obs["legal_actions"][cur_player] = legal_actions

    info_state = rl_environment.TimeStep(
        observations=self._obs, rewards=None, discounts=None, step_type=None)

    with self._policies[cur_player].temp_mode_as(self._mode):
      p = self._policies[cur_player].step(info_state, is_evaluation=True).probs
    prob_dict = {action: p[action] for action in legal_actions}
    counter[0] += 1
    return prob_dict

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

def main(unused_argv):
  game = "kuhn_poker"
  num_players = 2
  path = '/home/jxu8/Code_update/open_spiel/sessions_nfsp/kuhn_poker_0.1'
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
  expl_single_agent = []
  with tf.Session() as sess:
    # pylint: disable=g-complex-comprehension
    agents = [
        nfsp.NFSP(sess, idx, info_state_size, num_actions, hidden_layers_sizes,
                  FLAGS.reservoir_buffer_capacity, FLAGS.anticipatory_param,
                  **kwargs) for idx in range(num_players)
    ]
    expl_policies_avg = NFSPPolicies(env, agents, nfsp.MODE.average_policy)
    # expl_single_agent.append(NFSPPolicy(env, agents[0], nfsp.MODE.average_policy))
    # expl_single_agent.append(NFSPPolicy(env, agents[1], nfsp.MODE.average_policy))
    saver = tf.train.Saver(max_to_keep = 2000)
    sess.run(tf.global_variables_initializer())

    for ep in range(FLAGS.num_train_episodes):
      if (ep + 1) % FLAGS.eval_every == 0:
        losses = [agent.loss for agent in agents]
        expl_list, expl = exploitability.exploitability(env.game, expl_policies_avg)
        # expl_agent0 = exploitability.exploitability(env.game, expl_single_agent[0])
        # expl_agent1 = exploitability.exploitability(env.game, expl_single_agent[1])
        win_rates_against_random_agent1 = eval_against_random_agent1(env, agents, random_agents, 1000)
        win_rates_against_random_agent0 = eval_against_random_agent0(env, agents, random_agents, 1000)
        win_rates_against_eachother = eval_against_trained_agents(env, agents, 1000)
        logging.info("Episode: %s", ep + 1)
        logging.info("Losses (SL loss, RL loss), Agent0: %s, Agent1: %s", losses[0], losses[1])
        logging.info("Exploitability AVG %s", expl)
        # logging.info("_____________________________________________")
        logging.info("Win rates: Trained_agent0 vs Random_agent1  %s", win_rates_against_random_agent1)
        logging.info("Win rates: Random_agent0 vs Trained_agent1  %s", win_rates_against_random_agent0)
        logging.info("Win rates: Trained_agent0 vs Trained_agent1 %s", win_rates_against_eachother)
        logging.info("__________________________________________________________________")

      time_step = env.reset()
      while not time_step.last():
        player_id = time_step.observations["current_player"]
        agent_output = agents[player_id].step(time_step)
        action_list = [agent_output.action]
        time_step = env.step(action_list)

      # Episode is over, step all agents with final info state.
      for agent in agents:
        agent.step(time_step)

      # if (ep + 1) % FLAGS.save_every == 0:
      #     saver.save(sess, path + "/episode-" + str(ep+1) + "/trained_model", global_step = 100)



if __name__ == "__main__":
  app.run(main)
