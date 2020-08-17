// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef OPEN_SPIEL_POLICY_H_
#define OPEN_SPIEL_POLICY_H_

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {

// Returns the probability for the specified action, or -1 if not found.
double GetProb(const ActionsAndProbs& action_and_probs, Action action);

// Set an action probability for the specified action.
void SetProb(ActionsAndProbs* actions_and_probs, Action action, double prob);

// Helper for deterministic policies: returns the single action if the policy
// is deterministic, otherwise returns kInvalidAction.
Action GetAction(const ActionsAndProbs& action_and_probs);

// Returns a policy where every legal action has probability 1 / (number of
// legal actions).
ActionsAndProbs UniformStatePolicy(const State& state);

// Return a new policy with all the same actions, but with probability 1 on the
// specified action, and 0 on the others.
ActionsAndProbs ToDeterministicPolicy(const ActionsAndProbs& actions_and_probs,
                                      Action action);

// A general policy object. A policy is a mapping from states to list of
// (action, prob) pairs for all the legal actions at the state.
class Policy {
 public:
  virtual ~Policy() = default;

  // A convenience method for callers that want to use arrays.
  virtual std::pair<std::vector<Action>, std::vector<double>>
  GetStatePolicyAsParallelVectors(const State& state) const {
    std::pair<std::vector<Action>, std::vector<double>> parray;
    for (const auto& action_and_prob : GetStatePolicy(state)) {
      parray.first.push_back(action_and_prob.first);
      parray.second.push_back(action_and_prob.second);
    }
    return parray;
  }

  // A convenience method for callers that want to use arrays.
  virtual std::pair<std::vector<Action>, std::vector<double>>
  GetStatePolicyAsParallelVectors(const std::string info_state) const {
    std::pair<std::vector<Action>, std::vector<double>> parray;
    for (const auto& action_and_prob : GetStatePolicy(info_state)) {
      parray.first.push_back(action_and_prob.first);
      parray.second.push_back(action_and_prob.second);
    }
    return parray;
  }

  virtual std::unordered_map<Action, double> GetStatePolicyAsMap(
      const State& state) const {
    std::unordered_map<Action, double> pmap;
    for (const auto& action_and_prob : GetStatePolicy(state)) {
      pmap[action_and_prob.first] = action_and_prob.second;
    }
    return pmap;
  }

  virtual std::unordered_map<Action, double> GetStatePolicyAsMap(
      const std::string& info_state) const {
    std::unordered_map<Action, double> pmap;
    for (const auto& action_and_prob : GetStatePolicy(info_state)) {
      pmap[action_and_prob.first] = action_and_prob.second;
    }
    return pmap;
  }

  // Returns a list of (action, prob) pairs for the policy at this state.
  // If the policy is not available at the state, returns and empty list.
  virtual ActionsAndProbs GetStatePolicy(const State& state) const {
    return GetStatePolicy(state.InformationStateString());
  }

  // Returns a list of (action, prob) pairs for the policy at this info state.
  // If the policy is not available at the state, returns and empty list.
  // It is sufficient for subclasses to override only this method, but not all
  // forms of policies will be able to do so from just the information state.
  virtual ActionsAndProbs GetStatePolicy(const std::string& info_state) const {
    SpielFatalError("GetStatePolicy(const std::string&) unimplemented.");
  }
};

// A tabular policy represented internally as a map. Note that this
// implementation is not directly compatible with the Python TabularPolicy
// implementation; the latter is implemented as a table of size
// [num_states, num_actions], while this is implemented as a map. It is
// non-trivial to convert between the two, but we have a function that does so
// in the open_spiel/python/policy.py file.
class TabularPolicy : public Policy {
 public:
  TabularPolicy() = default;
  TabularPolicy(const Game& game);  // Construct a uniform random policy.
  TabularPolicy(const TabularPolicy& other) = default;
  TabularPolicy(const std::unordered_map<std::string, ActionsAndProbs>& table)
      : policy_table_(table) {}

  // Converts a policy to a TabularPolicy.
  TabularPolicy(const Game& game, const Policy& policy) : TabularPolicy(game) {
    for (auto& [infostate, is_policy] : policy_table_) {
      is_policy = policy.GetStatePolicy(infostate);
    }
  }

  // Creates a new TabularPolicy from a deterministic policy encoded as a
  // {info_state_str -> action} dict. The dummy_policy is used to initialize
  // the initial mapping.
  TabularPolicy(const TabularPolicy& dummy_policy,
                const std::unordered_map<std::string, Action>& action_map)
      : policy_table_(dummy_policy.policy_table_) {
    for (const auto& entry : action_map) {
      std::string info_state = entry.first;
      Action action_taken = action_map.at(entry.first);
      for (auto& action_and_prob : policy_table_[info_state]) {
        action_and_prob.second =
            (action_and_prob.first == action_taken ? 1.0 : 0.0);
      }
    }
  }

  ActionsAndProbs GetStatePolicy(const std::string& info_state) const override {
    auto iter = policy_table_.find(info_state);
    if (iter == policy_table_.end()) {
      return {};
    } else {
      return iter->second;
    }
  }

  // Set the probability for action at the info state. If the info state is not
  // in the policy, it is added. If the action is not in the info state policy,
  // it is added. Otherwise it is modified.
  void SetProb(const std::string& info_state, Action action, double prob) {
    auto iter = policy_table_.find(info_state);
    if (iter == policy_table_.end()) {
      auto iter_and_bool = policy_table_.insert({info_state, {}});
      iter = iter_and_bool.first;
    }
    open_spiel::SetProb(&(iter->second), action, prob);
  }

  void SetStatePolicy(const std::string& info_state,
                      const ActionsAndProbs& state_policy) {
    policy_table_[info_state] = state_policy;
  }

  std::unordered_map<std::string, ActionsAndProbs>& PolicyTable() {
    return policy_table_;
  }

  const std::unordered_map<std::string, ActionsAndProbs>& PolicyTable() const {
    return policy_table_;
  }

  const std::string ToString() const;

  // A ToString where the keys are sorted.
  const std::string ToStringSorted() const;

 private:
  std::unordered_map<std::string, ActionsAndProbs> policy_table_;
};

// Chooses all legal actions with equal probability. This is equivalent to the
// tabular version, except that this works for large games.
class UniformPolicy : public Policy {
 public:
  ActionsAndProbs GetStatePolicy(const State& state) const override {
    return UniformStatePolicy(state);
  }
};

// Helper functions that generate policies for testing.
TabularPolicy GetEmptyTabularPolicy(const Game& game,
                                    bool initialize_to_uniform = false);
TabularPolicy GetUniformPolicy(const Game& game);
TabularPolicy GetRandomPolicy(const Game& game, int seed = 0);
TabularPolicy GetFirstActionPolicy(const Game& game);

std::string PrintPolicy(const ActionsAndProbs& policy);

}  // namespace open_spiel

#endif  // OPEN_SPIEL_POLICY_H_
