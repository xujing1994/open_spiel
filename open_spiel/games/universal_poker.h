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

#ifndef OPEN_SPIEL_GAMES_UNIVERSAL_POKER_H_
#define OPEN_SPIEL_GAMES_UNIVERSAL_POKER_H_

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/games/universal_poker/acpc_cpp/acpc_game.h"
#include "open_spiel/games/universal_poker/logic/card_set.h"
#include "open_spiel/policy.h"
#include "open_spiel/spiel.h"

// This is a wrapper around the Annual Computer Poker Competition bot (ACPC)
// environment. See http://www.computerpokercompetition.org/. The code is
// initially available at https://github.com/ethansbrown/acpc
// It is an optional dependency (see install.md for documentation and
// open_spiel/scripts/global_variables.sh to enable this).
//
// It has not been extensively reviewed/tested by the DeepMind OpenSpiel team.
namespace open_spiel {
namespace universal_poker {

class UniversalPokerGame;

constexpr uint8_t kMaxUniversalPokerPlayers = 10;

// This is the mapping from int to action. E.g. the legal action "0" is fold,
// the legal action "1" is check/call, etc.
enum ActionType { kFold = 0, kCall = 1, kBet = 2, kAllIn = 3, kDeal = 4 };
enum BettingAbstraction { kFCPA = 0, kFC = 1, kFULLGAME = 2 };

enum StateActionType {
  ACTION_DEAL = 1,
  ACTION_FOLD = 2,
  ACTION_CHECK_CALL = 4,
  ACTION_BET = 8,
  ACTION_ALL_IN = 16
};

constexpr StateActionType ALL_ACTIONS[5] = {
    ACTION_DEAL, ACTION_FOLD, ACTION_CHECK_CALL, ACTION_BET, ACTION_ALL_IN};

class UniversalPokerState : public State {
 public:
  explicit UniversalPokerState(std::shared_ptr<const Game> game);

  bool IsTerminal() const override;
  bool IsChanceNode() const override;
  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action move) const override;
  std::string ToString() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void InformationStateTensor(Player player,
                              absl::Span<float> values) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;

  // The probability of taking each possible action in a particular info state.
  std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
  std::vector<Action> LegalActions() const override;

  // Used to make UpdateIncrementalStateDistribution much faster.
  std::unique_ptr<HistoryDistribution> GetHistoriesConsistentWithInfostate(
      int player_id) const override;

 protected:
  void DoApplyAction(Action action_id) override;

 private:
  void _CalculateActionsAndNodeType();

  double GetTotalReward(Player player) const;

  const uint32_t &GetPossibleActionsMask() const { return possibleActions_; }
  const int GetPossibleActionCount() const;

  void ApplyChoiceAction(StateActionType action_type, int size);
  const std::string &GetActionSequence() const { return actionSequence_; }

  void AddHoleCard(uint8_t card) {
    Player p = hole_cards_dealt_ / acpc_game_->GetNbHoleCardsRequired();
    const int card_index =
        hole_cards_dealt_ % acpc_game_->GetNbHoleCardsRequired();
    acpc_state_.AddHoleCard(p, card_index, card);
    ++hole_cards_dealt_;
  }

  void AddBoardCard(uint8_t card) {
    acpc_state_.AddBoardCard(board_cards_dealt_, card);
    ++board_cards_dealt_;
  }

  logic::CardSet HoleCards(Player player) const {
    logic::CardSet hole_cards;
    const int num_players = acpc_game_->GetNbPlayers();
    const int num_cards_dealt_to_all = hole_cards_dealt_ / num_players;
    int num_cards_dealt_to_player = num_cards_dealt_to_all;
    // We deal to players in order from 0 to n - 1. So if the number of cards
    // dealt % num_players is > the player, we haven't dealt them a card yet;
    // otherwise we have.
    if (player < (hole_cards_dealt_ % num_players) &&
        num_cards_dealt_to_all < acpc_game_->GetNbHoleCardsRequired()) {
      ++num_cards_dealt_to_player;
    }
    SPIEL_CHECK_LT(player, acpc_game_->GetNbPlayers());
    SPIEL_CHECK_LE(num_cards_dealt_to_player,
                   static_cast<int>(acpc_game_->GetNbHoleCardsRequired()));
    for (int i = 0; i < num_cards_dealt_to_player; ++i) {
      hole_cards.AddCard(acpc_state_.hole_cards(player, i));
    }
    return hole_cards;
  }

  logic::CardSet BoardCards() const {
    logic::CardSet board_cards;
    const int num_board_cards =
        std::min(board_cards_dealt_,
                 static_cast<int>(acpc_game_->GetTotalNbBoardCards()));
    for (int i = 0; i < num_board_cards; ++i) {
      board_cards.AddCard(acpc_state_.board_cards(i));
    }
    return board_cards;
  }

  const acpc_cpp::ACPCGame *acpc_game_;
  mutable acpc_cpp::ACPCState acpc_state_;
  logic::CardSet deck_;  // The remaining cards to deal.
  int hole_cards_dealt_ = 0;
  int board_cards_dealt_ = 0;

  // The current player:
  // kChancePlayerId for chance nodes
  // kTerminalPlayerId when we everyone except one player has fold, or that
  // we have reached the showdown.
  // The current player >= 0 otherwise.
  Player cur_player_;
  uint32_t possibleActions_;
  int32_t potSize_ = 0;
  int32_t allInSize_ = 0;
  std::string actionSequence_;

  BettingAbstraction betting_abstraction_;
};

class UniversalPokerGame : public Game {
 public:
  explicit UniversalPokerGame(const GameParameters &params);

  int NumDistinctActions() const override;
  std::unique_ptr<State> NewInitialState() const override;
  int NumPlayers() const override;
  double MinUtility() const override;
  double MaxUtility() const override;
  int MaxChanceOutcomes() const override;
  double UtilitySum() const override { return 0; }
  std::shared_ptr<const Game> Clone() const override;
  std::vector<int> InformationStateTensorShape() const override;
  std::vector<int> ObservationTensorShape() const override;
  int MaxGameLength() const override;
  BettingAbstraction betting_abstraction() const {
    return betting_abstraction_;
  }

  int big_blind() const { return big_blind_; }
  int starting_stack_big_blinds() const { return starting_stack_big_blinds_; }

 private:
  std::string gameDesc_;
  const acpc_cpp::ACPCGame acpc_game_;
  absl::optional<int> max_game_length_;
  BettingAbstraction betting_abstraction_ = BettingAbstraction::kFULLGAME;

 public:
  const acpc_cpp::ACPCGame *GetACPCGame() const { return &acpc_game_; }

  std::string parseParameters(const GameParameters &map);
  int big_blind_;
  int starting_stack_big_blinds_;
};

// Only supported for UniversalPoker. Randomly plays an action from a fixed list
// of actions. If none of the actions are legal, selects uniformly from the
// list of legal actions.
class UniformRestrictedActions : public Policy {
 public:
  // Actions will be restricted to this list when legal. If no such action is
  // legal, uniform random over all legal actions will be returned.
  explicit UniformRestrictedActions(std::vector<ActionType> actions)
      : actions_(std::move(actions)) {}

  ActionsAndProbs GetStatePolicy(const State &state) const {
    ActionsAndProbs policy;
    std::vector<Action> legal_actions = state.LegalActions();
    for (Action action : legal_actions) {
      if (absl::c_find(actions_, action) != actions_.end()) {
        policy.emplace_back(action, 1.);
      }
    }

    // If we have a non-empty policy, normalize it!
    if (!policy.empty()) {
      const double size = static_cast<double>(policy.size());
      absl::c_for_each(policy, [size](std::pair<Action, double> &a_and_p) {
        a_and_p.second /= size;
      });
      return policy;
    }

    // Otherwise, we return uniform random.
    policy.reserve(legal_actions.size());
    absl::c_for_each(legal_actions, [&policy, &legal_actions](Action a) {
      policy.push_back({a, 1. / static_cast<double>(legal_actions.size())});
    });
    SPIEL_CHECK_EQ(policy.size(), legal_actions.size());
    return policy;
  }

 private:
  std::vector<ActionType> actions_;
};

std::ostream &operator<<(std::ostream &os, const BettingAbstraction &betting);
}  // namespace universal_poker
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_UNIVERSAL_POKER_H_
