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

#include "open_spiel/games/universal_poker.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <utility>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_format.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/games/universal_poker/logic/card_set.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace universal_poker {
namespace {

std::string BettingAbstractionToString(const BettingAbstraction &betting) {
  switch (betting) {
    case BettingAbstraction::kFC: {
      return "BettingAbstration: FC";
      break;
    }
    case BettingAbstraction::kFCPA: {
      return "BettingAbstration: FCPA";
      break;
    }
    case BettingAbstraction::kFULLGAME: {
      return "BettingAbstraction: FULLGAME";
      break;
    }
    default:
      SpielFatalError("Unknown betting abstraction.");
      break;
  }
}

}  // namespace

const GameType kGameType{
    /*short_name=*/"universal_poker",
    /*long_name=*/"Universal Poker",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kExplicitStochastic,
    GameType::Information::kImperfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/10,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/true,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/

    {// The ACPC code uses a specific configuration file to describe the game.
     // The following has been copied from ACPC documentation:
     //
     // Empty lines or lines with '#' as the very first character will be
     // ignored
     //
     // The Game definitions should start with "gamedef" and end with
     // "end gamedef" and can have the fields documented bellow (case is
     // ignored)
     //
     // If you are creating your own game definitions, please note that game.h
     // defines some constants for maximums in games (e.g., number of rounds).
     // These may need to be changed for games outside of the what is being run
     // for the Annual Computer Poker Competition.

     // The ACPC gamedef string.  When present, it will take precedence over
     // everything and no other argument should be provided.
     {"gamedef", GameParameter(std::string(""))},
     // Instead of a single gamedef, specifying each line is also possible.
     // The documentation is adapted from project_acpc_server/game.cc.
     //
     // Number of Players (up to 10)
     {"numPlayers", GameParameter(2)},
     // Betting Type "limit" "nolimit"
     {"betting", GameParameter(std::string("nolimit"))},
     // The stack size for each player at the start of each hand (for
     // no-limit). It will be ignored on "limit".
     // TODO(author2): It's unclear what happens on limit. It defaults to
     // INT32_MAX for all players when not provided.
     {"stack", GameParameter(std::string("1200 1200"))},
     // The size of the blinds for each player (relative to the dealer)
     {"blind", GameParameter(std::string("100 100"))},
     // The size of raises on each round (for limit games only) as numrounds
     // integers. It will be ignored for nolimite games.
     {"raiseSize", GameParameter(std::string("100 100"))},
     // Number of betting rounds per hand of the game
     {"numRounds", GameParameter(2)},
     // The player that acts first (relative to the dealer) on each round
     {"firstPlayer", GameParameter(std::string("1 1"))},
     // maxraises - the maximum number of raises on each round. If not
     // specified, it will default to UINT8_MAX.
     {"maxRaises", GameParameter(std::string(""))},
     // The number of different suits in the deck
     {"numSuits", GameParameter(4)},
     // The number of different ranks in the deck
     {"numRanks", GameParameter(6)},
     // The number of private cards to  deal to each player
     {"numHoleCards", GameParameter(1)},
     // The number of cards revealed on each round
     {"numBoardCards", GameParameter(std::string("0 1"))},
     // Specify which actions are available to the player, in both limit and
     // nolimit games. Available options are: "fc" for fold and check/call.
     // "fcpa" for fold, check/call, bet pot and all in (default).
     // Use "fullgame" for the unabstracted game.
     {"bettingAbstraction", GameParameter(std::string("fcpa"))}}};

std::shared_ptr<const Game> Factory(const GameParameters &params) {
  return std::shared_ptr<const Game>(new UniversalPokerGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

// Returns how many actions are available at a choice node (3 when limit
// and 4 for no limit).
// TODO(author2): Is that a bug? There are 5 actions? Is no limit means
// "bet bot" is added? or should "all in" be also added?
inline uint32_t GetMaxBettingActions(const acpc_cpp::ACPCGame &acpc_game) {
  return acpc_game.IsLimitGame() ? 3 : 4;
}

// namespace universal_poker
UniversalPokerState::UniversalPokerState(std::shared_ptr<const Game> game)
    : State(game),
      acpc_game_(
          static_cast<const UniversalPokerGame *>(game.get())->GetACPCGame()),
      acpc_state_(acpc_game_),
      deck_(/*num_suits=*/acpc_game_->NumSuitsDeck(),
            /*num_ranks=*/acpc_game_->NumRanksDeck()),
      cur_player_(kChancePlayerId),
      possibleActions_(ACTION_DEAL),
      betting_abstraction_(static_cast<const UniversalPokerGame *>(game.get())
                               ->betting_abstraction()) {}

std::string UniversalPokerState::ToString() const {
  std::string str =
      absl::StrCat(BettingAbstractionToString(betting_abstraction_), "\n");
  for (int p = 0; p < acpc_game_->GetNbPlayers(); ++p) {
    absl::StrAppend(&str, "P", p, " Cards: ", HoleCards(p).ToString(), "\n");
  }
  absl::StrAppend(&str, "BoardCards ", BoardCards().ToString(), "\n");

  if (IsChanceNode()) {
    absl::StrAppend(&str, "PossibleCardsToDeal ", deck_.ToString(), "\n");
  }
  if (IsTerminal()) {
    for (int p = 0; p < acpc_game_->GetNbPlayers(); ++p) {
      absl::StrAppend(&str, "P", p, " Reward: ", GetTotalReward(p), "\n");
    }
  }
  absl::StrAppend(&str, "Node type?: ");
  if (IsChanceNode()) {
    absl::StrAppend(&str, "Chance node\n");
  } else if (IsTerminal()) {
    absl::StrAppend(&str, "Terminal Node!\n");
  } else {
    absl::StrAppend(&str, "Player node for player ", cur_player_, "\n");
  }

  if (betting_abstraction_ == BettingAbstraction::kFC ||
      betting_abstraction_ == BettingAbstraction::kFCPA) {
    absl::StrAppend(&str, "PossibleActions (", GetPossibleActionCount(),
                    "): [");
    for (StateActionType action : ALL_ACTIONS) {
      if (action & possibleActions_) {
        if (action == ACTION_ALL_IN) absl::StrAppend(&str, " ACTION_ALL_IN ");
        if (action == ACTION_BET) absl::StrAppend(&str, " ACTION_BET ");
        if (action == ACTION_CHECK_CALL) {
          absl::StrAppend(&str, " ACTION_CHECK_CALL ");
        }
        if (action == ACTION_FOLD) absl::StrAppend(&str, " ACTION_FOLD ");
        if (action == ACTION_DEAL) absl::StrAppend(&str, " ACTION_DEAL ");
      }
    }
  }
  absl::StrAppend(&str, "]", "\nRound: ", acpc_state_.GetRound(),
                  "\nACPC State: ", acpc_state_.ToString(),
                  "\nAction Sequence: ", actionSequence_);
  return str;
}

std::string UniversalPokerState::ActionToString(Player player,
                                                Action move) const {
  return absl::StrCat("player=", player, " move=", move);
}

bool UniversalPokerState::IsTerminal() const {
  bool finished = cur_player_ == kTerminalPlayerId;
  assert(acpc_state_.IsFinished() || !finished);
  return finished;
}

bool UniversalPokerState::IsChanceNode() const {
  return cur_player_ == kChancePlayerId;
}

Player UniversalPokerState::CurrentPlayer() const {
  if (IsTerminal()) {
    return kTerminalPlayerId;
  }
  if (IsChanceNode()) {
    return kChancePlayerId;
  }

  return Player(acpc_state_.CurrentPlayer());
}

std::vector<double> UniversalPokerState::Returns() const {
  if (!IsTerminal()) {
    return std::vector<double>(NumPlayers(), 0.0);
  }

  std::vector<double> returns(NumPlayers());
  for (Player player = 0; player < NumPlayers(); ++player) {
    // Money vs money at start.
    returns[player] = GetTotalReward(player);
  }

  return returns;
}

void UniversalPokerState::InformationStateTensor(
    Player player, absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);

  SPIEL_CHECK_EQ(values.size(), game_->InformationStateTensorShape()[0]);
  std::fill(values.begin(), values.end(), 0.);

  // Layout of observation:
  //   my player number: num_players bits
  //   my cards: Initial deck size bits (1 means you have the card), i.e.
  //             MaxChanceOutcomes() = NumSuits * NumRanks
  //   public cards: Same as above, but for the public cards.
  //   NumRounds() round sequence: (max round seq length)*2 bits
  int offset = 0;

  // Mark who I am.
  values[player] = 1;
  offset += NumPlayers();

  const logic::CardSet full_deck(acpc_game_->NumSuitsDeck(),
                                 acpc_game_->NumRanksDeck());
  const std::vector<uint8_t> deckCards = full_deck.ToCardArray();
  logic::CardSet holeCards = HoleCards(player);
  logic::CardSet boardCards = BoardCards();

  // TODO(author2): it should be way more efficient to iterate over the cards
  // of the player, rather than iterating over all the cards.
  for (uint32_t i = 0; i < full_deck.NumCards(); i++) {
    values[i + offset] = holeCards.ContainsCards(deckCards[i]) ? 1.0 : 0.0;
  }
  offset += full_deck.NumCards();

  // Public cards
  for (int i = 0; i < full_deck.NumCards(); ++i) {
    values[i + offset] = boardCards.ContainsCards(deckCards[i]) ? 1.0 : 0.0;
  }
  offset += full_deck.NumCards();

  const std::string actionSeq = GetActionSequence();
  const int length = actionSeq.length();
  SPIEL_CHECK_LT(length, game_->MaxGameLength());

  for (int i = 0; i < length; ++i) {
    SPIEL_CHECK_LT(offset + i + 1, values.size());
    if (actionSeq[i] == 'c') {
      // Encode call as 10.
      values[offset + (2 * i)] = 1;
      values[offset + (2 * i) + 1] = 0;
    } else if (actionSeq[i] == 'p') {
      // Encode raise as 01.
      values[offset + (2 * i)] = 0;
      values[offset + (2 * i) + 1] = 1;
    } else if (actionSeq[i] == 'a') {
      // Encode raise as 01.
      values[offset + (2 * i)] = 1;
      values[offset + (2 * i) + 1] = 1;
    } else if (actionSeq[i] == 'f') {
      // Encode fold as 00.
      // TODO(author2): Should this be 11?
      values[offset + (2 * i)] = 0;
      values[offset + (2 * i) + 1] = 0;
    } else if (actionSeq[i] == 'd') {
      values[offset + (2 * i)] = 0;
      values[offset + (2 * i) + 1] = 0;
    } else {
      SPIEL_CHECK_EQ(actionSeq[i], 'd');
    }
  }

  // Move offset up to the next round: 2 bits per move.
  offset += game_->MaxGameLength() * 2;
  SPIEL_CHECK_EQ(offset, game_->InformationStateTensorShape()[0]);
}

void UniversalPokerState::ObservationTensor(Player player,
                                            absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, NumPlayers());

  SPIEL_CHECK_EQ(values.size(), game_->ObservationTensorShape()[0]);
  std::fill(values.begin(), values.end(), 0.);

  // Layout of observation:
  //   my player number: num_players bits
  //   my cards: Initial deck size bits (1 means you have the card), i.e.
  //             MaxChanceOutcomes() = NumSuits * NumRanks
  //   public cards: Same as above, but for the public cards.
  //   the contribution of each player to the pot. num_players integers.
  int offset = 0;

  // Mark who I am.
  values[player] = 1;
  offset += NumPlayers();

  const logic::CardSet full_deck(acpc_game_->NumSuitsDeck(),
                                 acpc_game_->NumRanksDeck());
  const std::vector<uint8_t> all_cards = full_deck.ToCardArray();
  logic::CardSet holeCards = HoleCards(player);
  logic::CardSet boardCards = BoardCards();

  for (uint32_t i = 0; i < full_deck.NumCards(); i++) {
    values[i + offset] = holeCards.ContainsCards(all_cards[i]) ? 1.0 : 0.0;
  }
  offset += full_deck.NumCards();

  for (uint32_t i = 0; i < full_deck.NumCards(); i++) {
    values[i + offset] = boardCards.ContainsCards(all_cards[i]) ? 1.0 : 0.0;
  }
  offset += full_deck.NumCards();

  // Adding the contribution of each players to the pot.
  for (auto p = Player{0}; p < NumPlayers(); p++) {
    values[offset + p] = acpc_state_.Ante(p);
  }
  offset += NumPlayers();
  SPIEL_CHECK_EQ(offset, game_->ObservationTensorShape()[0]);
}

std::string UniversalPokerState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, acpc_game_->GetNbPlayers());
  const uint32_t pot = acpc_state_.MaxSpend() *
                       (acpc_game_->GetNbPlayers() - acpc_state_.NumFolded());
  std::vector<int> money;
  for (auto p = Player{0}; p < acpc_game_->GetNbPlayers(); p++) {
    money.emplace_back(acpc_state_.Money(p));
  }
  std::vector<std::string> sequences;
  for (auto r = 0; r <= acpc_state_.GetRound(); r++) {
    sequences.emplace_back(acpc_state_.BettingSequence(r));
  }

  return absl::StrFormat(
      "[Round %i][Player: %i][Pot: %i][Money: %s][Private: %s][Public: "
      "%s][Sequences: %s]",
      acpc_state_.GetRound(), CurrentPlayer(), pot, absl::StrJoin(money, " "),
      HoleCards(player).ToString(), BoardCards().ToString(),
      absl::StrJoin(sequences, "|"));
}

std::string UniversalPokerState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, acpc_game_->GetNbPlayers());
  std::string result;

  const uint32_t pot = acpc_state_.MaxSpend() *
                       (acpc_game_->GetNbPlayers() - acpc_state_.NumFolded());
  absl::StrAppend(&result, "[Round ", acpc_state_.GetRound(),
                  "][Player: ", CurrentPlayer(), "][Pot: ", pot, "][Money:");
  for (auto p = Player{0}; p < acpc_game_->GetNbPlayers(); p++) {
    absl::StrAppend(&result, " ", acpc_state_.Money(p));
  }
  // Add the player's private cards
  if (player != kChancePlayerId) {
    absl::StrAppend(&result, "[Private: ", HoleCards(player).ToString(), "]");
  }
  // Adding the contribution of each players to the pot
  absl::StrAppend(&result, "[Ante:");
  for (auto p = Player{0}; p < num_players_; p++) {
    absl::StrAppend(&result, " ", acpc_state_.Ante(p));
  }
  absl::StrAppend(&result, "]");

  return result;
}

std::unique_ptr<State> UniversalPokerState::Clone() const {
  return std::unique_ptr<State>(new UniversalPokerState(*this));
}

std::vector<std::pair<Action, double>> UniversalPokerState::ChanceOutcomes()
    const {
  SPIEL_CHECK_TRUE(IsChanceNode());
  auto available_cards = LegalActions();
  const int num_cards = available_cards.size();
  const double p = 1.0 / num_cards;

  // We need to convert std::vector<uint8_t> into std::vector<Action>.
  std::vector<std::pair<Action, double>> outcomes;
  outcomes.reserve(num_cards);
  for (const auto &card : available_cards) {
    outcomes.push_back({Action{card}, p});
  }
  return outcomes;
}

std::vector<Action> UniversalPokerState::LegalActions() const {
  if (IsChanceNode()) {
    const logic::CardSet full_deck(acpc_game_->NumSuitsDeck(),
                                   acpc_game_->NumRanksDeck());
    const std::vector<uint8_t> all_cards = full_deck.ToCardArray();
    std::vector<Action> actions;
    actions.reserve(deck_.NumCards());
    for (uint32_t i = 0; i < full_deck.NumCards(); i++) {
      if (deck_.ContainsCards(all_cards[i])) actions.push_back(i);
    }
    return actions;
  }

  std::vector<Action> legal_actions;

  if (betting_abstraction_ != BettingAbstraction::kFULLGAME) {
    if (ACTION_FOLD & possibleActions_) legal_actions.push_back(kFold);
    if (ACTION_CHECK_CALL & possibleActions_) legal_actions.push_back(kCall);
    if (ACTION_BET & possibleActions_) legal_actions.push_back(kBet);
    if (ACTION_ALL_IN & possibleActions_) legal_actions.push_back(kAllIn);
    return legal_actions;
  } else {
    if (acpc_state_.IsValidAction(
            acpc_cpp::ACPCState::ACPCActionType::ACPC_FOLD, 0)) {
      legal_actions.push_back(kFold);
    }
    if (acpc_state_.IsValidAction(
            acpc_cpp::ACPCState::ACPCActionType::ACPC_CALL, 0)) {
      legal_actions.push_back(kCall);
    }
    int32_t min_bet_size = 0;
    int32_t max_bet_size = 0;
    bool valid_to_raise =
        acpc_state_.RaiseIsValid(&min_bet_size, &max_bet_size);
    if (valid_to_raise) {
      const int big_blind =
          static_cast<const UniversalPokerGame *>(GetGame().get())->big_blind();
      SPIEL_CHECK_EQ(min_bet_size % big_blind, 0);
      for (int i = min_bet_size; i <= max_bet_size; i += big_blind) {
        legal_actions.push_back(1 + i / big_blind);
      }
    }
  }
  return legal_actions;
}

// We first deal the cards to each player, dealing all the cards to the first
// player first, then the second player, until all players have their private
// cards.
void UniversalPokerState::DoApplyAction(Action action_id) {
  if (IsChanceNode()) {
    // In chance nodes, the action_id is an index into the full deck.
    uint8_t card =
        logic::CardSet(acpc_game_->NumSuitsDeck(), acpc_game_->NumRanksDeck())
            .ToCardArray()[action_id];
    deck_.RemoveCard(card);
    actionSequence_ += 'd';

    // Check where to add this card
    if (hole_cards_dealt_ <
        acpc_game_->GetNbPlayers() * acpc_game_->GetNbHoleCardsRequired()) {
      AddHoleCard(card);
      _CalculateActionsAndNodeType();
      return;
    }

    if (board_cards_dealt_ <
        acpc_game_->GetNbBoardCardsRequired(acpc_state_.GetRound())) {
      AddBoardCard(card);
      _CalculateActionsAndNodeType();
      return;
    }
  } else {
    int action_int = static_cast<int>(action_id);
    if (action_int == kFold) {
      ApplyChoiceAction(ACTION_FOLD, 0);
      return;
    }
    if (action_int == kCall) {
      ApplyChoiceAction(ACTION_CHECK_CALL, 0);
      return;
    }
    if (betting_abstraction_ != BettingAbstraction::kFULLGAME) {
      if (action_int == kBet) {
        ApplyChoiceAction(ACTION_BET, potSize_);
        return;
      }
      if (action_int == kAllIn) {
        ApplyChoiceAction(ACTION_ALL_IN, allInSize_);
        return;
      }
    }
    if (action_int >= 2 && action_int <= NumDistinctActions()) {
      const int big_blind =
          static_cast<const UniversalPokerGame *>(GetGame().get())->big_blind();
      ApplyChoiceAction(ACTION_BET, (action_int - 1) * big_blind);
      return;
    }
    SpielFatalError(absl::StrFormat("Action not recognized: %i", action_id));
  }
}

double UniversalPokerState::GetTotalReward(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, acpc_game_->GetNbPlayers());
  return acpc_state_.ValueOfState(player);
}

std::unique_ptr<HistoryDistribution>
UniversalPokerState::GetHistoriesConsistentWithInfostate(int player_id) const {
  // This is only implemented for 2 players.
  if (acpc_game_->GetNbPlayers() != 2) return {};

  logic::CardSet is_cards;
  logic::CardSet our_cards = HoleCards(player_id);
  for (uint8_t card : our_cards.ToCardArray()) is_cards.AddCard(card);
  for (uint8_t card : BoardCards().ToCardArray()) is_cards.AddCard(card);
  logic::CardSet fresh_deck(/*num_suits=*/acpc_game_->NumSuitsDeck(),
                            /*num_ranks=*/acpc_game_->NumRanksDeck());
  for (uint8_t card : is_cards.ToCardArray()) fresh_deck.RemoveCard(card);
  auto dist = std::make_unique<HistoryDistribution>();
  for (uint8_t hole_card1 : fresh_deck.ToCardArray()) {
    logic::CardSet subset_deck = fresh_deck;
    subset_deck.RemoveCard(hole_card1);
    for (uint8_t hole_card2 : subset_deck.ToCardArray()) {
      if (hole_card1 < hole_card2) continue;
      std::unique_ptr<State> root = game_->NewInitialState();
      if (player_id == 0) {
        for (uint8_t card : our_cards.ToCardArray()) root->ApplyAction(card);
        root->ApplyAction(hole_card1);
        root->ApplyAction(hole_card2);
      } else if (player_id == 1) {
        root->ApplyAction(hole_card1);
        root->ApplyAction(hole_card2);
        for (uint8_t card : our_cards.ToCardArray()) root->ApplyAction(card);
      }
      SPIEL_CHECK_FALSE(root->IsChanceNode());
      dist->first.push_back(std::move(root));
      dist->second.push_back(1.);
    }
  }
  dist->second.resize(dist->first.size(),
                      1. / static_cast<double>(dist->first.size()));
  return dist;
}

/**
 * Universal Poker Game Constructor
 * @param params
 */
UniversalPokerGame::UniversalPokerGame(const GameParameters &params)
    : Game(kGameType, params),
      gameDesc_(parseParameters(params)),
      acpc_game_(gameDesc_) {
  max_game_length_ = MaxGameLength();
  SPIEL_CHECK_TRUE(max_game_length_.has_value());
  std::string betting_abstraction =
      ParameterValue<std::string>("bettingAbstraction");
  if (betting_abstraction == "fc") {
    betting_abstraction_ = BettingAbstraction::kFC;
  } else if (betting_abstraction == "fcpa") {
    betting_abstraction_ = BettingAbstraction::kFCPA;
  } else if (betting_abstraction == "fullgame") {
    betting_abstraction_ = BettingAbstraction::kFULLGAME;
  } else {
    SpielFatalError(absl::StrFormat("bettingAbstraction: %s not supported.",
                                    betting_abstraction));
  }
}

std::unique_ptr<State> UniversalPokerGame::NewInitialState() const {
  return absl::make_unique<UniversalPokerState>(shared_from_this());
}

std::vector<int> UniversalPokerGame::InformationStateTensorShape() const {
  // One-hot encoding for player number (who is to play).
  // 2 slots of cards (total_num_cards bits each): private card, public card
  // Followed by maximum game length * 2 bits each (call / raise)
  const int num_players = acpc_game_.GetNbPlayers();
  const int gameLength = MaxGameLength();
  const int total_num_cards = MaxChanceOutcomes();

  return {num_players + 2 * total_num_cards + 2 * gameLength};
}

std::vector<int> UniversalPokerGame::ObservationTensorShape() const {
  // One-hot encoding for player number (who is to play).
  // 2 slots of cards (total_cards_ bits each): private card, public card
  // Followed by the contribution of each player to the pot
  const int num_players = acpc_game_.GetNbPlayers();
  const int total_num_cards = MaxChanceOutcomes();
  return {2 * (num_players + total_num_cards)};
}

double UniversalPokerGame::MaxUtility() const {
  // In poker, the utility is defined as the money a player has at the end of
  // the game minus then money the player had before starting the game.
  // The most a player can win *per opponent* is the most each player can put
  // into the pot, which is the raise amounts on each round times the maximum
  // number raises, plus the original chip they put in to play.

  return (double)acpc_game_.StackSize(0) * (acpc_game_.GetNbPlayers() - 1);
}

double UniversalPokerGame::MinUtility() const {
  // In poker, the utility is defined as the money a player has at the end of
  // the game minus then money the player had before starting the game.
  // The most any single player can lose is the maximum number of raises per
  // round times the amounts of each of the raises, plus the original chip they
  // put in to play.
  return -1. * (double)acpc_game_.StackSize(0);
}

int UniversalPokerGame::MaxChanceOutcomes() const {
  return acpc_game_.NumSuitsDeck() * acpc_game_.NumRanksDeck();
}

int UniversalPokerGame::NumPlayers() const { return acpc_game_.GetNbPlayers(); }

int UniversalPokerGame::NumDistinctActions() const {
  if (betting_abstraction_ == BettingAbstraction::kFULLGAME) {
    // fold, check/call, bet/raise some multiple of BBs
    return starting_stack_big_blinds_ + 2;
  } else {
    return GetMaxBettingActions(acpc_game_);
  }
}

std::shared_ptr<const Game> UniversalPokerGame::Clone() const {
  return std::shared_ptr<const Game>(new UniversalPokerGame(*this));
}

int UniversalPokerGame::MaxGameLength() const {
  // We cache this as this is very slow to calculate.
  if (max_game_length_) return *max_game_length_;

  // Make a good guess here because bruteforcing the tree is far too slow
  // One Terminal Action
  int length = 1;

  // Deal Actions
  length += acpc_game_.GetTotalNbBoardCards() +
            acpc_game_.GetNbHoleCardsRequired() * acpc_game_.GetNbPlayers();

  // Check Actions
  length += (NumPlayers() * acpc_game_.NumRounds());

  // Bet Actions
  double maxStack = 0;
  double maxBlind = 0;
  for (uint32_t p = 0; p < NumPlayers(); p++) {
    maxStack =
        acpc_game_.StackSize(p) > maxStack ? acpc_game_.StackSize(p) : maxStack;
    maxBlind =
        acpc_game_.BlindSize(p) > maxStack ? acpc_game_.BlindSize(p) : maxBlind;
  }

  while (maxStack > maxBlind) {
    maxStack /= 2.0;         // You have always to bet the pot size
    length += NumPlayers();  // Each player has to react
  }
  return length;
}

/**
 * Parses the Game Paramters and makes a gameDesc out of it
 * @param map
 * @return
 */
std::string UniversalPokerGame::parseParameters(const GameParameters &map) {
  if (map.find("gamedef") != map.end()) {
    // We check for sanity that all parameters are empty
    if (map.size() != 1) {
      std::vector<std::string> game_parameter_keys;
      game_parameter_keys.reserve(map.size());
      for (auto const &imap : map) {
        game_parameter_keys.push_back(imap.first);
      }
      SpielFatalError(
          absl::StrCat("When loading a 'universal_poker' game, the 'gamedef' "
                       "field was present, but other fields were present too: ",
                       absl::StrJoin(game_parameter_keys, ", "),
                       "gamedef is exclusive with other paraemters."));
    }
    return ParameterValue<std::string>("gamedef");
  }

  std::string generated_gamedef = "GAMEDEF\n";

  absl::StrAppend(
      &generated_gamedef, ParameterValue<std::string>("betting"), "\n",
      "numPlayers = ", ParameterValue<int>("numPlayers"), "\n",
      "numRounds = ", ParameterValue<int>("numRounds"), "\n",
      "numsuits = ", ParameterValue<int>("numSuits"), "\n",
      "firstPlayer = ", ParameterValue<std::string>("firstPlayer"), "\n",
      "numRanks = ", ParameterValue<int>("numRanks"), "\n",
      "numHoleCards = ", ParameterValue<int>("numHoleCards"), "\n",
      "numBoardCards = ", ParameterValue<std::string>("numBoardCards"), "\n");

  std::string max_raises = ParameterValue<std::string>("maxRaises");
  if (!max_raises.empty()) {
    absl::StrAppend(&generated_gamedef, "maxRaises = ", max_raises, "\n");
  }

  if (ParameterValue<std::string>("betting") == "limit") {
    std::string raise_size = ParameterValue<std::string>("raiseSize");
    if (!raise_size.empty()) {
      absl::StrAppend(&generated_gamedef, "raiseSize = ", raise_size, "\n");
    }
  } else if (ParameterValue<std::string>("betting") == "nolimit") {
    std::string stack = ParameterValue<std::string>("stack");
    if (!stack.empty()) {
      absl::StrAppend(&generated_gamedef, "stack = ", stack, "\n");
    }
  } else {
    SpielFatalError(absl::StrCat("betting should be limit or nolimit, not ",
                                 ParameterValue<std::string>("betting")));
  }

  absl::StrAppend(&generated_gamedef,
                  "blind = ", ParameterValue<std::string>("blind"), "\n");
  absl::StrAppend(&generated_gamedef, "END GAMEDEF\n");

  std::vector<std::string> blinds =
      absl::StrSplit(ParameterValue<std::string>("blind"), ' ');
  std::vector<std::string> stacks =
      absl::StrSplit(ParameterValue<std::string>("stack"), ' ');
  big_blind_ = std::max(std::stoi(blinds[0]), std::stoi(blinds[1]));
  starting_stack_big_blinds_ =
      std::stoi(stacks[0]);  // assumes all stack sizes are equal

  return generated_gamedef;
}

const char *actions = "0df0c000p0000000a";

void UniversalPokerState::ApplyChoiceAction(StateActionType action_type,
                                            int size) {
  SPIEL_CHECK_GE(cur_player_, 0);  // No chance not terminal.

  actionSequence_ += (char)actions[action_type];
  switch (action_type) {
    case ACTION_FOLD:
      acpc_state_.DoAction(acpc_cpp::ACPCState::ACPCActionType::ACPC_FOLD, 0);
      break;
    case ACTION_CHECK_CALL:
      acpc_state_.DoAction(acpc_cpp::ACPCState::ACPCActionType::ACPC_CALL, 0);
      break;
    case ACTION_BET:
      if (betting_abstraction_ == BettingAbstraction::kFULLGAME) {
        acpc_state_.DoAction(acpc_cpp::ACPCState::ACPCActionType::ACPC_RAISE,
                             size);
      } else {
        acpc_state_.DoAction(acpc_cpp::ACPCState::ACPCActionType::ACPC_RAISE,
                             potSize_);
      }
      break;
    case ACTION_ALL_IN:
      acpc_state_.DoAction(acpc_cpp::ACPCState::ACPCActionType::ACPC_RAISE,
                           allInSize_);
      break;
    case ACTION_DEAL:
    default:
      assert(false);
      break;
  }

  _CalculateActionsAndNodeType();
}

void UniversalPokerState::_CalculateActionsAndNodeType() {
  possibleActions_ = 0;

  if (acpc_state_.IsFinished()) {
    if (acpc_state_.NumFolded() >= acpc_game_->GetNbPlayers() - 1) {
      // All players except one has fold.
      cur_player_ = kTerminalPlayerId;
    } else {
      if (board_cards_dealt_ <
          acpc_game_->GetNbBoardCardsRequired(acpc_state_.GetRound())) {
        cur_player_ = kChancePlayerId;
        possibleActions_ = ACTION_DEAL;
        return;
      }
      // Showdown!
      cur_player_ = kTerminalPlayerId;
    }

  } else {
    // Check if we need to deal cards. We assume all cards are dealt at the
    // start of the game.
    if (hole_cards_dealt_ <
        acpc_game_->GetNbHoleCardsRequired() * acpc_game_->GetNbPlayers()) {
      cur_player_ = kChancePlayerId;
      possibleActions_ = ACTION_DEAL;
      return;
    }
    // 2. We need to deal a public card.
    if (board_cards_dealt_ <
        acpc_game_->GetNbBoardCardsRequired(acpc_state_.GetRound())) {
      cur_player_ = kChancePlayerId;
      possibleActions_ = ACTION_DEAL;
      return;
    }

    // Check for CHOICE Actions
    cur_player_ = acpc_state_.CurrentPlayer();
    if (acpc_state_.IsValidAction(
            acpc_cpp::ACPCState::ACPCActionType::ACPC_FOLD, 0)) {
      possibleActions_ |= ACTION_FOLD;
    }
    if (acpc_state_.IsValidAction(
            acpc_cpp::ACPCState::ACPCActionType::ACPC_CALL, 0)) {
      possibleActions_ |= ACTION_CHECK_CALL;
    }

    potSize_ = 0;
    allInSize_ = 0;
    // We have to call this as this sets potSize_ and allInSize_.
    bool valid_to_raise = acpc_state_.RaiseIsValid(&potSize_, &allInSize_);
    if (betting_abstraction_ == BettingAbstraction::kFC) return;
    if (valid_to_raise) {
      if (acpc_game_->IsLimitGame()) {
        potSize_ = 0;
        // There's only one "bet" allowed in Limit, which is "all-in or fixed
        // bet".
        possibleActions_ |= ACTION_BET;
      } else {
        int cur_spent = acpc_state_.CurrentSpent(acpc_state_.CurrentPlayer());
        int pot_raise_to =
            acpc_state_.TotalSpent() + 2 * acpc_state_.MaxSpend() - cur_spent;

        if (pot_raise_to >= potSize_ && pot_raise_to <= allInSize_) {
          potSize_ = pot_raise_to;
          possibleActions_ |= ACTION_BET;
        }

        if (pot_raise_to != allInSize_) {
          // If the raise to amount happens to match the number of chips I have,
          // then this action was already added as a pot-bet.
          possibleActions_ |= ACTION_ALL_IN;
        }
      }
    }
  }
}

const int UniversalPokerState::GetPossibleActionCount() const {
  // _builtin_popcount(int) function is used to count the number of one's
  return __builtin_popcount(possibleActions_);
}

std::ostream &operator<<(std::ostream &os, const BettingAbstraction &betting) {
  os << BettingAbstractionToString(betting);
  return os;
}

}  // namespace universal_poker
}  // namespace open_spiel
