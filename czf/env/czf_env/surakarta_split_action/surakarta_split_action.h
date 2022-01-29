#pragma once
#include <array>
#include <map>

#include "czf/env/czf_env/game.h"

namespace czf::env::czf_env::surakarta {

constexpr int kBoardSize = 6;
constexpr int kNumOfGrids = kBoardSize * kBoardSize;
constexpr int kPolicyDim = kNumOfGrids;

/** Maximum number of steps */
constexpr int kMaxTurn = INT16_MAX;
/** Maximum number of repetitions */
constexpr int kMaxRepeatBoard = 2;
/** Maximum number of consecutive steps without capture */
constexpr int kMaxNoCaptureTurn = 50;

class SurakartaSplitActionState final : public State {
 public:
  SurakartaSplitActionState(GamePtr);
  SurakartaSplitActionState(const SurakartaSplitActionState &) = default;
  StatePtr clone() const override;
  void apply_action(const Action &) override;
  std::vector<Action> legal_actions() const override;
  bool is_terminal() const override;

  Player current_player() const override;
  std::vector<float> rewards() const override;
  std::vector<float> observation_tensor() const override;
  std::string to_string() const override;

  std::string serialize() const override;

  bool has_piece(const Player &) const;

 private:
  bool is_legal_action(const Action &action) const;
  bool is_legal_capture_this_circle(const Action &action, const Action &src,
                                    const std::array<int, 56> &circle) const;
  bool have_move(const Action &action) const;
  bool have_capture(const Action &action) const;
  bool have_capture_this_circle(const Action &action, const std::array<int, 56> &circle) const;
  /** 0 -> B (black)
   * 1 -> W (white)
   * 2 -> E (empty)*/
  enum Piece { BLACK, WHITE, EMPTY };
  /** number      // mean        // index
   * 1 1 1 1 1 1 // W W W W W W //  0  1  2  3  4  5
   * 1 1 1 1 1 1 // W W W W W W //  6  7  8  9 10 11
   * 2 2 2 2 2 2 // E E E E E E // 12 13 14 15 16 17
   * 2 2 2 2 2 2 // E E E E E E // 18 19 20 21 22 23
   * 0 0 0 0 0 0 // B B B B B B // 24 25 26 27 28 29
   * 0 0 0 0 0 0 // B B B B B B // 30 31 32 33 34 35 */
  std::array<short, kNumOfGrids> board_;
  /** Game rule for terminal: the board_ repeat 3 times */
  std::map<std::array<short, kNumOfGrids>, short> repeat_;
  std::vector<Action> history_;
  int turn_;
  int without_capture_turn_;
  int black_piece_;
  int white_piece_;

  Player winner_;
};

class SurakartaSplitActionGame final : public Game {
 public:
  std::string name() const override;
  int num_players() const override;
  int num_distinct_actions() const override;
  StatePtr new_initial_state() const override;
  std::vector<int> observation_tensor_shape() const override;

  int num_transformations() const override;
  std::vector<float> transform_observation(const std::vector<float> &observation,
                                           int) const override;
  std::vector<float> transform_policy(const std::vector<float> &policy, int) const override;
  std::vector<float> restore_policy(const std::vector<float> &policy, int) const override;

  std::string action_to_string(const Action &action) const override;
  std::vector<Action> string_to_action(const std::string &str) const override;
  StatePtr deserialize_state(const std::string &) const override;

 private:
  int transform_index(const int &index, int) const;
};

// *IMPORTANT* Register this game to the factory
namespace {
Registration<SurakartaSplitActionGame> registration;
}

constexpr bool kIsOuter[kNumOfGrids] = {
    false, false, true, true, false, false, false, false, true, true, false, false,
    true,  true,  true, true, true,  true,  true,  true,  true, true, true,  true,
    false, false, true, true, false, false, false, false, true, true, false, false};
constexpr bool kIsInter[kNumOfGrids] = {
    false, true, false, false, true, false, true,  true, true,  true,  true, true,
    false, true, false, false, true, false, false, true, false, false, true, false,
    true,  true, true,  true,  true, true,  false, true, false, false, true, false};
constexpr std::array<int, 56> kOuter = {32, 26, 20, 14, 8,  2,  -1, 12, 13, 14, 15, 16, 17, -1,
                                        3,  9,  15, 21, 27, 33, -1, 23, 22, 21, 20, 19, 18, -1,
                                        32, 26, 20, 14, 8,  2,  -1, 12, 13, 14, 15, 16, 17, -1,
                                        3,  9,  15, 21, 27, 33, -1, 23, 22, 21, 20, 19, 18, -1};
constexpr std::array<int, 56> kOuterReverse = {
    18, 19, 20, 21, 22, 23, -1, 33, 27, 21, 15, 9,  3,  -1, 17, 16, 15, 14, 13,
    12, -1, 2,  8,  14, 20, 26, 32, -1, 18, 19, 20, 21, 22, 23, -1, 33, 27, 21,
    15, 9,  3,  -1, 17, 16, 15, 14, 13, 12, -1, 2,  8,  14, 20, 26, 32, -1};
constexpr std::array<int, 56> kInter = {1,  7,  13, 19, 25, 31, -1, 24, 25, 26, 27, 28, 29, -1,
                                        34, 28, 22, 16, 10, 4,  -1, 11, 10, 9,  8,  7,  6,  -1,
                                        1,  7,  13, 19, 25, 31, -1, 24, 25, 26, 27, 28, 29, -1,
                                        34, 28, 22, 16, 10, 4,  -1, 11, 10, 9,  8,  7,  6,  -1};
constexpr std::array<int, 56> kInterReverse = {
    6,  7,  8,  9,  10, 11, -1, 4,  10, 16, 22, 28, 34, -1, 29, 28, 27, 26, 25,
    24, -1, 31, 25, 19, 13, 7,  1,  -1, 6,  7,  8,  9,  10, 11, -1, 4,  10, 16,
    22, 28, 34, -1, 29, 28, 27, 26, 25, 24, -1, 31, 25, 19, 13, 7,  1,  -1};
constexpr std::array<int, 8> MoveDirection = {-7, -6, -5, -1, +1, +5, +6, +7};
}  // namespace czf::env::czf_env::surakarta
