#pragma once

#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

namespace czf::env::czf_env {

using Player = int;
using Action = int;

class State;
using StatePtr = std::unique_ptr<State>;

class Game;
using GamePtr = std::shared_ptr<const Game>;

class State {
 public:
  State(GamePtr game) : game_(game) {}
  virtual ~State() = default;
  virtual Player current_player() const = 0;
  virtual std::vector<Action> legal_actions() const = 0;
  virtual std::vector<float> observation_tensor() const = 0;
  virtual std::vector<float> feature_tensor() const;
  virtual void apply_action(const Action&) = 0;
  virtual bool is_terminal() const = 0;
  virtual std::vector<float> rewards() const = 0;
  virtual StatePtr clone() const = 0;
  virtual std::string to_string() const { return ""; }
  GamePtr game() const { return game_; }

  virtual std::string serialize() const { return ""; }

 protected:
  GamePtr game_;
};

class Game : public std::enable_shared_from_this<Game> {
 public:
  virtual ~Game() = default;
  virtual std::string name() const = 0;
  virtual int num_players() const = 0;
  virtual int num_distinct_actions() const = 0;
  virtual StatePtr new_initial_state() const = 0;
  virtual std::vector<int> observation_tensor_shape() const = 0;

  virtual int num_transformations() const;
  virtual std::vector<float> transform_observation(const std::vector<float>&, int) const;
  virtual std::vector<float> transform_policy(const std::vector<float>&, int) const;
  virtual std::vector<float> restore_policy(const std::vector<float>&, int) const;

  virtual std::string action_to_string(const Action& action) const {
    return std::to_string(action);
  }
  virtual std::vector<Action> string_to_action(const std::string& str) const {
    return {std::stoi(str)};
  }
  virtual StatePtr deserialize_state(const std::string& str = "") const {
    return new_initial_state();
  }
};

class GameFactory {
 public:
  using CreateFunc = std::function<GamePtr()>;
  static GameFactory& instance();
  void add(const CreateFunc&);
  std::vector<std::string> games() const;
  GamePtr create(const std::string& name);

 private:
  std::unordered_map<std::string, CreateFunc> factory_;
};

template <class GameType>
struct Registration {
  Registration() {
    GameFactory::instance().add([]() { return std::make_shared<GameType>(); });
  }
};

GamePtr load_game(const std::string&);
std::vector<std::string> available_games();

}  // namespace czf::env::czf_env