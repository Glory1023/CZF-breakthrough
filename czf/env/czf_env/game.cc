#include "czf/env/czf_env/game.h"

#include <algorithm>
#include <sstream>

namespace czf::env::czf_env {

std::vector<float> State::feature_tensor() const { return observation_tensor(); }

int Game::num_transformations() const { return 0; }

std::vector<float> Game::transform_observation(const std::vector<float>& observation,
                                               int type = 0) const {
  return std::vector<float>(observation);
}

std::vector<float> Game::transform_policy(const std::vector<float>& policy, int type = 0) const {
  return std::vector<float>(policy);
}

std::vector<float> Game::restore_policy(const std::vector<float>& policy, int type = 0) const {
  return std::vector<float>(policy);
}

GameFactory& GameFactory::instance() {
  static GameFactory impl;
  return impl;
}

void GameFactory::add(const CreateFunc& create_func) {
  factory_[create_func()->name()] = create_func;
}

std::vector<std::string> GameFactory::games() const {
  std::vector<std::string> names;
  names.reserve(factory_.size());
  std::transform(factory_.begin(), factory_.end(), std::back_inserter(names),
                 [](auto p) { return p.first; });
  return names;
}

GamePtr GameFactory::create(const std::string& name) { return factory_[name](); }

GamePtr load_game(const std::string& name) { return GameFactory::instance().create(name); }
std::vector<std::string> available_games() { return GameFactory::instance().games(); }

}  // namespace czf::env::czf_env