#include "czf.h"

namespace czf {

GameRegistration::GameRegistration(
    std::function<std::shared_ptr<const Game>()> create_function) {
  factories()[create_function()->ShortName()] = create_function;
}

std::shared_ptr<const Game> GameRegistration::Create(
    const std::string& short_name) {
  return factories()[short_name]();
}

std::map<std::string, GameRegistration::CreateFunc>&
GameRegistration::factories() {
  static std::map<std::string, GameRegistration::CreateFunc> impl;
  return impl;
}

std::shared_ptr<const Game> LoadGame(const std::string& short_name) {
  return GameRegistration::Create(short_name);
}
}  // namespace czf