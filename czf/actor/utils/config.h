#pragma once

namespace czf::actor {
namespace GameOption {
const constexpr auto ActionDim = 9;
}

namespace BuildOption {
#ifdef VERBOSE
const constexpr auto Verbose = true;
#else
const constexpr auto Verbose = false;
#endif
#ifdef SELF_PLAY
const constexpr auto SelfPlay = true;
#else
const constexpr auto SelfPlay = false;
#endif
const constexpr auto OmpNumThread = 48;
const constexpr auto TorchNumIntraThread = 1;
const constexpr auto TorchNumInterThread = 1;
const constexpr auto FloatEps = 1e-9;
}  // namespace BuildOption

namespace MctsOption {
constexpr const auto SimulationCount = 800u;
constexpr const auto C_PUCT = 1.25f;
constexpr const auto SoftmaxMove = 30ul;
constexpr const auto DirichletParam = .03f;
constexpr const auto DirichletEpsilon = .25f;
constexpr const auto Discount = 1.f;
}  // namespace MctsOption
}  // namespace czf::actor
