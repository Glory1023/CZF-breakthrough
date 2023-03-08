#pragma once

#include <array>

#include "czf/env/czf_env/game.h"

namespace czf::env::czf_env::breakthrough {

    class BreakThroughState final : public State {
        public:
            BreakThroughState(GamePtr);
            BreakThroughState(const BreakThroughState&) = default;

            StatePtr clone() const override;
            void apply_action(const Action&) override;
            std::vector<Action> legal_actions() const override;
            bool is_terminal() const override;

            Player current_player() const override;
            std::vector<float> rewards() const override;
            std::vector<float> observation_tensor() const override;//return by feature_tensor
            std::string to_string() const override;

            bool arrive(const Player&) const;

            std::string serialize() const override;

            

        private:
            void set_all_possible_move();
            void set_btpm();
            //bool is_legal_action(const Action &action) const;
            //0 -> white
            //1 -> black
            //2 -> empty
            std::array<short, 64> board_;
            int possible_move[308];
            int board_possible_move[112][3];
            int chess_num[2];
            //int after_move[48];
            //int before_move[48];
            int turn_;
            Player winner_;
            std::vector<Action> history_;
    };

    class BreakThroughGame final : public Game {
        public:
            std::string name() const override;
            int num_players() const override;
            int num_distinct_actions() const override;
            StatePtr new_initial_state() const override;
            std::vector<int> observation_tensor_shape() const override;

            int num_transformations() const override;
            std::vector<float> transform_observation(const std::vector<float>&, int) const override;
            std::vector<float> transform_policy(const std::vector<float>&, int) const override;
            std::vector<float> restore_policy(const std::vector<float>&, int) const override;

            StatePtr deserialize_state(const std::string& str = "") const override;
    };


    namespace {//use for what?
        Registration<BreakThroughGame> registration;
    }
}
