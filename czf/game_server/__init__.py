'''CZF Game Server'''
import argparse
import asyncio
import random
from uuid import uuid4
import numpy as np
import zmq.asyncio

from czf.game_server.game_server import GameServer


def default_action_policy_fn(num_moves, legal_actions, legal_actions_policy):
    '''Default action policy: switch betweens softmax and argmax action policy'''
    if num_moves < 10:
        return softmax_action_policy_fn(num_moves, legal_actions,
                                        legal_actions_policy)
    return argmax_action_policy_fn(num_moves, legal_actions,
                                   legal_actions_policy)


def softmax_action_policy_fn(_, legal_actions, legal_actions_policy):
    '''Softmax action policy'''
    return random.choices(legal_actions, legal_actions_policy)[0]


def argmax_action_policy_fn(_, legal_actions, legal_actions_policy):
    '''Argmax action policy'''
    return legal_actions[np.argmax(legal_actions_policy)]


def eval_after_apply_metric(evaluated_state, game_state):
    '''Callback for the evaluation result after applying an action'''
    np.set_printoptions(precision=3)
    policy = evaluated_state.evaluation.policy
    value = evaluated_state.evaluation.value
    print(np.array(policy).reshape(3, 3), value, game_state, sep='\n')


async def main(args, callbacks):
    '''czf.game_server main program'''
    game_server = GameServer(args, callbacks)
    await game_server.loop()


def run_main():
    '''Run main program in asyncio'''
    parser = argparse.ArgumentParser(__package__, description=__doc__)
    parser.add_argument('-e',
                        '--env',
                        required=True,
                        help='czf_env game environment')
    parser.add_argument('-n', '--num-env', type=int, required=True)
    parser.add_argument('-b',
                        '--broker',
                        required=True,
                        metavar='host:port',
                        help='broker address. e.g., 127.0.0.1:5566')
    parser.add_argument('-u',
                        '--upstream',
                        required=True,
                        metavar='host:port',
                        help='learner address. e.g., 127.0.0.1:5577')
    parser.add_argument('--suffix',
                        metavar='unique_id',
                        help='unique id of the game server',
                        default=uuid4().hex)
    parser.add_argument('--eval',
                        action='store_true',
                        help='evaluation only (without sending trajectories)')
    args = parser.parse_args()

    callbacks = {
        'action_policy': default_action_policy_fn,
        'metric': {},
    }
    if args.eval:
        callbacks['action_policy'] = argmax_action_policy_fn
        callbacks['metric']['after_apply'] = eval_after_apply_metric
    try:
        asyncio.run(main(args, callbacks))
    except KeyboardInterrupt:
        zmq.asyncio.Context.instance().destroy()
        print('\rterminated by ctrl-c')


if __name__ == '__main__':
    run_main()
