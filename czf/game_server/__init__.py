'''CZF Game Server'''
import argparse
import asyncio
from functools import partial
from pathlib import Path
import random
from uuid import uuid4

import numpy as np
import yaml
import zmq.asyncio

from czf.game_server.game_server import GameServer
from czf.game_server.evaluator import EvalGameServer


def make_default_action_policy_fn(softmax_step, num_moves, legal_actions, legal_actions_policy):
    '''Default action policy: switch betweens softmax and argmax action policy'''
    if num_moves < softmax_step:
        return softmax_action_policy_fn(num_moves, legal_actions, legal_actions_policy)
    return argmax_action_policy_fn(num_moves, legal_actions, legal_actions_policy)


def softmax_action_policy_fn(_, legal_actions, legal_actions_policy):
    '''Softmax action policy'''
    return random.choices(legal_actions, legal_actions_policy)[0]


def argmax_action_policy_fn(_, legal_actions, legal_actions_policy):
    '''Argmax action policy'''
    return legal_actions[np.argmax(legal_actions_policy)]


def eval_after_apply_metric(board_shape, evaluated_state, game_state):
    '''Callback for the evaluation result after applying an action'''
    policy = evaluated_state.evaluation.policy
    value = evaluated_state.evaluation.value
    print(np.array(policy).reshape(board_shape), value, game_state, sep='\n')


async def main(args, config, callbacks):
    '''czf.game_server main program'''
    sever_cls = EvalGameServer if args.eval else GameServer
    server = sever_cls(args, config, callbacks)
    try:
        await server.loop()
    except asyncio.CancelledError:
        server.terminate()


def run_main():
    '''Run main program in asyncio'''
    parser = argparse.ArgumentParser(__package__, description=__doc__)
    parser.add_argument('-f', '--config', required=True, help='config file')
    parser.add_argument('-np',
                        '--num-proc',
                        type=int,
                        default=1,
                        help='number of EnvManager process')
    parser.add_argument('-n',
                        '--num-env',
                        type=int,
                        required=True,
                        help='number of environments per process')
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
                        default=uuid4().hex,
                        help='unique id of the game server')
    parser.add_argument('--eval', action='store_true', help='evaluation mode')
    parser.add_argument('-s', '--storage-dir', help='path to store model, trajectory, and log')
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    softmax_step = config['game_server']['softmax_temperature_step']
    callbacks = {
        'metric': {},
    }
    # default action policy
    if isinstance(softmax_step, bool):
        if softmax_step:
            callbacks['action_policy'] = softmax_action_policy_fn
        else:
            callbacks['action_policy'] = argmax_action_policy_fn
    elif isinstance(softmax_step, int):
        callbacks['action_policy'] = partial(make_default_action_policy_fn, softmax_step)
    if args.eval:
        callbacks['action_policy'] = argmax_action_policy_fn
        # np.set_printoptions(precision=3)
        # board_shape = config['game']['observation_shape'][-2:]
        # callbacks['metric']['after_apply'] = partial(eval_after_apply_metric, board_shape)
    try:
        asyncio.run(main(args, config, callbacks))
    except KeyboardInterrupt:
        zmq.asyncio.Context.instance().destroy()
        print('\rterminated by ctrl-c')


if __name__ == '__main__':
    run_main()
