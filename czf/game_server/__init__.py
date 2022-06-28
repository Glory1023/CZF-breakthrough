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

from czf.pb import czf_pb2
from czf.utils import get_zmq_dealer
import platform

from czf.game_server.game_server import GameServer
from czf.game_server.evaluator import EvalGameServer


def make_default_action_policy_fn(softmax_temperature, simulation_count, num_moves,
                                  model_iteration, legal_actions, legal_actions_policy):
    '''Default action policy: switch betweens softmax and argmax action policy'''
    steps = softmax_temperature['steps']
    temperatures = softmax_temperature['temperatures']
    current_temperature = None
    if isinstance(temperatures, float):
        current_temperature = temperatures
    elif isinstance(temperatures, list):
        for (iteration, temperature) in temperatures:
            # print(iteration, temperature)
            if model_iteration <= iteration:
                current_temperature = temperature
                break

    if not isinstance(current_temperature, float):
        return argmax_action_policy_fn(
            num_moves,
            model_iteration,
            legal_actions,
            legal_actions_policy,
        )
    if isinstance(steps, bool):
        if steps:
            return softmax_action_policy_fn(
                current_temperature,
                simulation_count,
                legal_actions,
                legal_actions_policy,
            )
        else:
            return argmax_action_policy_fn(
                num_moves,
                model_iteration,
                legal_actions,
                legal_actions_policy,
            )
    if isinstance(steps, int) and num_moves < steps:
        return softmax_action_policy_fn(
            current_temperature,
            simulation_count,
            legal_actions,
            legal_actions_policy,
        )
    return argmax_action_policy_fn(
        num_moves,
        model_iteration,
        legal_actions,
        legal_actions_policy,
    )


def softmax_action_policy_fn(temperature, simulation_count, legal_actions, legal_actions_policy):
    '''Softmax action policy'''
    # print(temperature)
    # recover original simulation count
    visit_counts = np.array(legal_actions_policy) * simulation_count
    policy = visit_counts**(1.0 / temperature)
    policy /= np.sum(policy)
    return random.choices(legal_actions, policy)[0]


def argmax_action_policy_fn(num_moves, model_iteration, legal_actions, legal_actions_policy):
    '''Argmax action policy'''
    # print('argmax')
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
    softmax_temperature = config['game_server']['softmax_temperature']
    callbacks = {
        'metric': {},
    }
    # default action policy
    callbacks['action_policy'] = partial(
        make_default_action_policy_fn,
        softmax_temperature,
        config['mcts']['simulation_count'],
    )
    if args.eval:
        callbacks['action_policy'] = argmax_action_policy_fn
        # np.set_printoptions(precision=3)
        # board_shape = config['game']['observation_shape'][-2:]
        # callbacks['metric']['after_apply'] = partial(eval_after_apply_metric, board_shape)
    try:
        asyncio.run(main(args, config, callbacks))
    except KeyboardInterrupt:
        zmq.asyncio.Context.instance().destroy()

        # send goodbye message to learner
        context = zmq.Context()
        learner_socket = context.socket(zmq.DEALER)
        node = czf_pb2.Node(
            identity=f'game-server-{args.suffix}',
            hostname=platform.node(),
        )
        learner_socket.setsockopt_string(zmq.IDENTITY, node.identity)
        learner_socket.connect(f'tcp://{args.upstream}')
        packet = czf_pb2.Packet(goodbye=czf_pb2.Heartbeat())
        learner_socket.send(packet.SerializeToString())

        # send goodbye message to broker
        broker_socket = context.socket(zmq.DEALER)
        broker_socket.setsockopt_string(zmq.IDENTITY, node.identity)
        broker_socket.connect(f'tcp://{args.broker}')
        packet = czf_pb2.Packet(goodbye=czf_pb2.Heartbeat())
        broker_socket.send(packet.SerializeToString())

        print('\rterminated by ctrl-c:')
        # print identity
        print("Identity: ", node.identity)


if __name__ == '__main__':
    run_main()
