'''CZF Trainer'''
import abc
from datetime import datetime
import multiprocessing as mp
from multiprocessing.managers import BaseManager
import os
from queue import Empty
import time
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm


def run_sampler(index_queue, sample_queue, replay_buffer, prefetch_factor):
    '''run sampler'''
    while True:
        index = [index_queue.get()]
        try:
            for _ in range(prefetch_factor):
                index.append(index_queue.get(block=False))
        except Empty:
            pass
        finally:
            for i in index:
                sample_queue.put(replay_buffer[i])


class MyDataLoader:
    '''DataLoader'''
    def __init__(
        self,
        index_queue,
        sample_queue,
        batch_size,
        collate_fn,
    ):
        self._index_queue = index_queue
        self._sample_queue = sample_queue
        self._batch_size = batch_size
        self._collate_fn = collate_fn
        self._num_sample = 0

    def put(self, sampler):
        '''Put all samples from `sampler` into `index_queue`'''
        for index in sampler:
            self._num_sample += 1
            self._index_queue.put(index)

    def __iter__(self):
        while self._num_sample > self._batch_size:
            data = self._collate_fn(
                [self._sample_queue.get() for _ in range(self._batch_size)])
            self._num_sample -= self._batch_size
            yield data
        data = self._collate_fn(
            [self._sample_queue.get() for _ in range(self._num_sample)])
        self._num_sample = 0
        yield data


def run_trainer(
    args,
    config,
    path,
    trajectory_queue,
    notify_model_queue,
    trainer_cls,
    dataloder_collate_fn,
    replay_buffer_cls,
    replay_buffer_exposed_methods,
    replay_buffer_config,
):
    '''run :class:`Trainer`'''
    storage_path, checkpoint_path, model_path, log_path, trajectory_path = path
    checkpoint_freq = config['learner']['checkpoint_freq']
    batch_size = config['learner']['batch_size']

    # replay buffer
    BaseManager.register('ReplayBuffer',
                         replay_buffer_cls,
                         exposed=replay_buffer_exposed_methods)
    manager = BaseManager()
    manager.start()
    replay_buffer = manager.ReplayBuffer(**replay_buffer_config)

    # restore the replay buffer
    if args.restore_buffer_dir:
        pass  # TODO: restore the replay buffer

    # sampler
    index_queue = mp.Queue()
    sample_queue = mp.Queue()
    prefetch_factor = 4
    samplers = [
        mp.Process(
            target=run_sampler,
            args=(
                index_queue,
                sample_queue,
                replay_buffer,
                prefetch_factor,
            ),
        ) for _ in range(args.num_proc)
    ]
    for sampler in samplers:
        sampler.start()

    # trainer
    trainer = trainer_cls(
        config,
        checkpoint_path,
        model_path,
        log_path,
        args.model_name,
        args.restore_checkpoint_path,
    )
    trainer.save_model()
    print('Storage path:', storage_path)

    # pretrain the trajectory
    if args.pretrain_trajectory_dir:
        pass  # TODO: pretrain the trajectory

    # dataloader
    dataloader = MyDataLoader(
        index_queue,
        sample_queue,
        batch_size=batch_size,
        collate_fn=dataloder_collate_fn,
    )

    pbar = tqdm(total=replay_buffer.get_num_to_add(), desc='Collect Trajs')
    # training loop
    while True:
        trajectories = trajectory_queue.get_all()
        progress = 0
        for trajectory in trajectories:
            progress += replay_buffer.add_trajectory(trajectory)
        pbar.update(min(progress, pbar.total - pbar.n))
        states_to_train = replay_buffer.get_states_to_train()
        if states_to_train > 0:
            pbar.close()
            print(f'[{datetime.now().strftime("%Y/%m/%d %H:%M:%S")}] >> '
                  'Start optimization')
            print('>> States to train:', states_to_train)
            start = time.time()
            trainer.log_statistics(replay_buffer)
            replay_buffer.save_trajectory(trajectory_path, trainer.iteration)
            sampler = WeightedRandomSampler(
                replay_buffer.get_weights(),
                states_to_train,
            )
            dataloader.put(sampler)
            trainer.train(dataloader, replay_buffer)
            save_ckpt = (trainer.iteration % checkpoint_freq == 0)
            trainer.save_model(save_ckpt)
            notify_model_queue.put((trainer.model_name, trainer.iteration))
            print(f'[{datetime.now().strftime("%Y/%m/%d %H:%M:%S")}] >> '
                  f'Finish optimization with time {time.time() - start:.3f}')
            pbar = tqdm(
                total=replay_buffer.get_num_to_add(),
                desc='Collect Trajs',
            )


class TrainerRunner:
    '''Manage :class:`Trainer` to run in another process.'''
    def __init__(
        self,
        args,
        config,
        path,
        trajectory_queue,
        trainer_cls,
        dataloder_collate_fn,
        replay_buffer_cls,
        replay_buffer_exposed_methods,
        replay_buffer_config,
    ):
        self._notify_model_queue = mp.Queue()
        self._process = mp.Process(
            target=run_trainer,
            args=(
                args,
                config,
                path,
                trajectory_queue,
                self._notify_model_queue,
                trainer_cls,
                dataloder_collate_fn,
                replay_buffer_cls,
                replay_buffer_exposed_methods,
                replay_buffer_config,
            ),
        )
        self._process.start()

    def get_notify(self):
        '''get notify for a new model'''
        return self._notify_model_queue.get()


class Trainer(abc.ABC):
    @abc.abstractmethod
    def train(self, dataloader, replay_buffer):
        '''distributed training wrapper'''
        raise NotImplementedError('Trainer must be able to train')

    @abc.abstractmethod
    def log_statistics(self, replay_buffer):
        '''Log statistics for recent trajectories'''
        raise NotImplementedError('Trainer must be able to log statistics')

    @abc.abstractmethod
    def save_model(self, checkpoint=False):
        '''save model to file'''
        raise NotImplementedError('Trainer must be able to save model')
