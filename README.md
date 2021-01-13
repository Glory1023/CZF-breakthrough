# czf: CGI Zero Framework

## Install

```shell
pip install .
```

## Run

- Create a training configuration: `game.yaml` (ref: `samples/gomoku.yaml`)
- Run the following commands in different shells:

### Training Mode

```bash=
python -m czf.learner -l 5577 -f game.yaml
python -m czf.broker -l 5566
python -m czf.actor -b 127.0.0.1:5566 -u 127.0.0.1:5577 -f game.yaml -bs 64
python -m czf.game_server -b 127.0.0.1:5566 -u 127.0.0.1:5577 -f game.yaml -np 1 -n 1
```

### Evaluation Mode

```bash=
python -m czf.broker -l 5566
python -m czf.actor -b 127.0.0.1:5566 -u 127.0.0.1:5577 -f game.yaml -bs 64 --eval 1P
python -m czf.actor -b 127.0.0.1:5566 -u 127.0.0.1:5577 -f game.yaml -bs 64 --eval 2P
python -m czf.game_server -b 127.0.0.1:5566 -u 127.0.0.1:5577 -f game.yaml -np 1 -n 1 --eval
```

## Run in a Container Pod

- The container image for runtime: `chengscott/czf:runtime`

Suppose the current directory is the root directory of this repo, and the repo is built in `$RUNTIME_DIR` (e.g., `RUNTIME_DIR=$PWD/build/lib.linux-x86_64-3.8`)

First, create a pod with ports properly exported:

```bash
podman pod create --name czf -p 5577:5577 -p 6006:6006
```

Then, run each command inside the pod: (each command starts with `podman run --rm -v $RUNTIME_DIR:/czf -w /czf --pod czf -it chengscott/czf:runtime`)

```bash=
podman run --rm -v $RUNTIME_DIR:/czf -w /czf --pod czf -it chengscott/czf:runtime \
    python -m czf.learner -l 5577 -f game.yaml
podman run --rm -v $RUNTIME_DIR:/czf -w /czf --pod czf -it chengscott/czf:runtime \
    python -m czf.broker -l 5566
podman run --rm -v $RUNTIME_DIR:/czf -w /czf --pod czf -it chengscott/czf:runtime \
    python -m czf.actor -b 127.0.0.1:5566 -u 127.0.0.1:5577 -f game.yaml -bs 64
podman run --rm -v $RUNTIME_DIR:/czf -w /czf --pod czf -it chengscott/czf:runtime \
    python -m czf.game_server -b 127.0.0.1:5566 -u 127.0.0.1:5577 -f game.yaml -np 1 -n 1
```

## Development

### Get

```bash
git clone https://github.com/chengscott/czf
git submodule update --init --recursive --remote --depth 1
```

### Build

Debug build:

```shell
python setup.py build -g
```

- Release build: `python setup.py build`
- Build cpp extensions only: `python setup.py build_ext`
- Build docs only: `python setup.py build_docs`
- Build protobuf only: `python setup.py proto`
- Format: `python setup.py format`
- Test: `python setup.py tests`

### Build in a Container

- The container image for development: `chengscott/czf:devel`

Suppose the current directory is the root directory of this repo:

```bash
podman run --rm -v $PWD:/czf -w /czf chengscott/czf:devel python setup.py build -g
```
