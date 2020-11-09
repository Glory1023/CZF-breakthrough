# czf: CGI Zero Framework

## Install

```shell
pip install .
```

## Examples

### Run Locally

- Create a training configuration: `game.yaml` (ref: `config.example.yaml`)
- Run the following commands in different shells:

```shell
czf-actor -b 127.0.0.1:5566 -u 127.0.0.1:5588 -f game.yaml -bs 64
czf-learner -l 5577 -f game.yaml -s $STORAGE
czf-broker -l 5566
czf-model-provider -l 5588 -s $STORAGE
czf-game-server -b 127.0.0.1:5566 -u 127.0.0.1:5577 -f game.yaml -n 64
```

### Evaluation Mode

- can share the same `czf-broker` and `czf-model-provider` as in the training

```shell
czf-actor -b 127.0.0.1:5566 -u 127.0.0.1:5588 -f game.yaml -bs 64 --eval 1P
czf-actor -b 127.0.0.1:5566 -u 127.0.0.1:5588 -f game.yaml -bs 64 --eval 2P
czf-broker -l 5566
czf-model-provider -l 5588 -s $STORAGE
czf-game-server -b 127.0.0.1:5566 -u 127.0.0.1:5577 -f game.yaml -n 32 --eval
```

## Development

```shell
python setup.py build -g
```

- Format: `python setup.py format`
- Test: `python setup.py tests`

### Protobuf

- protoc 3.12.4
- Compile: `python setup.py proto`
