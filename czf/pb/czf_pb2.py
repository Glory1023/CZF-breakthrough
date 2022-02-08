# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: czf.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='czf.proto',
  package='czf.pb',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\tczf.proto\x12\x06\x63zf.pb\"\x0b\n\tHeartbeat\"*\n\x04Node\x12\x10\n\x08identity\x18\x01 \x01(\t\x12\x10\n\x08hostname\x18\x02 \x01(\t\"*\n\tModelInfo\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0f\n\x07version\x18\x02 \x01(\x05\"7\n\x05Model\x12\x1f\n\x04info\x18\x01 \x01(\x0b\x32\x11.czf.pb.ModelInfo\x12\r\n\x05\x62lobs\x18\x02 \x03(\x0c\"\x9a\x04\n\x0bWorkerState\x12\x15\n\rlegal_actions\x18\x01 \x03(\x05\x12\x1a\n\x12observation_tensor\x18\x02 \x03(\x02\x12\x33\n\x0btree_option\x18\x03 \x01(\x0b\x32\x1e.czf.pb.WorkerState.TreeOption\x12\x32\n\nevaluation\x18\x04 \x01(\x0b\x32\x1e.czf.pb.WorkerState.Evaluation\x12\x32\n\ntransition\x18\x05 \x01(\x0b\x32\x1e.czf.pb.WorkerState.Transition\x12\x18\n\x10serialized_state\x18\x06 \x01(\t\x1a\xac\x01\n\nTreeOption\x12\x18\n\x10simulation_count\x18\x01 \x01(\x05\x12\x16\n\x0etree_min_value\x18\x02 \x01(\x02\x12\x16\n\x0etree_max_value\x18\x03 \x01(\x02\x12\x0e\n\x06\x63_puct\x18\x04 \x01(\x02\x12\x17\n\x0f\x64irichlet_alpha\x18\x05 \x01(\x02\x12\x19\n\x11\x64irichlet_epsilon\x18\x06 \x01(\x02\x12\x10\n\x08\x64iscount\x18\x07 \x01(\x02\x1a+\n\nEvaluation\x12\r\n\x05value\x18\x01 \x01(\x02\x12\x0e\n\x06policy\x18\x02 \x03(\x02\x1a\x45\n\nTransition\x12\x16\n\x0e\x63urrent_player\x18\x01 \x01(\x05\x12\x0e\n\x06\x61\x63tion\x18\x02 \x01(\x05\x12\x0f\n\x07rewards\x18\x03 \x03(\x02\"\xc3\x03\n\x03Job\x12\x10\n\x08identity\x18\x01 \x01(\t\x12\x1f\n\tinitiator\x18\x02 \x01(\x0b\x32\x0c.czf.pb.Node\x12 \n\x05model\x18\x03 \x01(\x0b\x32\x11.czf.pb.ModelInfo\x12(\n\tprocedure\x18\x04 \x03(\x0e\x32\x15.czf.pb.Job.Operation\x12\x0c\n\x04step\x18\x05 \x01(\x05\x12\x1d\n\x07workers\x18\x06 \x03(\x0b\x32\x0c.czf.pb.Node\x12$\n\x07payload\x18\x07 \x01(\x0b\x32\x13.czf.pb.Job.Payload\x1a@\n\x07Payload\x12\x11\n\tenv_index\x18\x01 \x01(\x05\x12\"\n\x05state\x18\x02 \x01(\x0b\x32\x13.czf.pb.WorkerState\"\xa7\x01\n\tOperation\x12\x0b\n\x07UNKNOWN\x10\x00\x12\x14\n\x10\x41LPHAZERO_SEARCH\x10\x01\x12\x19\n\x15\x41LPHAZERO_EVALUATE_1P\x10\x02\x12\x19\n\x15\x41LPHAZERO_EVALUATE_2P\x10\x03\x12\x11\n\rMUZERO_SEARCH\x10\x04\x12\x16\n\x12MUZERO_EVALUATE_1P\x10\x05\x12\x16\n\x12MUZERO_EVALUATE_2P\x10\x06\"H\n\nJobRequest\x12(\n\toperation\x18\x01 \x01(\x0e\x32\x15.czf.pb.Job.Operation\x12\x10\n\x08\x63\x61pacity\x18\x02 \x01(\x05\"%\n\x08JobBatch\x12\x19\n\x04jobs\x18\x01 \x03(\x0b\x32\x0b.czf.pb.Job\"\x97\x01\n\nTrajectory\x12#\n\x06states\x18\x01 \x03(\x0b\x32\x13.czf.pb.WorkerState\x12\x31\n\nstatistics\x18\x02 \x01(\x0b\x32\x1d.czf.pb.Trajectory.Statistics\x1a\x31\n\nStatistics\x12\x0f\n\x07rewards\x18\x01 \x03(\x02\x12\x12\n\ngame_steps\x18\x02 \x01(\x05\";\n\x0fTrajectoryBatch\x12(\n\x0ctrajectories\x18\x01 \x03(\x0b\x32\x12.czf.pb.Trajectory\"\xb0\x03\n\x06Packet\x12&\n\theartbeat\x18\x01 \x01(\x0b\x32\x11.czf.pb.HeartbeatH\x00\x12$\n\x07goodbye\x18\x02 \x01(\x0b\x32\x11.czf.pb.HeartbeatH\x00\x12,\n\x0fmodel_subscribe\x18\x03 \x01(\x0b\x32\x11.czf.pb.HeartbeatH\x00\x12\'\n\nmodel_info\x18\x04 \x01(\x0b\x32\x11.czf.pb.ModelInfoH\x00\x12*\n\rmodel_request\x18\x05 \x01(\x0b\x32\x11.czf.pb.ModelInfoH\x00\x12\'\n\x0emodel_response\x18\x06 \x01(\x0b\x32\r.czf.pb.ModelH\x00\x12)\n\x0bjob_request\x18\x07 \x01(\x0b\x32\x12.czf.pb.JobRequestH\x00\x12%\n\tjob_batch\x18\x08 \x01(\x0b\x32\x10.czf.pb.JobBatchH\x00\x12\x1a\n\x03job\x18\t \x01(\x0b\x32\x0b.czf.pb.JobH\x00\x12\x33\n\x10trajectory_batch\x18\n \x01(\x0b\x32\x17.czf.pb.TrajectoryBatchH\x00\x42\t\n\x07payloadb\x06proto3')
)



_JOB_OPERATION = _descriptor.EnumDescriptor(
  name='Operation',
  full_name='czf.pb.Job.Operation',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='UNKNOWN', index=0, number=0,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ALPHAZERO_SEARCH', index=1, number=1,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ALPHAZERO_EVALUATE_1P', index=2, number=2,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ALPHAZERO_EVALUATE_2P', index=3, number=3,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MUZERO_SEARCH', index=4, number=4,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MUZERO_EVALUATE_1P', index=5, number=5,
      serialized_options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='MUZERO_EVALUATE_2P', index=6, number=6,
      serialized_options=None,
      type=None),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1005,
  serialized_end=1172,
)
_sym_db.RegisterEnumDescriptor(_JOB_OPERATION)


_HEARTBEAT = _descriptor.Descriptor(
  name='Heartbeat',
  full_name='czf.pb.Heartbeat',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=21,
  serialized_end=32,
)


_NODE = _descriptor.Descriptor(
  name='Node',
  full_name='czf.pb.Node',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='identity', full_name='czf.pb.Node.identity', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='hostname', full_name='czf.pb.Node.hostname', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=34,
  serialized_end=76,
)


_MODELINFO = _descriptor.Descriptor(
  name='ModelInfo',
  full_name='czf.pb.ModelInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='czf.pb.ModelInfo.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='version', full_name='czf.pb.ModelInfo.version', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=78,
  serialized_end=120,
)


_MODEL = _descriptor.Descriptor(
  name='Model',
  full_name='czf.pb.Model',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='info', full_name='czf.pb.Model.info', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='blobs', full_name='czf.pb.Model.blobs', index=1,
      number=2, type=12, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=122,
  serialized_end=177,
)


_WORKERSTATE_TREEOPTION = _descriptor.Descriptor(
  name='TreeOption',
  full_name='czf.pb.WorkerState.TreeOption',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='simulation_count', full_name='czf.pb.WorkerState.TreeOption.simulation_count', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='tree_min_value', full_name='czf.pb.WorkerState.TreeOption.tree_min_value', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='tree_max_value', full_name='czf.pb.WorkerState.TreeOption.tree_max_value', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='c_puct', full_name='czf.pb.WorkerState.TreeOption.c_puct', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dirichlet_alpha', full_name='czf.pb.WorkerState.TreeOption.dirichlet_alpha', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='dirichlet_epsilon', full_name='czf.pb.WorkerState.TreeOption.dirichlet_epsilon', index=5,
      number=6, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='discount', full_name='czf.pb.WorkerState.TreeOption.discount', index=6,
      number=7, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=430,
  serialized_end=602,
)

_WORKERSTATE_EVALUATION = _descriptor.Descriptor(
  name='Evaluation',
  full_name='czf.pb.WorkerState.Evaluation',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='value', full_name='czf.pb.WorkerState.Evaluation.value', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='policy', full_name='czf.pb.WorkerState.Evaluation.policy', index=1,
      number=2, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=604,
  serialized_end=647,
)

_WORKERSTATE_TRANSITION = _descriptor.Descriptor(
  name='Transition',
  full_name='czf.pb.WorkerState.Transition',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='current_player', full_name='czf.pb.WorkerState.Transition.current_player', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='action', full_name='czf.pb.WorkerState.Transition.action', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='rewards', full_name='czf.pb.WorkerState.Transition.rewards', index=2,
      number=3, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=649,
  serialized_end=718,
)

_WORKERSTATE = _descriptor.Descriptor(
  name='WorkerState',
  full_name='czf.pb.WorkerState',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='legal_actions', full_name='czf.pb.WorkerState.legal_actions', index=0,
      number=1, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='observation_tensor', full_name='czf.pb.WorkerState.observation_tensor', index=1,
      number=2, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='tree_option', full_name='czf.pb.WorkerState.tree_option', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='evaluation', full_name='czf.pb.WorkerState.evaluation', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='transition', full_name='czf.pb.WorkerState.transition', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='serialized_state', full_name='czf.pb.WorkerState.serialized_state', index=5,
      number=6, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_WORKERSTATE_TREEOPTION, _WORKERSTATE_EVALUATION, _WORKERSTATE_TRANSITION, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=180,
  serialized_end=718,
)


_JOB_PAYLOAD = _descriptor.Descriptor(
  name='Payload',
  full_name='czf.pb.Job.Payload',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='env_index', full_name='czf.pb.Job.Payload.env_index', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='state', full_name='czf.pb.Job.Payload.state', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=938,
  serialized_end=1002,
)

_JOB = _descriptor.Descriptor(
  name='Job',
  full_name='czf.pb.Job',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='identity', full_name='czf.pb.Job.identity', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='initiator', full_name='czf.pb.Job.initiator', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='model', full_name='czf.pb.Job.model', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='procedure', full_name='czf.pb.Job.procedure', index=3,
      number=4, type=14, cpp_type=8, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='step', full_name='czf.pb.Job.step', index=4,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='workers', full_name='czf.pb.Job.workers', index=5,
      number=6, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='payload', full_name='czf.pb.Job.payload', index=6,
      number=7, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_JOB_PAYLOAD, ],
  enum_types=[
    _JOB_OPERATION,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=721,
  serialized_end=1172,
)


_JOBREQUEST = _descriptor.Descriptor(
  name='JobRequest',
  full_name='czf.pb.JobRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='operation', full_name='czf.pb.JobRequest.operation', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='capacity', full_name='czf.pb.JobRequest.capacity', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1174,
  serialized_end=1246,
)


_JOBBATCH = _descriptor.Descriptor(
  name='JobBatch',
  full_name='czf.pb.JobBatch',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='jobs', full_name='czf.pb.JobBatch.jobs', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1248,
  serialized_end=1285,
)


_TRAJECTORY_STATISTICS = _descriptor.Descriptor(
  name='Statistics',
  full_name='czf.pb.Trajectory.Statistics',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='rewards', full_name='czf.pb.Trajectory.Statistics.rewards', index=0,
      number=1, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='game_steps', full_name='czf.pb.Trajectory.Statistics.game_steps', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1390,
  serialized_end=1439,
)

_TRAJECTORY = _descriptor.Descriptor(
  name='Trajectory',
  full_name='czf.pb.Trajectory',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='states', full_name='czf.pb.Trajectory.states', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='statistics', full_name='czf.pb.Trajectory.statistics', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[_TRAJECTORY_STATISTICS, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1288,
  serialized_end=1439,
)


_TRAJECTORYBATCH = _descriptor.Descriptor(
  name='TrajectoryBatch',
  full_name='czf.pb.TrajectoryBatch',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='trajectories', full_name='czf.pb.TrajectoryBatch.trajectories', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1441,
  serialized_end=1500,
)


_PACKET = _descriptor.Descriptor(
  name='Packet',
  full_name='czf.pb.Packet',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='heartbeat', full_name='czf.pb.Packet.heartbeat', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='goodbye', full_name='czf.pb.Packet.goodbye', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='model_subscribe', full_name='czf.pb.Packet.model_subscribe', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='model_info', full_name='czf.pb.Packet.model_info', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='model_request', full_name='czf.pb.Packet.model_request', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='model_response', full_name='czf.pb.Packet.model_response', index=5,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='job_request', full_name='czf.pb.Packet.job_request', index=6,
      number=7, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='job_batch', full_name='czf.pb.Packet.job_batch', index=7,
      number=8, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='job', full_name='czf.pb.Packet.job', index=8,
      number=9, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='trajectory_batch', full_name='czf.pb.Packet.trajectory_batch', index=9,
      number=10, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='payload', full_name='czf.pb.Packet.payload',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=1503,
  serialized_end=1935,
)

_MODEL.fields_by_name['info'].message_type = _MODELINFO
_WORKERSTATE_TREEOPTION.containing_type = _WORKERSTATE
_WORKERSTATE_EVALUATION.containing_type = _WORKERSTATE
_WORKERSTATE_TRANSITION.containing_type = _WORKERSTATE
_WORKERSTATE.fields_by_name['tree_option'].message_type = _WORKERSTATE_TREEOPTION
_WORKERSTATE.fields_by_name['evaluation'].message_type = _WORKERSTATE_EVALUATION
_WORKERSTATE.fields_by_name['transition'].message_type = _WORKERSTATE_TRANSITION
_JOB_PAYLOAD.fields_by_name['state'].message_type = _WORKERSTATE
_JOB_PAYLOAD.containing_type = _JOB
_JOB.fields_by_name['initiator'].message_type = _NODE
_JOB.fields_by_name['model'].message_type = _MODELINFO
_JOB.fields_by_name['procedure'].enum_type = _JOB_OPERATION
_JOB.fields_by_name['workers'].message_type = _NODE
_JOB.fields_by_name['payload'].message_type = _JOB_PAYLOAD
_JOB_OPERATION.containing_type = _JOB
_JOBREQUEST.fields_by_name['operation'].enum_type = _JOB_OPERATION
_JOBBATCH.fields_by_name['jobs'].message_type = _JOB
_TRAJECTORY_STATISTICS.containing_type = _TRAJECTORY
_TRAJECTORY.fields_by_name['states'].message_type = _WORKERSTATE
_TRAJECTORY.fields_by_name['statistics'].message_type = _TRAJECTORY_STATISTICS
_TRAJECTORYBATCH.fields_by_name['trajectories'].message_type = _TRAJECTORY
_PACKET.fields_by_name['heartbeat'].message_type = _HEARTBEAT
_PACKET.fields_by_name['goodbye'].message_type = _HEARTBEAT
_PACKET.fields_by_name['model_subscribe'].message_type = _HEARTBEAT
_PACKET.fields_by_name['model_info'].message_type = _MODELINFO
_PACKET.fields_by_name['model_request'].message_type = _MODELINFO
_PACKET.fields_by_name['model_response'].message_type = _MODEL
_PACKET.fields_by_name['job_request'].message_type = _JOBREQUEST
_PACKET.fields_by_name['job_batch'].message_type = _JOBBATCH
_PACKET.fields_by_name['job'].message_type = _JOB
_PACKET.fields_by_name['trajectory_batch'].message_type = _TRAJECTORYBATCH
_PACKET.oneofs_by_name['payload'].fields.append(
  _PACKET.fields_by_name['heartbeat'])
_PACKET.fields_by_name['heartbeat'].containing_oneof = _PACKET.oneofs_by_name['payload']
_PACKET.oneofs_by_name['payload'].fields.append(
  _PACKET.fields_by_name['goodbye'])
_PACKET.fields_by_name['goodbye'].containing_oneof = _PACKET.oneofs_by_name['payload']
_PACKET.oneofs_by_name['payload'].fields.append(
  _PACKET.fields_by_name['model_subscribe'])
_PACKET.fields_by_name['model_subscribe'].containing_oneof = _PACKET.oneofs_by_name['payload']
_PACKET.oneofs_by_name['payload'].fields.append(
  _PACKET.fields_by_name['model_info'])
_PACKET.fields_by_name['model_info'].containing_oneof = _PACKET.oneofs_by_name['payload']
_PACKET.oneofs_by_name['payload'].fields.append(
  _PACKET.fields_by_name['model_request'])
_PACKET.fields_by_name['model_request'].containing_oneof = _PACKET.oneofs_by_name['payload']
_PACKET.oneofs_by_name['payload'].fields.append(
  _PACKET.fields_by_name['model_response'])
_PACKET.fields_by_name['model_response'].containing_oneof = _PACKET.oneofs_by_name['payload']
_PACKET.oneofs_by_name['payload'].fields.append(
  _PACKET.fields_by_name['job_request'])
_PACKET.fields_by_name['job_request'].containing_oneof = _PACKET.oneofs_by_name['payload']
_PACKET.oneofs_by_name['payload'].fields.append(
  _PACKET.fields_by_name['job_batch'])
_PACKET.fields_by_name['job_batch'].containing_oneof = _PACKET.oneofs_by_name['payload']
_PACKET.oneofs_by_name['payload'].fields.append(
  _PACKET.fields_by_name['job'])
_PACKET.fields_by_name['job'].containing_oneof = _PACKET.oneofs_by_name['payload']
_PACKET.oneofs_by_name['payload'].fields.append(
  _PACKET.fields_by_name['trajectory_batch'])
_PACKET.fields_by_name['trajectory_batch'].containing_oneof = _PACKET.oneofs_by_name['payload']
DESCRIPTOR.message_types_by_name['Heartbeat'] = _HEARTBEAT
DESCRIPTOR.message_types_by_name['Node'] = _NODE
DESCRIPTOR.message_types_by_name['ModelInfo'] = _MODELINFO
DESCRIPTOR.message_types_by_name['Model'] = _MODEL
DESCRIPTOR.message_types_by_name['WorkerState'] = _WORKERSTATE
DESCRIPTOR.message_types_by_name['Job'] = _JOB
DESCRIPTOR.message_types_by_name['JobRequest'] = _JOBREQUEST
DESCRIPTOR.message_types_by_name['JobBatch'] = _JOBBATCH
DESCRIPTOR.message_types_by_name['Trajectory'] = _TRAJECTORY
DESCRIPTOR.message_types_by_name['TrajectoryBatch'] = _TRAJECTORYBATCH
DESCRIPTOR.message_types_by_name['Packet'] = _PACKET
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Heartbeat = _reflection.GeneratedProtocolMessageType('Heartbeat', (_message.Message,), dict(
  DESCRIPTOR = _HEARTBEAT,
  __module__ = 'czf_pb2'
  # @@protoc_insertion_point(class_scope:czf.pb.Heartbeat)
  ))
_sym_db.RegisterMessage(Heartbeat)

Node = _reflection.GeneratedProtocolMessageType('Node', (_message.Message,), dict(
  DESCRIPTOR = _NODE,
  __module__ = 'czf_pb2'
  # @@protoc_insertion_point(class_scope:czf.pb.Node)
  ))
_sym_db.RegisterMessage(Node)

ModelInfo = _reflection.GeneratedProtocolMessageType('ModelInfo', (_message.Message,), dict(
  DESCRIPTOR = _MODELINFO,
  __module__ = 'czf_pb2'
  # @@protoc_insertion_point(class_scope:czf.pb.ModelInfo)
  ))
_sym_db.RegisterMessage(ModelInfo)

Model = _reflection.GeneratedProtocolMessageType('Model', (_message.Message,), dict(
  DESCRIPTOR = _MODEL,
  __module__ = 'czf_pb2'
  # @@protoc_insertion_point(class_scope:czf.pb.Model)
  ))
_sym_db.RegisterMessage(Model)

WorkerState = _reflection.GeneratedProtocolMessageType('WorkerState', (_message.Message,), dict(

  TreeOption = _reflection.GeneratedProtocolMessageType('TreeOption', (_message.Message,), dict(
    DESCRIPTOR = _WORKERSTATE_TREEOPTION,
    __module__ = 'czf_pb2'
    # @@protoc_insertion_point(class_scope:czf.pb.WorkerState.TreeOption)
    ))
  ,

  Evaluation = _reflection.GeneratedProtocolMessageType('Evaluation', (_message.Message,), dict(
    DESCRIPTOR = _WORKERSTATE_EVALUATION,
    __module__ = 'czf_pb2'
    # @@protoc_insertion_point(class_scope:czf.pb.WorkerState.Evaluation)
    ))
  ,

  Transition = _reflection.GeneratedProtocolMessageType('Transition', (_message.Message,), dict(
    DESCRIPTOR = _WORKERSTATE_TRANSITION,
    __module__ = 'czf_pb2'
    # @@protoc_insertion_point(class_scope:czf.pb.WorkerState.Transition)
    ))
  ,
  DESCRIPTOR = _WORKERSTATE,
  __module__ = 'czf_pb2'
  # @@protoc_insertion_point(class_scope:czf.pb.WorkerState)
  ))
_sym_db.RegisterMessage(WorkerState)
_sym_db.RegisterMessage(WorkerState.TreeOption)
_sym_db.RegisterMessage(WorkerState.Evaluation)
_sym_db.RegisterMessage(WorkerState.Transition)

Job = _reflection.GeneratedProtocolMessageType('Job', (_message.Message,), dict(

  Payload = _reflection.GeneratedProtocolMessageType('Payload', (_message.Message,), dict(
    DESCRIPTOR = _JOB_PAYLOAD,
    __module__ = 'czf_pb2'
    # @@protoc_insertion_point(class_scope:czf.pb.Job.Payload)
    ))
  ,
  DESCRIPTOR = _JOB,
  __module__ = 'czf_pb2'
  # @@protoc_insertion_point(class_scope:czf.pb.Job)
  ))
_sym_db.RegisterMessage(Job)
_sym_db.RegisterMessage(Job.Payload)

JobRequest = _reflection.GeneratedProtocolMessageType('JobRequest', (_message.Message,), dict(
  DESCRIPTOR = _JOBREQUEST,
  __module__ = 'czf_pb2'
  # @@protoc_insertion_point(class_scope:czf.pb.JobRequest)
  ))
_sym_db.RegisterMessage(JobRequest)

JobBatch = _reflection.GeneratedProtocolMessageType('JobBatch', (_message.Message,), dict(
  DESCRIPTOR = _JOBBATCH,
  __module__ = 'czf_pb2'
  # @@protoc_insertion_point(class_scope:czf.pb.JobBatch)
  ))
_sym_db.RegisterMessage(JobBatch)

Trajectory = _reflection.GeneratedProtocolMessageType('Trajectory', (_message.Message,), dict(

  Statistics = _reflection.GeneratedProtocolMessageType('Statistics', (_message.Message,), dict(
    DESCRIPTOR = _TRAJECTORY_STATISTICS,
    __module__ = 'czf_pb2'
    # @@protoc_insertion_point(class_scope:czf.pb.Trajectory.Statistics)
    ))
  ,
  DESCRIPTOR = _TRAJECTORY,
  __module__ = 'czf_pb2'
  # @@protoc_insertion_point(class_scope:czf.pb.Trajectory)
  ))
_sym_db.RegisterMessage(Trajectory)
_sym_db.RegisterMessage(Trajectory.Statistics)

TrajectoryBatch = _reflection.GeneratedProtocolMessageType('TrajectoryBatch', (_message.Message,), dict(
  DESCRIPTOR = _TRAJECTORYBATCH,
  __module__ = 'czf_pb2'
  # @@protoc_insertion_point(class_scope:czf.pb.TrajectoryBatch)
  ))
_sym_db.RegisterMessage(TrajectoryBatch)

Packet = _reflection.GeneratedProtocolMessageType('Packet', (_message.Message,), dict(
  DESCRIPTOR = _PACKET,
  __module__ = 'czf_pb2'
  # @@protoc_insertion_point(class_scope:czf.pb.Packet)
  ))
_sym_db.RegisterMessage(Packet)


# @@protoc_insertion_point(module_scope)
