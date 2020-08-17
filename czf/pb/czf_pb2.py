# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: czf.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='czf.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\tczf.proto\")\n\x08MCTSRoot\x12\r\n\x05value\x18\x01 \x01(\x02\x12\x0e\n\x06policy\x18\x02 \x03(\x02\")\n\nGameOrigin\x12\x0c\n\x04node\x18\x01 \x01(\x0c\x12\r\n\x05index\x18\x02 \x01(\x05\"1\n\x08\x41\x66\x66inity\x12\x15\n\rpreprocessors\x18\x01 \x03(\x0c\x12\x0e\n\x06\x61\x63tors\x18\x02 \x03(\x0c\"$\n\x05Model\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\r\n\x05\x62lobs\x18\x02 \x03(\x0c\"\x0b\n\tHeartbeat\"M\n\nJobRequest\x12\x1a\n\x08job_type\x18\x01 \x01(\x0e\x32\x08.JobType\x12\x0f\n\x07vacancy\x18\x02 \x01(\x05\x12\x12\n\nefficiency\x18\x03 \x01(\x02\"~\n\x11PreprocessRequest\x12 \n\x0bgame_origin\x18\x01 \x01(\x0b\x32\x0b.GameOrigin\x12\x13\n\x0bobservation\x18\x02 \x01(\x0c\x12\x15\n\rlegal_actions\x18\x03 \x03(\x05\x12\x1b\n\x08\x61\x66\x66inity\x18\x04 \x01(\x0b\x32\t.Affinity\"\x8a\x01\n\rSearchRequest\x12 \n\x0bgame_origin\x18\x01 \x01(\x0b\x32\x0b.GameOrigin\x12\r\n\x05state\x18\x02 \x01(\x0c\x12\x15\n\rlegal_actions\x18\x03 \x03(\x05\x12\x14\n\x0cpreprocessor\x18\x04 \x01(\x0c\x12\x1b\n\x08\x61\x66\x66inity\x18\x05 \x01(\x0b\x32\t.Affinity\"v\n\x0eSearchResponse\x12 \n\x0bgame_origin\x18\x01 \x01(\x0b\x32\x0b.GameOrigin\x12\x1d\n\nmcts_roots\x18\x02 \x03(\x0b\x32\t.MCTSRoot\x12\x14\n\x0cpreprocessor\x18\x03 \x01(\x0c\x12\r\n\x05\x61\x63tor\x18\x04 \x01(\x0c\"M\n\nTrajectory\x12\x0f\n\x07\x61\x63tions\x18\x01 \x03(\x05\x12\x0f\n\x07rewards\x18\x02 \x03(\x02\x12\x1d\n\nmcts_roots\x18\x03 \x03(\x0b\x32\t.MCTSRoot\"\x9c\x02\n\x06Packet\x12\x1f\n\theartbeat\x18\x01 \x01(\x0b\x32\n.HeartbeatH\x00\x12\"\n\x0bjob_request\x18\x02 \x01(\x0b\x32\x0b.JobRequestH\x00\x12\x30\n\x12preprocess_request\x18\x03 \x01(\x0b\x32\x12.PreprocessRequestH\x00\x12(\n\x0esearch_request\x18\x04 \x01(\x0b\x32\x0e.SearchRequestH\x00\x12*\n\x0fsearch_response\x18\x05 \x01(\x0b\x32\x0f.SearchResponseH\x00\x12\x17\n\x05model\x18\x06 \x01(\x0b\x32\x06.ModelH\x00\x12!\n\ntrajectory\x18\x07 \x01(\x0b\x32\x0b.TrajectoryH\x00\x42\t\n\x07payload*2\n\x07JobType\x12\x0b\n\x07UNKNOWN\x10\x00\x12\x0e\n\nPREPROCESS\x10\x01\x12\n\n\x06SEARCH\x10\x02\x62\x06proto3'
)

_JOBTYPE = _descriptor.EnumDescriptor(
  name='JobType',
  full_name='JobType',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='UNKNOWN', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='PREPROCESS', index=1, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='SEARCH', index=2, number=2,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1035,
  serialized_end=1085,
)
_sym_db.RegisterEnumDescriptor(_JOBTYPE)

JobType = enum_type_wrapper.EnumTypeWrapper(_JOBTYPE)
UNKNOWN = 0
PREPROCESS = 1
SEARCH = 2



_MCTSROOT = _descriptor.Descriptor(
  name='MCTSRoot',
  full_name='MCTSRoot',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='value', full_name='MCTSRoot.value', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='policy', full_name='MCTSRoot.policy', index=1,
      number=2, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=13,
  serialized_end=54,
)


_GAMEORIGIN = _descriptor.Descriptor(
  name='GameOrigin',
  full_name='GameOrigin',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='node', full_name='GameOrigin.node', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='index', full_name='GameOrigin.index', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=56,
  serialized_end=97,
)


_AFFINITY = _descriptor.Descriptor(
  name='Affinity',
  full_name='Affinity',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='preprocessors', full_name='Affinity.preprocessors', index=0,
      number=1, type=12, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='actors', full_name='Affinity.actors', index=1,
      number=2, type=12, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=99,
  serialized_end=148,
)


_MODEL = _descriptor.Descriptor(
  name='Model',
  full_name='Model',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='Model.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='blobs', full_name='Model.blobs', index=1,
      number=2, type=12, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=150,
  serialized_end=186,
)


_HEARTBEAT = _descriptor.Descriptor(
  name='Heartbeat',
  full_name='Heartbeat',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
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
  serialized_start=188,
  serialized_end=199,
)


_JOBREQUEST = _descriptor.Descriptor(
  name='JobRequest',
  full_name='JobRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='job_type', full_name='JobRequest.job_type', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='vacancy', full_name='JobRequest.vacancy', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='efficiency', full_name='JobRequest.efficiency', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=201,
  serialized_end=278,
)


_PREPROCESSREQUEST = _descriptor.Descriptor(
  name='PreprocessRequest',
  full_name='PreprocessRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='game_origin', full_name='PreprocessRequest.game_origin', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='observation', full_name='PreprocessRequest.observation', index=1,
      number=2, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='legal_actions', full_name='PreprocessRequest.legal_actions', index=2,
      number=3, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='affinity', full_name='PreprocessRequest.affinity', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=280,
  serialized_end=406,
)


_SEARCHREQUEST = _descriptor.Descriptor(
  name='SearchRequest',
  full_name='SearchRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='game_origin', full_name='SearchRequest.game_origin', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='state', full_name='SearchRequest.state', index=1,
      number=2, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='legal_actions', full_name='SearchRequest.legal_actions', index=2,
      number=3, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='preprocessor', full_name='SearchRequest.preprocessor', index=3,
      number=4, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='affinity', full_name='SearchRequest.affinity', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=409,
  serialized_end=547,
)


_SEARCHRESPONSE = _descriptor.Descriptor(
  name='SearchResponse',
  full_name='SearchResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='game_origin', full_name='SearchResponse.game_origin', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='mcts_roots', full_name='SearchResponse.mcts_roots', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='preprocessor', full_name='SearchResponse.preprocessor', index=2,
      number=3, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='actor', full_name='SearchResponse.actor', index=3,
      number=4, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=549,
  serialized_end=667,
)


_TRAJECTORY = _descriptor.Descriptor(
  name='Trajectory',
  full_name='Trajectory',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='actions', full_name='Trajectory.actions', index=0,
      number=1, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='rewards', full_name='Trajectory.rewards', index=1,
      number=2, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='mcts_roots', full_name='Trajectory.mcts_roots', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
  serialized_start=669,
  serialized_end=746,
)


_PACKET = _descriptor.Descriptor(
  name='Packet',
  full_name='Packet',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='heartbeat', full_name='Packet.heartbeat', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='job_request', full_name='Packet.job_request', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='preprocess_request', full_name='Packet.preprocess_request', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='search_request', full_name='Packet.search_request', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='search_response', full_name='Packet.search_response', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='model', full_name='Packet.model', index=5,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='trajectory', full_name='Packet.trajectory', index=6,
      number=7, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
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
      name='payload', full_name='Packet.payload',
      index=0, containing_type=None,
      create_key=_descriptor._internal_create_key,
    fields=[]),
  ],
  serialized_start=749,
  serialized_end=1033,
)

_JOBREQUEST.fields_by_name['job_type'].enum_type = _JOBTYPE
_PREPROCESSREQUEST.fields_by_name['game_origin'].message_type = _GAMEORIGIN
_PREPROCESSREQUEST.fields_by_name['affinity'].message_type = _AFFINITY
_SEARCHREQUEST.fields_by_name['game_origin'].message_type = _GAMEORIGIN
_SEARCHREQUEST.fields_by_name['affinity'].message_type = _AFFINITY
_SEARCHRESPONSE.fields_by_name['game_origin'].message_type = _GAMEORIGIN
_SEARCHRESPONSE.fields_by_name['mcts_roots'].message_type = _MCTSROOT
_TRAJECTORY.fields_by_name['mcts_roots'].message_type = _MCTSROOT
_PACKET.fields_by_name['heartbeat'].message_type = _HEARTBEAT
_PACKET.fields_by_name['job_request'].message_type = _JOBREQUEST
_PACKET.fields_by_name['preprocess_request'].message_type = _PREPROCESSREQUEST
_PACKET.fields_by_name['search_request'].message_type = _SEARCHREQUEST
_PACKET.fields_by_name['search_response'].message_type = _SEARCHRESPONSE
_PACKET.fields_by_name['model'].message_type = _MODEL
_PACKET.fields_by_name['trajectory'].message_type = _TRAJECTORY
_PACKET.oneofs_by_name['payload'].fields.append(
  _PACKET.fields_by_name['heartbeat'])
_PACKET.fields_by_name['heartbeat'].containing_oneof = _PACKET.oneofs_by_name['payload']
_PACKET.oneofs_by_name['payload'].fields.append(
  _PACKET.fields_by_name['job_request'])
_PACKET.fields_by_name['job_request'].containing_oneof = _PACKET.oneofs_by_name['payload']
_PACKET.oneofs_by_name['payload'].fields.append(
  _PACKET.fields_by_name['preprocess_request'])
_PACKET.fields_by_name['preprocess_request'].containing_oneof = _PACKET.oneofs_by_name['payload']
_PACKET.oneofs_by_name['payload'].fields.append(
  _PACKET.fields_by_name['search_request'])
_PACKET.fields_by_name['search_request'].containing_oneof = _PACKET.oneofs_by_name['payload']
_PACKET.oneofs_by_name['payload'].fields.append(
  _PACKET.fields_by_name['search_response'])
_PACKET.fields_by_name['search_response'].containing_oneof = _PACKET.oneofs_by_name['payload']
_PACKET.oneofs_by_name['payload'].fields.append(
  _PACKET.fields_by_name['model'])
_PACKET.fields_by_name['model'].containing_oneof = _PACKET.oneofs_by_name['payload']
_PACKET.oneofs_by_name['payload'].fields.append(
  _PACKET.fields_by_name['trajectory'])
_PACKET.fields_by_name['trajectory'].containing_oneof = _PACKET.oneofs_by_name['payload']
DESCRIPTOR.message_types_by_name['MCTSRoot'] = _MCTSROOT
DESCRIPTOR.message_types_by_name['GameOrigin'] = _GAMEORIGIN
DESCRIPTOR.message_types_by_name['Affinity'] = _AFFINITY
DESCRIPTOR.message_types_by_name['Model'] = _MODEL
DESCRIPTOR.message_types_by_name['Heartbeat'] = _HEARTBEAT
DESCRIPTOR.message_types_by_name['JobRequest'] = _JOBREQUEST
DESCRIPTOR.message_types_by_name['PreprocessRequest'] = _PREPROCESSREQUEST
DESCRIPTOR.message_types_by_name['SearchRequest'] = _SEARCHREQUEST
DESCRIPTOR.message_types_by_name['SearchResponse'] = _SEARCHRESPONSE
DESCRIPTOR.message_types_by_name['Trajectory'] = _TRAJECTORY
DESCRIPTOR.message_types_by_name['Packet'] = _PACKET
DESCRIPTOR.enum_types_by_name['JobType'] = _JOBTYPE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

MCTSRoot = _reflection.GeneratedProtocolMessageType('MCTSRoot', (_message.Message,), {
  'DESCRIPTOR' : _MCTSROOT,
  '__module__' : 'czf_pb2'
  # @@protoc_insertion_point(class_scope:MCTSRoot)
  })
_sym_db.RegisterMessage(MCTSRoot)

GameOrigin = _reflection.GeneratedProtocolMessageType('GameOrigin', (_message.Message,), {
  'DESCRIPTOR' : _GAMEORIGIN,
  '__module__' : 'czf_pb2'
  # @@protoc_insertion_point(class_scope:GameOrigin)
  })
_sym_db.RegisterMessage(GameOrigin)

Affinity = _reflection.GeneratedProtocolMessageType('Affinity', (_message.Message,), {
  'DESCRIPTOR' : _AFFINITY,
  '__module__' : 'czf_pb2'
  # @@protoc_insertion_point(class_scope:Affinity)
  })
_sym_db.RegisterMessage(Affinity)

Model = _reflection.GeneratedProtocolMessageType('Model', (_message.Message,), {
  'DESCRIPTOR' : _MODEL,
  '__module__' : 'czf_pb2'
  # @@protoc_insertion_point(class_scope:Model)
  })
_sym_db.RegisterMessage(Model)

Heartbeat = _reflection.GeneratedProtocolMessageType('Heartbeat', (_message.Message,), {
  'DESCRIPTOR' : _HEARTBEAT,
  '__module__' : 'czf_pb2'
  # @@protoc_insertion_point(class_scope:Heartbeat)
  })
_sym_db.RegisterMessage(Heartbeat)

JobRequest = _reflection.GeneratedProtocolMessageType('JobRequest', (_message.Message,), {
  'DESCRIPTOR' : _JOBREQUEST,
  '__module__' : 'czf_pb2'
  # @@protoc_insertion_point(class_scope:JobRequest)
  })
_sym_db.RegisterMessage(JobRequest)

PreprocessRequest = _reflection.GeneratedProtocolMessageType('PreprocessRequest', (_message.Message,), {
  'DESCRIPTOR' : _PREPROCESSREQUEST,
  '__module__' : 'czf_pb2'
  # @@protoc_insertion_point(class_scope:PreprocessRequest)
  })
_sym_db.RegisterMessage(PreprocessRequest)

SearchRequest = _reflection.GeneratedProtocolMessageType('SearchRequest', (_message.Message,), {
  'DESCRIPTOR' : _SEARCHREQUEST,
  '__module__' : 'czf_pb2'
  # @@protoc_insertion_point(class_scope:SearchRequest)
  })
_sym_db.RegisterMessage(SearchRequest)

SearchResponse = _reflection.GeneratedProtocolMessageType('SearchResponse', (_message.Message,), {
  'DESCRIPTOR' : _SEARCHRESPONSE,
  '__module__' : 'czf_pb2'
  # @@protoc_insertion_point(class_scope:SearchResponse)
  })
_sym_db.RegisterMessage(SearchResponse)

Trajectory = _reflection.GeneratedProtocolMessageType('Trajectory', (_message.Message,), {
  'DESCRIPTOR' : _TRAJECTORY,
  '__module__' : 'czf_pb2'
  # @@protoc_insertion_point(class_scope:Trajectory)
  })
_sym_db.RegisterMessage(Trajectory)

Packet = _reflection.GeneratedProtocolMessageType('Packet', (_message.Message,), {
  'DESCRIPTOR' : _PACKET,
  '__module__' : 'czf_pb2'
  # @@protoc_insertion_point(class_scope:Packet)
  })
_sym_db.RegisterMessage(Packet)


# @@protoc_insertion_point(module_scope)
