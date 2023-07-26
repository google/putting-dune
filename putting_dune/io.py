# Copyright 2024 The Putting Dune Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""IO Utilities."""
import contextlib
import os
import typing
from typing import Iterable, Iterator, Literal, Type, TypeVar

from etils import epath
from putting_dune import microscope_utils
import tensorflow as tf


ProtoModelT = TypeVar("ProtoModelT", bound=microscope_utils.ProtoModel)

PathLike = epath.Path | str | os.PathLike[str]


@typing.overload
def read_records(
    file: PathLike, record_type: Type[ProtoModelT]
) -> Iterator[ProtoModelT]:
  ...


@typing.overload
def read_records(
    file: PathLike, record_type: Literal[None] = None
) -> Iterator[str]:
  ...


def read_records(
    file: PathLike,
    record_type: Type[ProtoModelT] | None = None,
) -> Iterator[str | ProtoModelT]:
  """Read records from array records."""
  file = epath.Path(file)
  with contextlib.ExitStack() as stack:
    match file.suffix:
      case ".tfrecords":
        records = tf.data.TFRecordDataset(file).as_numpy_iterator()
      case _:
        raise ValueError(f"File {file} has unknown extension {file.suffix}")

    for record in records:
      if record_type and issubclass(record_type, microscope_utils.ProtoModel):
        yield record_type.from_proto_string(record)
      else:
        yield record


def write_records(
    file: PathLike, records: Iterable[str | microscope_utils.ProtoModel]
) -> None:
  """Write records to disk."""
  file = epath.Path(file)
  with contextlib.ExitStack() as stack:
    match file.suffix:
      case ".tfrecords":
        writer = stack.enter_context(tf.io.TFRecordWriter(file.as_posix()))
      case _:
        raise ValueError(f"File {file} has unknown extension {file.suffix}")

    for record in records:
      if isinstance(record, microscope_utils.ProtoModel):
        record = record.to_proto().SerializeToString()
      match writer:
        case tf.io.TFRecordWriter():
          writer.write(record)
