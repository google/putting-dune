# Copyright 2023 The Putting Dune Authors.
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

"""IO Tests."""
import dataclasses
from typing import ClassVar, Type

from absl.testing import absltest
from absl.testing import parameterized
from putting_dune import io
from putting_dune import microscope_utils
from putting_dune import putting_dune_pb2


@dataclasses.dataclass(frozen=True)
class TestPoint2D(microscope_utils.ProtoModel[putting_dune_pb2.Point2D]):
  ProtoMessage: ClassVar[Type[putting_dune_pb2.Point2D]] = (  # pylint: disable=invalid-name
      putting_dune_pb2.Point2D
  )

  x: float
  y: float

  @classmethod
  def from_proto(cls, message: putting_dune_pb2.Point2D) -> "TestPoint2D":
    return cls(x=message.x, y=message.y)

  def to_proto(self) -> putting_dune_pb2.Point2D:
    return putting_dune_pb2.Point2D(x=self.x, y=self.y)


class IoTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.records = [
        TestPoint2D(x=1, y=1),
        TestPoint2D(x=2, y=2),
        TestPoint2D(x=3, y=3),
    ]

  @parameterized.parameters(
      ("test.tfrecords",),
  )
  def test_writes_and_reads_multiple_records(self, filename: str):
    tempfile = self.create_tempfile(filename)

    io.write_records(tempfile.full_path, self.records)
    restored_records = io.read_records(tempfile.full_path, TestPoint2D)

    for record_a, record_b in zip(self.records, restored_records):
      self.assertEqual(record_a, record_b)

  @parameterized.parameters(
      ("test.tfrecords",),
  )
  def test_writes_and_reads_multiple_records_with_bytes(self, filename: str):
    tempfile = self.create_tempfile(filename)

    io.write_records(
        tempfile.full_path,
        [
            record.to_proto().SerializeToString(deterministic=True)
            for record in self.records
        ],
    )
    restored_records = io.read_records(tempfile.full_path)

    for record_a, record_b_bytes in zip(self.records, restored_records):
      record_b = TestPoint2D.from_proto(
          TestPoint2D.ProtoMessage.FromString(record_b_bytes)
      )
      self.assertEqual(record_a, record_b)

  def test_raises_exception_on_invalid_extension(self):
    tempfile = self.create_tempfile("test.badextension")

    with self.assertRaises(ValueError):
      next(io.read_records(tempfile.full_path))

    with self.assertRaises(ValueError):
      next(io.read_records(tempfile.full_path, TestPoint2D))

    with self.assertRaises(ValueError):
      next(io.write_records(tempfile.full_path, self.records))


if __name__ == "__main__":
  absltest.main()
