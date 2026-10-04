import itertools
import struct
import unittest

from experiments.loopkv.capacity import Geometry, account
from experiments.loopkv.oracle import AddressOracle, DenseOracle, resolve_source


class AddressContractTests(unittest.TestCase):
    def test_dense_payload_identity_and_actual_versions(self):
        for loops, prompt, tail in itertools.product((1, 2, 4), (0, 15, 16, 17), (0, 1, 17)):
            with self.subTest(loops=loops, prompt=prompt, tail=tail):
                dense, alias = DenseOracle(loops, 2), AddressOracle(loops, 2)
                counts = [loops] * prompt + [1 if p % 2 == 0 else loops for p in range(tail)]
                for position, count in enumerate(counts):
                    for depth in range(count):
                        for layer in range(2):
                            value = struct.pack("4I", alias.generation, position, depth, layer)
                            dense.write(position, depth, layer, value)
                            alias.write(position, depth, layer, value)
                            self.assertEqual(
                                alias.read(position, depth, layer, 0, current=True), value
                            )
                    dense.finalize(position, count)
                    alias.finalize(position, count)
                    for depth, layer in itertools.product(range(loops), range(2)):
                        self.assertEqual(
                            alias.read(position, depth, layer, 0),
                            dense.payload[position, depth, layer],
                        )
                self.assertEqual(len(alias.records), sum(counts) * 2)
                self.assertEqual(len(dense.payload), len(counts) * loops * 2)

    def test_publication_requires_all_layers_and_earlier_depths(self):
        alias = AddressOracle(4, 2)
        alias.write(0, 1, 0, b"x")
        with self.assertRaisesRegex(ValueError, "unwritten"):
            alias.finalize(0, 2)
        with self.assertRaisesRegex(ValueError, "unfinalized"):
            alias.read(0, 1, 0, 0)
        for count in (0, 5):
            with self.assertRaisesRegex(ValueError, "loops_done"):
                alias.finalize(0, count)

    def test_release_reuse_rejects_stale_generation(self):
        alias = AddressOracle(1, 1)
        alias.write(0, 0, 0, b"old")
        alias.finalize(0, 1)
        alias.release()
        self.assertFalse(alias.records)
        alias.write(0, 0, 0, b"new")
        alias.finalize(0, 1)
        with self.assertRaisesRegex(ValueError, "stale"):
            alias.read(0, 0, 0, 0)
        self.assertEqual(alias.read(0, 0, 0, 1), b"new")
        with self.assertRaisesRegex(ValueError, "overwrite"):
            alias.write(0, 0, 0, b"bad")

    def test_depth_indices_reject_python_negative_indexing(self):
        for query, exit_depth in ((-1, 0), (0, -1), (4, 0), (0, 4)):
            with self.assertRaises(ValueError):
                resolve_source(query, exit_depth, 4)


class CapacityTests(unittest.TestCase):
    def test_document_payload_examples(self):
        shape = Geometry(24, 16, 128, 4)
        self.assertEqual(shape.record_bytes, 196608)
        row = account(shape, 128, [2] * 1024)
        self.assertEqual(row["logical_kv_bytes"], 864 * 1024**2)
        self.assertEqual(row["unique_live_payload_bytes"], 480 * 1024**2)
        self.assertEqual(row["avoidable_promotion_payload_bytes"], 384 * 1024**2)
        self.assertEqual(row["compact_record_map_int32_bytes"], 16384)

    def test_full_depth_negative_control_and_fragmentation(self):
        row = account(Geometry(2, 2, 8, 4), 17, [4])
        self.assertEqual(row["avoidable_promotion_payload_bytes"], 0)
        self.assertGreater(row["rectangular_fragmentation_bytes"], 0)

    def test_invalid_loop_counts(self):
        for counts in ([0], [5], [-1], [True]):
            with self.assertRaises(ValueError):
                account(Geometry(2, 2, 8, 4), 0, counts)


if __name__ == "__main__":
    unittest.main()
