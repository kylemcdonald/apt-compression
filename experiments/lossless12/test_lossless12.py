from __future__ import annotations

import numpy as np

from .lossless12 import (
    decode_mass_delta_varint,
    decode_mass_elias_fano,
    decode_mass_rice,
    decode_mass_rice_blocks,
    decode_mass_rice_compact,
    decode_mass_rice_split,
    decode_mass_prefix_spatial,
    decode_spatial_mass,
    encode_mass_delta_varint,
    encode_mass_elias_fano,
    encode_mass_rice,
    encode_mass_rice_blocks,
    encode_mass_rice_compact,
    encode_mass_rice_split,
    encode_mass_prefix_spatial,
    encode_spatial_mass,
    grouped_curve_sorted,
    mass_spatial_sorted,
    mass_prefix_spatial_sorted,
    pack_columns12,
    pack_low_six,
    quantize12,
    spatial_hilbert,
    spatial_hilbert_inverse,
    spatial_mass_sorted,
    tuple_keys,
    unpack_columns12,
    unpack_low_six,
    varint_decode,
    varint_encode,
)


def fixture(seed: int = 19, count: int = 20_003) -> np.ndarray:
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(count, 4)).astype(np.float32)
    points[:, 0] *= 20
    points[:, 1] *= 15
    points[:, 2] = points[:, 2] * 60 + 100
    points[:, 3] = rng.choice(
        np.array([6.0, 14.0, 27.98, 55.94, 119.0], dtype=np.float32),
        size=count,
        p=[0.02, 0.08, 0.65, 0.22, 0.03],
    ) + rng.normal(0, 0.02, count)
    return points


def test_fixed_width_packers_roundtrip():
    values = quantize12(fixture()).values
    keys = tuple_keys(values)
    assert np.array_equal(unpack_low_six(pack_low_six(keys)), keys)
    assert np.array_equal(
        unpack_columns12(pack_columns12(values), len(values)),
        values,
    )


def test_varints_roundtrip():
    values = np.array(
        [0, 1, 127, 128, 16_383, 16_384, (1 << 36) - 1, (1 << 48) - 1],
        dtype=np.uint64,
    )
    assert np.array_equal(varint_decode(varint_encode(values), len(values)), values)


def test_hilbert_roundtrip():
    rng = np.random.default_rng(81)
    values = rng.integers(0, 4096, size=(50_000, 3), dtype=np.uint16)
    assert np.array_equal(
        spatial_hilbert_inverse(spatial_hilbert(values)),
        values,
    )
    assert len(np.unique(spatial_hilbert(values))) == len(np.unique(
        values.astype(np.uint64)[:, 0]
        | (values.astype(np.uint64)[:, 1] << np.uint64(12))
        | (values.astype(np.uint64)[:, 2] << np.uint64(24))
    ))


def test_mass_grouped_codecs_roundtrip_exact_multiset():
    values = quantize12(fixture()).values
    ordered, counts = mass_spatial_sorted(values)
    for encode, decode in (
        (encode_mass_delta_varint, decode_mass_delta_varint),
        (encode_mass_elias_fano, decode_mass_elias_fano),
        (encode_mass_rice, decode_mass_rice),
        (encode_mass_rice_split, decode_mass_rice_split),
        (encode_mass_rice_compact, decode_mass_rice_compact),
        (
            lambda ordered, counts: encode_mass_rice_blocks(
                ordered,
                counts,
                block_size=64,
            ),
            decode_mass_rice_blocks,
        ),
    ):
        encoded = encode(ordered, counts)
        assert np.array_equal(decode(encoded), ordered)


def test_all_primary_groupings_encode_exact_combined_keys():
    values = quantize12(fixture(count=10_003)).values
    for primary in range(4):
        for curve in ("morton", "hilbert"):
            ordered, counts = grouped_curve_sorted(values, primary, curve)
            assert len(ordered) == len(values)
            assert int(counts.sum()) == len(values)
            encoded = encode_mass_rice(ordered, counts)
            assert np.array_equal(decode_mass_rice(encoded), ordered)


def test_spatial_plus_mass_roundtrip():
    values = quantize12(fixture(count=25_003)).values
    for curve in ("morton", "hilbert"):
        ordered = spatial_mass_sorted(values, curve)
        assert np.array_equal(
            decode_spatial_mass(encode_spatial_mass(ordered)),
            ordered,
        )


def test_mass_prefix_spatial_roundtrip():
    values = quantize12(fixture(count=25_003)).values
    for prefix_bits in (0, 2, 4, 6, 8, 10, 12):
        ordered = mass_prefix_spatial_sorted(
            values,
            "hilbert",
            prefix_bits,
        )
        encoded = encode_mass_prefix_spatial(ordered, prefix_bits)
        assert np.array_equal(
            decode_mass_prefix_spatial(encoded),
            ordered,
        )
