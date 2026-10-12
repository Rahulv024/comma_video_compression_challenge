"""Lossless adaptive arithmetic coder for the renderer's quantized weight codes.

Codes are integers in [-7, 7] (15 symbols). Every tensor is coded with its own adaptive
frequency model, chosen by the encoder from a small catalog and signalled with one nibble,
and all tensors share one arithmetic-coded bit stream.

Everything in the coding path is integer arithmetic, so the decoder reproduces the
encoder's intervals exactly on any machine.
"""

from __future__ import annotations

import numpy as np

ALPHABET = 15
OFFSET = 7

_HALF = 1 << 31
_QUARTER = 1 << 30
_TOP = (1 << 32) - 1

_TENSOR_LIMIT = 1 << 16  # halve a tensor model when its total exceeds this
_ROW_LIMIT = 1 << 12     # same for the per-row counts of the row-mix models

# mode -> (kind, parameter)
#   "plain": one adaptive count table per tensor; parameter is the per-symbol increment.
#   "row":   per-row counts mixed with the tensor table as a prior; parameter is the
#            prior weight K, i.e. p(s) = (row[s] + K * tensor[s] / tensor_total) / (row_total + K).
MODES = (
    ("plain", 1),
    ("plain", 2),
    ("plain", 4),
    ("plain", 8),
    ("row", 4),
    ("row", 16),
    ("row", 64),
    ("row", 256),
)
_ROW_TENSOR_INCREMENT = 2


class RendererCodeCoderError(ValueError):
    """A code stream or its coded payload is malformed."""


class _Model:
    """Adaptive frequencies for one tensor. `frequencies()` returns positive integers."""

    def __init__(self, mode: int, row_length: int):
        self.kind, self.parameter = MODES[mode]
        self.row_length = row_length
        self.tensor = [1] * ALPHABET
        self.tensor_total = ALPHABET
        self.row = [0] * ALPHABET
        self.row_total = 0
        self.position = 0

    def frequencies(self) -> tuple[list[int], int]:
        if self.kind == "plain":
            return self.tensor, self.tensor_total
        prior, tensor_total = self.parameter, self.tensor_total
        freqs = [r * tensor_total + prior * t for r, t in zip(self.row, self.tensor)]
        return freqs, (self.row_total + prior) * tensor_total

    def update(self, symbol: int) -> None:
        increment = self.parameter if self.kind == "plain" else _ROW_TENSOR_INCREMENT
        self.tensor[symbol] += increment
        self.tensor_total += increment
        if self.tensor_total > _TENSOR_LIMIT:
            self.tensor = [(count + 1) // 2 for count in self.tensor]
            self.tensor_total = sum(self.tensor)
        if self.kind == "row":
            self.row[symbol] += 1
            self.row_total += 1
            if self.row_total > _ROW_LIMIT:
                self.row = [(count + 1) // 2 for count in self.row]
                self.row_total = sum(self.row)
            self.position += 1
            if self.position == self.row_length:
                self.row = [0] * ALPHABET
                self.row_total = 0
                self.position = 0


class _Encoder:
    def __init__(self) -> None:
        self.low = 0
        self.high = _TOP
        self.pending = 0
        self.bits: list[int] = []

    def _emit(self, bit: int) -> None:
        self.bits.append(bit)
        if self.pending:
            self.bits.extend([bit ^ 1] * self.pending)
            self.pending = 0

    def encode(self, cum_low: int, cum_high: int, total: int) -> None:
        span = self.high - self.low + 1
        self.high = self.low + span * cum_high // total - 1
        self.low = self.low + span * cum_low // total
        while True:
            if self.high < _HALF:
                self._emit(0)
            elif self.low >= _HALF:
                self._emit(1)
                self.low -= _HALF
                self.high -= _HALF
            elif self.low >= _QUARTER and self.high < 3 * _QUARTER:
                self.pending += 1
                self.low -= _QUARTER
                self.high -= _QUARTER
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1

    def finish(self) -> bytes:
        self.pending += 1
        self._emit(0 if self.low < _QUARTER else 1)
        packed = np.packbits(np.asarray(self.bits, dtype=np.uint8)).tobytes()
        # the decoder reads zeros past the end, so trailing zero bytes carry nothing
        return packed.rstrip(b"\x00")


class _Decoder:
    def __init__(self, payload: bytes) -> None:
        self.bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8)).tolist()
        self.cursor = 0
        self.low = 0
        self.high = _TOP
        self.value = 0
        for _ in range(32):
            self.value = (self.value << 1) | self._next()

    def _next(self) -> int:
        cursor = self.cursor
        self.cursor = cursor + 1
        return self.bits[cursor] if cursor < len(self.bits) else 0

    def decode(self, freqs: list[int], total: int) -> int:
        span = self.high - self.low + 1
        target = ((self.value - self.low + 1) * total - 1) // span
        cum_low = 0
        symbol = 0
        while cum_low + freqs[symbol] <= target:
            cum_low += freqs[symbol]
            symbol += 1
        self.high = self.low + span * (cum_low + freqs[symbol]) // total - 1
        self.low = self.low + span * cum_low // total
        while True:
            if self.high < _HALF:
                pass
            elif self.low >= _HALF:
                self.low -= _HALF
                self.high -= _HALF
                self.value -= _HALF
            elif self.low >= _QUARTER and self.high < 3 * _QUARTER:
                self.low -= _QUARTER
                self.high -= _QUARTER
                self.value -= _QUARTER
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1
            self.value = (self.value << 1) | self._next()
        return symbol


def _row_lengths(counts: list[int], groups: list[int] | None) -> list[int]:
    if groups is None:
        groups = [1] * len(counts)
    if len(groups) != len(counts):
        raise RendererCodeCoderError("groups and streams differ in length")
    lengths = []
    for count, group in zip(counts, groups):
        if count < 0 or group < 1 or count % group:
            raise RendererCodeCoderError("a stream does not divide into its row groups")
        lengths.append(max(1, count // group))
    return lengths


def _symbols(stream: np.ndarray) -> list[int]:
    values = np.asarray(stream).astype(np.int64).ravel()
    if values.size and (values.min() < -OFFSET or values.max() > OFFSET):
        raise RendererCodeCoderError("renderer codes must lie in [-7, 7]")
    return (values + OFFSET).tolist()


def _encode_stream(encoder: _Encoder, symbols: list[int], mode: int, row_length: int) -> None:
    model = _Model(mode, row_length)
    for symbol in symbols:
        freqs, total = model.frequencies()
        cum_low = sum(freqs[:symbol])
        encoder.encode(cum_low, cum_low + freqs[symbol], total)
        model.update(symbol)


def _best_mode(symbols: list[int], row_length: int) -> int:
    """Smallest stand-alone encoding; ties go to the lowest mode number."""
    best_mode, best_bits = 0, None
    for mode in range(len(MODES)):
        trial = _Encoder()
        _encode_stream(trial, symbols, mode, row_length)
        bits = len(trial.bits) + trial.pending
        if best_bits is None or bits < best_bits:
            best_mode, best_bits = mode, bits
    return best_mode


def encode_codes(streams: list[np.ndarray], groups: list[int] | None = None) -> bytes:
    """Encode one array of codes per quantized tensor, in template order.

    `groups[i]` is the number of equal-length rows stream i is stored in (default 1).
    It only steers the row-aware models; any value that divides the stream is valid.
    """
    symbol_lists = [_symbols(stream) for stream in streams]
    row_lengths = _row_lengths([len(symbols) for symbols in symbol_lists], groups)
    modes = [_best_mode(symbols, length) for symbols, length in zip(symbol_lists, row_lengths)]
    header = bytearray((len(modes) + 1) // 2)
    for index, mode in enumerate(modes):
        header[index // 2] |= mode << (4 * (index % 2))
    encoder = _Encoder()
    for symbols, mode, length in zip(symbol_lists, modes, row_lengths):
        _encode_stream(encoder, symbols, mode, length)
    return bytes(header) + encoder.finish()


def decode_codes(payload: bytes, counts: list[int], groups: list[int] | None = None) -> list[np.ndarray]:
    """Inverse of encode_codes; counts[i] is the number of codes in stream i."""
    row_lengths = _row_lengths(list(counts), groups)
    header_bytes = (len(counts) + 1) // 2
    if len(payload) < header_bytes:
        raise RendererCodeCoderError("truncated renderer code header")
    modes = [(payload[index // 2] >> (4 * (index % 2))) & 0xF for index in range(len(counts))]
    if any(mode >= len(MODES) for mode in modes):
        raise RendererCodeCoderError("unknown renderer code model")
    if len(counts) % 2 and payload[header_bytes - 1] >> 4:
        raise RendererCodeCoderError("nonzero renderer code header padding")
    decoder = _Decoder(payload[header_bytes:])
    output = []
    for count, mode, length in zip(counts, modes, row_lengths):
        model = _Model(mode, length)
        codes = np.empty(count, dtype=np.int8)
        for index in range(count):
            freqs, total = model.frequencies()
            symbol = decoder.decode(freqs, total)
            codes[index] = symbol - OFFSET
            model.update(symbol)
        output.append(codes)
    return output
