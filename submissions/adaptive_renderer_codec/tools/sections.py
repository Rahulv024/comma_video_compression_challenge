#!/usr/bin/env python
"""Split a reference archive into its stored sections, parse the renderer packet, and rebuild archives.

Harness for lossless-format experiments. Everything here is exact: `build_archive` with the
original section and block plan must reproduce the reference archive byte for byte, and
`serialize_mode6(parse_mode6(blob))` must return `blob`.
"""
import io, struct, sys, zipfile
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import brotli


_DECODER_DIR = None


def use_decoder(decoder_dir: Path):
  """Put one submission folder's decoder modules on sys.path (call once per process)."""
  global _DECODER_DIR
  decoder_dir = _DECODER_DIR = Path(decoder_dir).resolve()
  for p in (decoder_dir / "cpr1", decoder_dir):
    if str(p) not in sys.path:
      sys.path.insert(0, str(p))


@dataclass
class Sections:
  rx1_fields: tuple      # (version, codec, table_mode, reserved)
  hpac: bytes            # stored entropy-model section
  renderer: bytes        # stored renderer section (LAY1-wrapped packet)
  carrier: bytes         # stored carrier section
  tail: bytes            # residual table + token stream, stored after the model blocks


def split_sections(archive_path: Path) -> Sections:
  from runtime import residual_archive as RA
  from runtime.block_container import decode_model_blocks
  payload = zipfile.ZipFile(archive_path).read("p")
  models, tail = decode_model_blocks(payload)
  magic, *fields = RA.RX1_MODEL_HEADER.unpack_from(models)
  assert magic == RA.RX1_MAGIC
  hpac_b, sem_b, car_b = fields[-3:]
  o = RA.RX1_MODEL_HEADER.size
  assert o + hpac_b + sem_b + car_b == len(models)
  return Sections(tuple(fields[:4]), models[o:o + hpac_b], models[o + hpac_b:o + hpac_b + sem_b],
                  models[o + hpac_b + sem_b:], tail)


def renderer_packet(sections: Sections) -> bytes:
  from runtime.segment_layout import unpack_segments
  return unpack_segments(sections.renderer)


# ---- renderer packet (SM3R mode 6) ------------------------------------------------------

@dataclass
class Item:
  name: str
  kind: str                 # "fp16" (biases, norms) or "q" (quantized)
  raw: bytes = b""          # fp16: the raw little-endian values
  bits: int = 0             # q: stored code width
  bitmap: bytes = b""       # q: kept-row bitmap for row-pruned tensors, else empty
  scales: bytes = b""       # q: raw fp16 row scales
  codes: np.ndarray = None  # q: int8 codes, flat, in stored order
  rows: int = 0             # q: number of scale groups the codes are stored in


@dataclass
class Packet:
  keep_percent: int
  items: list = field(default_factory=list)

  def quantized(self):
    return [item for item in self.items if item.kind == "q"]


def _template():
  # cpr1/inflate.py shares its name with the submission's top-level inflate.py, so load it the decoder's way
  from runtime import f26_inflate
  return f26_inflate._load_renderer(_DECODER_DIR / "cpr1").SemanticTokenRenderer(96).state_dict()


def pack_signed_bits(codes: np.ndarray, bits: int) -> bytes:
  """Inverse of the decoder's _unpack_signed_bits."""
  values = np.asarray(codes, dtype=np.int16)
  assert values.min() >= -(1 << (bits - 1)) and values.max() < (1 << (bits - 1))
  unsigned = (values & ((1 << bits) - 1)).astype(np.uint16)
  bitstream = ((unsigned[:, None] >> np.arange(bits, dtype=np.uint16)[None]) & 1).astype(np.uint8)
  return np.packbits(bitstream.ravel(), bitorder="little").tobytes()


def parse_mode6(blob: bytes) -> Packet:
  import torch
  import ddm_mp2_semantic_receiver as R
  assert blob[:4] == R.SM3R_MAGIC and blob[4] == R.SM3R_VERSION and blob[5] == R.SM3R_ROW_PRUNE_MIXED_MODE
  template = _template()
  names = R._quantized_names(template)
  packet = Packet(keep_percent=blob[6])
  rem = memoryview(blob)[10:]
  depths, rem = R._decode_depth_nibbles(rem, len(names), "SM3R")
  allocation = dict(zip(names, depths))
  for name, value in template.items():
    if value.ndim < 2:
      raw, rem = R._take(rem, value.numel() * 2, name)
      packet.items.append(Item(name, "fp16", raw=bytes(raw)))
      continue
    tmpl, label, bitmap = value, name, b""
    if name in R.ROW_PRUNE_NAMES:
      rows = int(value.shape[0])
      view, rem = R._take(rem, (rows + 7) // 8, "bitmap")
      bitmap = bytes(view)
      kept = int(np.unpackbits(np.frombuffer(view, np.uint8), bitorder="little")[:rows].sum())
      tmpl, label = torch.empty((kept, value.numel() // rows)), "pruned.rows"
    groups = R._scale_count(label, tmpl)
    scales, rem = R._take(rem, groups * 2, "scales")
    codes, rem = R._unpack_signed_bits(rem, tmpl.numel(), allocation[name])
    packet.items.append(Item(name, "q", bits=allocation[name], bitmap=bitmap, scales=bytes(scales),
                             codes=codes.numpy().astype(np.int8), rows=groups))
  assert len(rem) == 0
  return packet


def serialize_mode6(packet: Packet) -> bytes:
  import ddm_mp2_semantic_receiver as R
  template = _template()
  names = R._quantized_names(template)
  mask = R._selection_mask(names, R.ROW_PRUNE_NAMES)
  depths = [item.bits for item in packet.quantized()]
  nibbles = bytes((depths[i] | ((depths[i + 1] if i + 1 < len(depths) else 0) << 4)) for i in range(0, len(depths), 2))
  out = [R.SM3R_MAGIC, bytes((R.SM3R_VERSION, R.SM3R_ROW_PRUNE_MIXED_MODE, packet.keep_percent, 0)),
         struct.pack("<H", mask), nibbles]
  for item in packet.items:
    if item.kind == "fp16":
      out.append(item.raw)
    else:
      out += [item.bitmap, item.scales, pack_signed_bits(item.codes, item.bits)]
  return b"".join(out)


# ---- archive building ------------------------------------------------------------------

REFERENCE_CUTS = (13276, 20827)   # #146's block boundaries inside the model buffer


def _zip_p(payload: bytes) -> bytes:
  buf = io.BytesIO()
  with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as z:
    info = zipfile.ZipInfo("p", date_time=(1980, 1, 1, 0, 0, 0))
    info.create_system = 3
    info.external_attr = 0o644 << 16
    z.writestr(info, payload)
  return buf.getvalue()


def _best_block(data: bytes, wide: bool):
  """Smallest encoding of one block: (bytes including its header, description)."""
  from runtime.block_container import transpose
  best = (4 + len(data), bytes((0,)) + len(data).to_bytes(3, "little") + data, "raw")
  qualities, windows, modes = ((10, 11), (16, 18, 20, 22), (0, 1)) if wide else ((10, 11), (16,), (0,))
  for stride in (0, 2):
    source = transpose(data, stride) if stride else data
    for q in qualities:
      for w in windows:
        for m in modes:
          comp = brotli.compress(source, quality=q, lgwin=w, mode=m)
          size = 4 + (2 if stride else 0) + len(comp)
          if size < best[0]:
            head = bytes((2 | (128 if stride else 0),)) + len(comp).to_bytes(3, "little") + (bytes((1, stride)) if stride else b"")
            best = (size, head + comp, f"brotli q{q} lgwin{w} mode{m} stride{stride}")
  return best


def build_archive(sections: Sections, renderer_section: bytes = None, cuts=REFERENCE_CUTS, wide=False, verbose=False) -> bytes:
  """Assemble archive.zip. `cuts` are candidate block boundaries (offsets into the model buffer);
  the cheapest subset of them is chosen by dynamic programming."""
  from runtime import residual_archive as RA
  from runtime.block_container import decode_model_blocks
  renderer_section = sections.renderer if renderer_section is None else renderer_section
  models = RA.RX1_MODEL_HEADER.pack(RA.RX1_MAGIC, *sections.rx1_fields, len(sections.hpac), len(renderer_section),
                                    len(sections.carrier)) + sections.hpac + renderer_section + sections.carrier
  points = sorted({0, len(models), *(c for c in cuts if 0 < c < len(models))})
  cache = {}
  cost = {0: (0, [])}
  for j in range(1, len(points)):
    options = []
    for i in range(j):
      key = (points[i], points[j])
      if key not in cache:
        cache[key] = _best_block(models[key[0]:key[1]], wide)
      size, block, how = cache[key]
      options.append((cost[points[i]][0] + size, cost[points[i]][1] + [(key, block, how)]))
    cost[points[j]] = min(options, key=lambda x: x[0])
  blocks = cost[len(models)][1]
  payload = b"BLK2" + struct.pack("<IB", len(models), len(blocks)) + b"".join(b for _, b, _ in blocks) + sections.tail
  assert decode_model_blocks(payload) == (models, sections.tail), "block round trip failed"
  if verbose:
    for (a, b), block, how in blocks:
      print(f"  block [{a}:{b}] {b - a} B -> {len(block)} B  {how}")
  return _zip_p(payload)
