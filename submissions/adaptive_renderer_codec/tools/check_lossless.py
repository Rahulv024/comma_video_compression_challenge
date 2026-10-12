#!/usr/bin/env python
"""Gates for the lossless renderer-code experiment. Run from the repo root:

  python submissions/adaptive_renderer_codec/tools/check_lossless.py

  gate 1  the archive builder reproduces the reference archive byte for byte
  gate 2  the renderer packet parser/serializer round-trips exactly
  gate 3  cpr1/renderer_code_coder.py round-trips the codes; reports its size against the targets
  gate 4  a packet written by tools/write_packet.py decodes, through this submission's own decoder,
          to weights bit-identical to the reference; reports the new archive size and projected score
Gates 3 and 4 are skipped until the corresponding code exists.
"""
import argparse, hashlib, importlib, math, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUBMISSION = HERE.parent
sys.path.insert(0, str(HERE))
import sections as S

REFERENCE_SHA256 = "95a9fca6c79b397a36c7f8ea4f5d922ed604d0a9749dd7eb626c2ee4ce6348ae"
# measured on the reference archive: what its 59,376 codes cost today, and what coders reach
CODES_COST_TODAY = 25_721          # block 2 (47,761 B) minus the carrier stored alone (22,040 B)
TARGET_EXISTING_RANS = 25_525      # the decoder's runtime/entropy/adaptive_ans.py model, 16 streams
TARGET_ADAPTIVE = 25_313           # best of three simple adaptive order-0 models per tensor (simulated)
# reference distortions on this machine (exact_score.py --gate)
REF_SEG, REF_POSE, UNCOMPRESSED = 1.24333703e-04, 3.64099515e-06, 37_545_489


def score(archive_bytes):
  return 100 * REF_SEG + math.sqrt(10 * REF_POSE) + 25 * archive_bytes / UNCOMPRESSED


def edge_cases():
  """Small synthetic stream lists that stress tensor boundaries, extreme symbols and long runs."""
  import numpy as np
  rng = np.random.default_rng(0)
  i8 = lambda values: np.asarray(values, dtype=np.int8)
  yield "one code", [i8([0])]
  yield "one code per tensor", [i8([c]) for c in range(-7, 8)]
  yield "all -7", [i8([-7] * 5000)]
  yield "all +7", [i8([7] * 5000)]
  yield "long run then a rare symbol", [i8([0] * 20000 + [-7])]
  yield "alternating extremes", [i8([-7, 7] * 3000)]
  yield "uniform random, 16 tensors", [i8(rng.integers(-7, 8, size=n)) for n in rng.integers(1, 3000, size=16)]
  yield "boundary between very different tensors", [i8([7] * 4000), i8([-7] * 4000), i8([0] * 16)]


def round_trips(coder, case):
  try:
    out = coder.decode_codes(coder.encode_codes(case), [s.size for s in case])
    return len(out) == len(case) and all(a.size == b.size and (a == b).all() for a, b in zip(out, case))
  except Exception as error:  # report which case broke instead of stopping at the first
    print(f"        edge case raised {type(error).__name__}: {error}")
    return False


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--reference", type=Path, default=SUBMISSION.parent / "joint_hops" / "archive.zip")
  parser.add_argument("--out", type=Path, default=None, help="write the gate-4 archive here")
  args = parser.parse_args()
  S.use_decoder(SUBMISSION)
  import torch
  import ddm_mp2_semantic_receiver as R

  reference = args.reference.read_bytes()
  assert hashlib.sha256(reference).hexdigest() == REFERENCE_SHA256, "unexpected reference archive"
  sections = S.split_sections(args.reference)

  rebuilt = S.build_archive(sections)
  print(f"gate 1  rebuilt reference: {len(rebuilt):,} B  {'PASS' if rebuilt == reference else 'FAIL (bytes differ)'}")
  wide = S.build_archive(sections, cuts=S.REFERENCE_CUTS, wide=True)
  print(f"        same sections, wider block search: {len(wide):,} B ({len(wide) - len(reference):+d})")

  blob = S.renderer_packet(sections)
  packet = S.parse_mode6(blob)
  print(f"gate 2  packet round trip: {'PASS' if S.serialize_mode6(packet) == blob else 'FAIL'}  "
        f"({len(blob):,} B, {sum(i.codes.size for i in packet.quantized()):,} codes in {len(packet.quantized())} tensors)")
  if rebuilt != reference or S.serialize_mode6(packet) != blob:
    sys.exit(1)

  streams = [item.codes for item in packet.quantized()]
  groups = [item.rows for item in packet.quantized()]
  import renderer_code_coder as coder
  try:
    payload = coder.encode_codes(streams, groups)
  except NotImplementedError as error:
    print(f"gate 3  skipped: {error}")
    return
  decoded = coder.decode_codes(payload, [s.size for s in streams], groups)
  ok = len(decoded) == len(streams) and all((a == b).all() and a.size == b.size for a, b in zip(decoded, streams))
  print(f"gate 3  coder round trip: {'PASS' if ok else 'FAIL'}  {len(payload):,} B  "
        f"(today {CODES_COST_TODAY:,}; existing rANS model {TARGET_EXISTING_RANS:,}; adaptive target {TARGET_ADAPTIVE:,})")
  if not ok:
    sys.exit(1)
  deterministic = coder.encode_codes(streams, groups) == payload
  modes = [payload[i // 2] >> (4 * (i % 2)) & 0xF for i in range(len(streams))]
  print(f"        codes reconstructed: {sum(d.size for d in decoded):,}   models chosen per tensor: {modes}")
  failures = [name for name, case in edge_cases() if not round_trips(coder, case)]
  print(f"        deterministic: {'PASS' if deterministic else 'FAIL'}   "
        f"edge cases: {'PASS' if not failures else 'FAIL ' + ', '.join(failures)}")
  if not deterministic or failures:
    sys.exit(1)

  try:
    writer = importlib.import_module("write_packet")
  except ModuleNotFoundError:
    print("gate 4  skipped: tools/write_packet.py (write_packet(packet) -> bytes, cut offsets) does not exist yet")
    return
  new_blob, packet_cuts = writer.write_packet(packet)
  template = S._template()
  want = R.unpack_variant_semantic_or_none(blob, template)
  got = R.unpack_variant_semantic_or_none(new_blob, template)
  same = got is not None and all(torch.equal(want[k], got[k]) for k in want)
  print(f"gate 4  decoded weights bit-identical to the reference: {'PASS' if same else 'FAIL'}")
  if not same:
    sys.exit(1)
  from runtime.segment_layout import pack_segments
  from runtime import residual_archive as RA
  section = pack_segments([new_blob])
  base = RA.RX1_MODEL_HEADER.size + len(sections.hpac) + (len(section) - len(new_blob))
  cuts = [RA.RX1_MODEL_HEADER.size + len(sections.hpac), base + len(new_blob), *(base + c for c in packet_cuts), *S.REFERENCE_CUTS[:1]]
  archive = S.build_archive(sections, section, cuts=cuts, wide=True, verbose=True)
  print(f"        archive {len(archive):,} B ({len(archive) - len(reference):+d})  "
        f"projected score {score(len(archive)):.6f} vs {score(len(reference)):.6f}")
  if args.out:
    args.out.write_bytes(archive)
    print(f"        wrote {args.out}  sha256 {hashlib.sha256(archive).hexdigest()}")


if __name__ == "__main__":
  main()
