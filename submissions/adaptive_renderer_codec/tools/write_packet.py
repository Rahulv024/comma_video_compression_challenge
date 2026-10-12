#!/usr/bin/env python
"""Write the SM3R mode 7 renderer packet: metadata first, then one arithmetic-coded code stream.

  "SM3R" | u8 version | u8 mode 7 | u8 keep_percent | u8 0 | u16 row-prune mask
  for every tensor in template order:  fp16 values (rank < 2)  or  [row bitmap] + fp16 row scales
  renderer_code_coder.encode_codes(codes of every rank >= 2 tensor)

The mode 6 depth table is dropped: every code lies in [-7, 7], which is the coder's alphabet.
"""
import struct

import ddm_mp2_semantic_receiver as R
import renderer_code_coder as coder
import sections as S


def write_packet(packet: S.Packet):
  """Returns (packet bytes, offsets inside the packet worth trying as archive block boundaries)."""
  names = R._quantized_names(S._template())
  head = [R.SM3R_MAGIC, bytes((R.SM3R_VERSION, R.SM3R_ROW_PRUNE_CODED_MODE, packet.keep_percent, 0)),
          struct.pack("<H", R._selection_mask(names, R.ROW_PRUNE_NAMES))]
  metadata = b"".join(item.raw if item.kind == "fp16" else item.bitmap + item.scales for item in packet.items)
  quantized = packet.quantized()
  payload = coder.encode_codes([item.codes for item in quantized], [item.rows for item in quantized])
  prefix = b"".join(head) + metadata
  return prefix + payload, [len(prefix)]
