#!/usr/bin/env python
"""Exact in-memory scorer for renderer experiments.

Scores a candidate renderer state against cached ground-truth network outputs without
writing the 3.6 GB raw video, using the same operations and batch sizes as the reference
decoder's render_video and the repo's evaluate.py. Only frame 1 of each pair depends on
the renderer, so frame 0 is read from an existing reference decode.

  python exact_score.py --gate     # must reproduce the reference decode and its score

Needs a CUDA GPU with DALI video decoding, and a finished reference decode (inflated/0.raw
plus its token checkpoint) in --decode-dir.
"""
import argparse, hashlib, math, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from frame_utils import DaliVideoDataset, camera_size
from modules import DistortionNet, segnet_sd_path, posenet_sd_path

N = 600
EVAL_BATCH = 16     # evaluate.py default
RENDER_BATCH = 8    # render_video's cuda batch
UNCOMPRESSED_BYTES = 37_545_489


def compute_score(seg, pose, archive_bytes):
  return 100 * seg + math.sqrt(10 * pose) + 25 * archive_bytes / UNCOMPRESSED_BYTES


class Reference:
  """The untouched reference decoder, used as a library."""
  def __init__(self, ref_dir: Path, decode_dir: Path):
    self.ref_dir, self.decode_dir = ref_dir.resolve(), decode_dir.resolve()
    sys.path.insert(0, str(self.ref_dir))
    from runtime import f26_inflate
    self.f26 = f26_inflate
    self.archive_path = self.ref_dir / "archive.zip"
    self.device, _ = f26_inflate._configure_device(torch, "cuda", 4)
    # the T4 grader has no TF32; keep the judge networks in full fp32 as well
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    self.parts = f26_inflate.read_residual_archive(self.archive_path)
    self.renderer = f26_inflate._load_renderer(self.ref_dir / "cpr1")

  def semantic_state(self):
    template = self.renderer.SemanticTokenRenderer(96).state_dict()
    state = self.renderer.unpack_variant_semantic_or_none(self.parts.semantic_blob, template)
    assert state is not None, "reference archive is expected to carry a tagged renderer packet"
    return state

  def build_semantic(self, state):
    semantic = self.renderer.SemanticTokenRenderer(96)
    semantic.load_state_dict(state, strict=True)
    return semantic.eval().to(self.device)

  def tokens(self):
    r = self.renderer
    path = self.decode_dir / "inflated" / ".f26_decode_checkpoints" / "tokens_cpu_stage_complete.u8"
    tokens = np.fromfile(path, dtype=np.uint8)
    assert tokens.size == N * r.EVAL_H * r.EVAL_W, f"unexpected token count in {path}"
    return torch.from_numpy(tokens.reshape(N, r.EVAL_H, r.EVAL_W))

  def raw_frames(self):
    W, H = camera_size
    return np.memmap(self.decode_dir / "inflated" / "0.raw", dtype=np.uint8, mode="r", shape=(2 * N, H, W, 3))


def load_judges(device):
  net = DistortionNet().eval().to(device)
  net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
  return net


@torch.inference_mode()
def ground_truth(net, device, cache_path: Path):
  """PoseNet outputs and SegNet argmax of the original video, decoded with DALI like the GPU grader."""
  if cache_path.exists():
    return torch.load(cache_path, map_location=device)
  ds = DaliVideoDataset(["0.mkv"], data_dir=REPO / "videos", batch_size=EVAL_BATCH, device=device)
  ds.prepare_data()
  pose, seg = [], []
  for _, _, batch in ds:
    posenet_out, segnet_out = net(batch.to(device))
    pose.append(posenet_out["pose"].clone())
    seg.append(segnet_out.argmax(dim=1).to(torch.uint8))
  gt = {"pose": torch.cat(pose), "seg": torch.cat(seg)}
  assert gt["pose"].shape[0] == N and gt["seg"].shape[0] == N
  cache_path.parent.mkdir(parents=True, exist_ok=True)
  torch.save(gt, cache_path)
  return gt


@torch.inference_mode()
def render_masters(ref: Reference, semantic, tokens, start, end):
  """Frame 1 of pairs [start, end): same ops as the first loop of render_video. Returns uint8 (B, H, W, 3) on the GPU."""
  r, device = ref.renderer, ref.device
  out = []
  for s in range(start, end, RENDER_BATCH):
    e = min(s + RENDER_BATCH, end)
    indices = torch.arange(s, e, device=device)
    master = F.interpolate(semantic(tokens[s:e].long().to(device), indices), size=(r.CAMERA_H, r.CAMERA_W),
                           mode="bilinear", align_corners=False).clamp(0.0, 255.0).round()
    out.append(master.to(torch.uint8).permute(0, 2, 3, 1))
  return torch.cat(out)


@torch.inference_mode()
def score_state(ref: Reference, net, gt, state, tokens=None, check_against_raw=False):
  """Returns per-pair (pose_dist, seg_dist) tensors for a candidate renderer state_dict."""
  assert N % RENDER_BATCH == 0 and EVAL_BATCH % RENDER_BATCH == 0, "render batches must line up with render_video's"
  device = ref.device
  semantic = ref.build_semantic(state)
  tokens = ref.tokens() if tokens is None else tokens
  raw = ref.raw_frames()
  pose_d, seg_d, mismatched = [], [], 0
  for s in range(0, N, EVAL_BATCH):
    e = min(s + EVAL_BATCH, N)
    masters = render_masters(ref, semantic, tokens, s, e)
    if check_against_raw:
      mismatched += int((masters.cpu() != torch.from_numpy(np.ascontiguousarray(raw[2 * s + 1:2 * e:2]))).sum())
    frame0 = torch.from_numpy(np.ascontiguousarray(raw[2 * s:2 * e:2])).to(device)
    batch = torch.stack([frame0, masters], dim=1)  # (B, 2, H, W, 3)
    posenet_out, segnet_out = net(batch)
    pose_d.append(net.posenet.compute_distortion(posenet_out, {"pose": gt["pose"][s:e]}))
    seg_d.append((segnet_out.argmax(dim=1) != gt["seg"][s:e]).float().mean(dim=(1, 2)))
  result = (torch.cat(pose_d), torch.cat(seg_d))
  return (*result, mismatched) if check_against_raw else result


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--ref-dir", type=Path, default=REPO / "submissions" / "joint_hops")
  parser.add_argument("--decode-dir", type=Path, required=True, help="dir holding inflated/0.raw from the reference decode")
  parser.add_argument("--gt-cache", type=Path, default=Path("/work/gt_cache.pt"))
  parser.add_argument("--gate", action="store_true", help="check the reference state reproduces the reference decode")
  args = parser.parse_args()

  ref = Reference(args.ref_dir, args.decode_dir)
  net = load_judges(ref.device)
  gt = ground_truth(net, ref.device, args.gt_cache)
  tokens = ref.tokens()
  print("token sha256", hashlib.sha256(tokens.numpy().tobytes()).hexdigest())

  torch.cuda.reset_peak_memory_stats()
  import time
  started = time.time()
  pose_d, seg_d, mismatched = score_state(ref, net, gt, ref.semantic_state(), tokens, check_against_raw=True)
  pose, seg = pose_d.mean().item(), seg_d.mean().item()
  archive_bytes = ref.archive_path.stat().st_size
  print(f"pose {pose:.8e}  seg {seg:.8e}  bytes {archive_bytes:,}  score {compute_score(seg, pose, archive_bytes):.6f}")
  print(f"render+score {time.time() - started:.1f}s  peak VRAM {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
  if args.gate:
    print(f"rendered frame-1 bytes differing from the reference decode: {mismatched}")
    sys.exit(0 if mismatched == 0 else 1)


if __name__ == "__main__":
  main()
