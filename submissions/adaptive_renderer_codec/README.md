# adaptive_renderer_codec

## What this is

This submission improves the compression efficiency of an existing video compression solution. It reduces the archive size without changing the reconstructed video or increasing distortion. The verified score is 0.136097.

## What it builds on

The submission builds on PR #146, retaining its existing renderer, model weights, class maps, and video reconstruction process. The improvement focuses specifically on how the renderer's quantized weight codes are compressed and stored.

## What changed

The original implementation stored renderer weight codes using fixed 4-bit values. The updated implementation uses adaptive arithmetic coding, which assigns fewer bits to frequently occurring values and more bits to less common values. This reduces the storage required without changing the underlying weight values.

## Why the archive is smaller and the video is the same

The new compression method is lossless, meaning the decoder reconstructs exactly the same weight codes as the original implementation. Therefore, the renderer produces identical video frames with unchanged distortion. The final archive is 447 bytes smaller, reducing its size from 177,106 bytes to 176,659 bytes.

## Usage

```bash
bash inflate.sh <archive_dir> <output_dir> <video_names_file>
bash compress.sh <path to PR #146 (joint_hops) archive.zip>
```

- `inflate.sh` needs a CUDA GPU, a C compiler (`cc`) and Brotli 1.2.0.
- `compress.sh` rebuilds `archive.zip` from PR #146's archive and stops unless every check in `tools/check_lossless.py` passes.

## Files changed from PR #146

- `cpr1/renderer_code_coder.py`: new, the arithmetic coder.
- `cpr1/ddm_mp2_semantic_receiver.py`: adds renderer packet mode 7.
- `tools/`: new, archive rebuild, packet writer and checks.
- `inflate.py`: pinned archive checksum and size.
- `compress.sh`: rebuilds this archive; `compress.py` removed.

## Archive

- Size: 176,659 bytes
- SHA-256: `a020fe5532f90383eeef37c8bed628798d8aba38af56877dc08e24d927237531`

Lineage: PR #146 (joint_hops), which builds on PR #141, PR #140 and PR #135.
