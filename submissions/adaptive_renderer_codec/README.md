# joint_hops

Decoder for the `joint_hops` archive. Run `inflate.sh <archive_dir> <output_dir> <video_names_file>` (needs a CUDA GPU).

Builds on PR #141 (semantic_blocks), PR #140 (semantic_joint_ctxmix) and PR #135 (semantic-pose-HPAC_CPR1_polished).
This folder is #141's decoder. Changes: the archive checksum/size in `inflate.py`, the carrier gray/amp constants in
`cpr1/inflate.py`, and the same gray constant in `runtime/ddm_wc1_advisory_runtime.py` (parallel render path).

`compress.sh <pieces_dir>` rebuilds `archive.zip` byte for byte from its stored sections (token stream, HPAC, renderer,
carrier, table) and the recorded block plan, and refuses unless the SHA-256 matches. The sections are in `pieces/` of
https://github.com/RChang06/comma_joint_hops, which also holds the training and search code that produced them.
