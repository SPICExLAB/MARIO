# Third-party components

## Not redistributed with this repository

The two components below are needed only to **train PoseNet**. They carry licenses that
are incompatible with redistributing them inside a permissively licensed repository, so
they are not included here. `scripts/fetch_third_party.sh` puts them in place, and
`.gitignore` keeps them out of the repository afterwards.

Nothing in the inertial-odometry pipeline imports them: training, inference and
evaluation of every model in `configs/{nymeria,aria,tlio}/` work without them.

| Fetched into | Origin | License | Used by |
|---|---|---|---|
| `core/articulate/` | `articulate` toolkit from [Xinyu-Yi/TransPose](https://github.com/Xinyu-Yi/TransPose) | GPL-3.0 | `datasets/nymeriadataset.py`, to turn Xsens poses into SMPL joint labels |
| `nymeria/` | [facebookresearch/nymeria_dataset](https://github.com/facebookresearch/nymeria_dataset), branch `nymeria_dataset_legacy` | CC BY-NC 4.0 | `preprocess_nymeria_body.py`, to read Xsens body motion |
| `assets/smpl/basicmodel_m.pkl` | [SMPL](https://smpl.is.tue.mpg.de) (Max Planck Institute) | SMPL Model License | The body model itself; download separately |

The `nymeria` project on PyPI is an unrelated package published by Nymeria LLC.
`pip install nymeria` does **not** install Meta's loader.

Both components are used through their public APIs, so the upstream versions work
unmodified: `datasets/nymeriadataset.py` calls `ParametricModel.forward_kinematics`, and
`preprocess_nymeria_body.py` passes only the options the installed loader declares.

## Included in this repository

MARIO's own code is released under the MIT License (see `LICENSE`). Parts of
`model/code.py`, `train_motion.py`, `utils/` and `datasets/` are derived from
[AirIO](https://github.com/Air-IO/Air-IO) and remain subject to the BSD 3-Clause
License, reproduced in full below as that license requires.

```
BSD 3-Clause License

Copyright (c) 2025, Carnegie Mellon University

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Datasets

Nymeria, Aria Everyday Activities and TLIO are distributed by their owners under their
own terms and are not included here.
