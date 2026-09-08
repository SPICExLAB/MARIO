#!/usr/bin/env bash
# Fetch the third-party components needed to TRAIN PoseNet.
#
# They are not redistributed with MARIO because they carry licenses of their own
# (see THIRD_PARTY.md). Nothing here is needed to train, run or evaluate the
# inertial-odometry models; only the PoseNet data path uses them.
#
# Usage:  bash scripts/fetch_third_party.sh
set -euo pipefail
cd "$(dirname "$0")/.."
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT

# 1) SMPL kinematics: the `articulate` toolkit from TransPose (GPL-3.0).
#    Used by datasets/nymeriadataset.py to turn Xsens poses into SMPL joint labels.
if [ -d core/articulate ]; then
    echo "core/articulate already present, skipping"
else
    git clone --depth 1 https://github.com/Xinyu-Yi/TransPose.git "$TMP/TransPose"
    cp -r "$TMP/TransPose/articulate" core/articulate
    echo "installed core/articulate from TransPose"
fi

# 2) Nymeria dataset loader from Meta (CC BY-NC 4.0).
#    Used by preprocess_nymeria_body.py to read Xsens body motion.
#    NOTE: the `nymeria` project on PyPI is an unrelated package by Nymeria LLC.
#    Do not `pip install nymeria`. The loader lives on the legacy branch, because
#    main has since been reorganised around the `nymeriaplus` package.
if [ -d nymeria ]; then
    echo "nymeria already present, skipping"
else
    git clone --depth 1 -b nymeria_dataset_legacy \
        https://github.com/facebookresearch/nymeria_dataset.git "$TMP/nymeria_dataset"
    cp -r "$TMP/nymeria_dataset/nymeria" nymeria
    echo "installed nymeria from nymeria_dataset_legacy"
fi

echo
echo "Done. You still need the SMPL body model at assets/smpl/basicmodel_m.pkl"
echo "(download from https://smpl.is.tue.mpg.de, license required)."
