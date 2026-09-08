# MARIO: Motion-Augmented Real-Time Multi-Sensor Inertial Odometry

Official code for **MARIO: Motion-Augmented Real-Time Multi-Sensor Inertial Odometry**
(Yiquan Li\*, Taeyoung Yeon\*, Chenfeng Gao, Vasco Xu, Xuanyou Liu, Karan Ahuja).

- Paper: https://arxiv.org/abs/2606.02996
- Project page: https://spice-lab.org/projects/MARIO/

MARIO augments learning-based inertial odometry with (1) **PoseNet**, a frozen
IMU-to-body-pose prior, and (2) a **Multi-Sensor Fusion Module** that adds the
barometer, magnetometer and secondary IMU already available on AR glasses.

This repository is the minimal code needed to preprocess the datasets and to train
and evaluate the AirIO-based MARIO models. It reproduces the AirIO rows of the
paper's tables. The TLIO, EqNIO and RoNIN-LSTM backbones were trained in their own
upstream repositories with the fusion features from this code plugged in.

Trained checkpoints are not distributed. Every model below trains from scratch with
the commands in section 3. Training PoseNet additionally needs two third-party
components that are not redistributed here; see section 3, step 0.

## 1. Setup

```bash
conda create -n mario python=3.10 -y
conda activate mario
pip install -r requirements.txt
export DATA_ROOT=/path/to/processed_data     # every dataset config reads this
```

Expected layout under `$DATA_ROOT` (created by the preprocessing scripts in section 2):

```
processed_nymeria_2imu_both/{train,val,test}/   Nymeria, CPF frame, 200 Hz, dual IMU + baro + mag
aria_processed_cpf_both/{train,val,test,all}/   Aria Everyday Activities, same format
tlio_dataset_cpf/{train,val,test}/              TLIO converted to the CPF frame
processed_data_body_Ty/, processed_data_test_body_Ty/   Nymeria at 50 Hz with body pose (PoseNet only)
```

## 2. Data preprocessing

| Dataset | Command |
|---|---|
| Nymeria (200 Hz, dual IMU, baro, mag) | `python preprocess_nymeria_mag_baro_2imu_both.py <nymeria_root> -o $DATA_ROOT/processed_nymeria_2imu_both --sample-hz 200` |
| Aria Everyday Activities | `python preprocess_aria_mag_baro_2imu_both.py <aea_root> -o $DATA_ROOT/aria_processed_cpf_both --sample-hz 200` |
| TLIO | `python tlio_to_cpf.py <tlio_root>/test --output $DATA_ROOT/tlio_dataset_cpf/test` (same for train / val) |
| Nymeria body pose (PoseNet) | `python preprocess_nymeria_body.py <nymeria_root> -o $DATA_ROOT/processed_data_body_Ty --imu-hz 50 --pose-hz 50` |
| train / val / test split | `python tool_split_dataset.py $DATA_ROOT/processed_nymeria_2imu_both` |

The Nymeria sequences used for training and testing are listed in
`train_sequences.txt` and `test_sequences.txt`. Coordinate conventions
(device -> CPF -> gravity-aligned CPF-world) follow the supplementary material.
Barometric altitude, vertical velocity and magnetometer yaw are derived at load
time in `datasets/nymeriadataset.py`.

## 3. Training

### Step 0: third-party components, only if you train PoseNet

Deriving PoseNet's labels needs SMPL kinematics and Meta's Nymeria loader. Both carry
licenses of their own and are therefore not shipped with MARIO (see `THIRD_PARTY.md`).
Fetch them into place with:

```bash
bash scripts/fetch_third_party.sh
```

This clones the `articulate` toolkit from TransPose into `core/articulate/` and the
Nymeria loader into `nymeria/`. Do not `pip install nymeria`: that name on PyPI belongs
to an unrelated package. Skip this step entirely if you only train and evaluate the
inertial-odometry models; nothing else in the repository imports these two packages.

Two things to know about the Nymeria loader. It expects the standard dataset layout,
with each sequence's body motion at `<sequence>/body/xdata.npz`; if you downloaded head
and body data into separate trees, `preprocess_nymeria_body.py` looks for the body tree
by replacing `head` with `body_motion` in the path. It also needs `pymomentum`, which
must be built against the same PyTorch you are running, or importing it after `torch`
fails with an undefined-symbol error.

### Step 1: PoseNet (required by the `_pose` and `_all` models)

PoseNet is trained once on Nymeria at 50 Hz with gravity removed, then frozen and
reused as a prior. Its labels (SMPL joint positions and 6D joint rotations) are
derived on the fly from the Xsens body pose with the SMPL body model.

1. Download `basicmodel_m.pkl` from https://smpl.is.tue.mpg.de (license required) and
   place it at `assets/smpl/basicmodel_m.pkl`, or set `SMPL_MODEL_PATH`.
2. Preprocess the 50 Hz body-pose data (section 2).
3. Train:

```bash
python train_motion.py --config configs/nymeria/posenet.conf
```

This writes `experiments/posenet/ckpt/best_model.ckpt`, which is where the
`prior.pretrained_path` field of every `_pose` and `_all` config already points.

### Step 2: the odometry models

```bash
python train_motion.py --config configs/nymeria/airio.conf        # single IMU (AirIO baseline)
python train_motion.py --config configs/nymeria/airio_pose.conf   # + PoseNet prior
python train_motion.py --config configs/nymeria/airio_baro.conf   # + barometer
python train_motion.py --config configs/nymeria/airio_mag.conf    # + magnetometer
python train_motion.py --config configs/nymeria/airio_imu2.conf   # + secondary IMU
python train_motion.py --config configs/nymeria/airio_all.conf    # + all of the above
```

Checkpoints go to `experiments/<dataset>/<config name>/ckpt/`. The same six configs
exist for the other datasets as `configs/aria/train_airio*.conf` and
`configs/tlio/airio*.conf`. Logging uses Weights & Biases when you are logged in and
falls back to stdout otherwise.

## 4. Inference and evaluation

```bash
# 1) run the network over the test split (sliding window of 1000 samples)
python inference_motion.py --config configs/nymeria/airio_pose.conf --seqlen 1000 --stepsize 1
# 2) integrate velocities and report ATE, RTE-1s, RTE-5s and drift
python evaluation/evaluate_motion.py --dataconf configs/nymeria/airio_pose.conf \
       --exp experiments/nymeria/airio_pose/ --seqlen 1000 --vis
```

Inference loads `<exp_dir>/ckpt/best_model.ckpt` by default and writes
`net_output.pickle` next to it. Evaluation reads that file and writes
`result/<exp>/result.json`. `RTE_200` and `RTE_1000` are RTE-1s and RTE-5s at 200 Hz.

**Cross-dataset evaluation.** `configs/aria/eval_airio*.conf` run over all 143 Aria
Everyday Activities recordings. Use `--ckpt` to evaluate a Nymeria-trained model on
Aria without fine-tuning:

```bash
python inference_motion.py --config configs/aria/eval_airio_pose.conf \
       --ckpt experiments/nymeria/airio_pose/ckpt/best_model.ckpt --seqlen 1000 --stepsize 1
python evaluation/evaluate_motion.py --dataconf configs/aria/eval_airio_pose.conf \
       --exp experiments/aria/eval_airio_pose/ --seqlen 1000
```

Note that the `all` split of Aria is the union of its train, val and test splits, so it
is only a valid evaluation set for a model that was not trained on Aria.

## 5. Repository layout

```
train_motion.py, inference_motion.py, evaluation/evaluate_motion.py   train / infer / evaluate
preprocess_*.py, tlio_to_cpf.py, tool_split_dataset.py               raw data -> $DATA_ROOT
model/code.py        CodeNetMotionwithRot (AirIO backbone), CodeNetMotionwithPose_v4 (PoseNet),
                     CodeNetMotionwithRot_{Pose,Baro,Mag,LeftIMU,Combined} (fusion variants)
model/losses.py      velocity, covariance and joint losses
datasets/            sequence loaders (nymeria, ariaCPF, tlioCPF), windowing, collate
utils/               integration, dual-IMU preprocessing, plotting
core/paths.py        where to find the SMPL body model
scripts/             fetches the third-party components needed for PoseNet training
configs/datasets/    data locations and loader options
configs/{nymeria,aria,tlio}/   experiment configs
```

## License

MARIO's own code is released under the MIT License (see `LICENSE`). Portions derived
from AirIO remain under BSD-3-Clause, and the components this repository does not
distribute carry licenses of their own; both are documented in `THIRD_PARTY.md`.

## Citation

```bibtex
@article{li2026mario,
  title   = {MARIO: Motion-Augmented Real-Time Multi-Sensor Inertial Odometry},
  author  = {Li, Yiquan and Yeon, Taeyoung and Gao, Chenfeng and Xu, Vasco and Liu, Xuanyou and Ahuja, Karan},
  journal = {arXiv preprint arXiv:2606.02996},
  year    = {2026}
}
```

## Acknowledgements

Built on [AirIO](https://github.com/Air-IO/Air-IO) and [PyPose](https://github.com/pypose/pypose).
Data from [Nymeria](https://github.com/facebookresearch/nymeria_dataset),
[Aria Everyday Activities](https://www.projectaria.com/datasets/aea/) and
[TLIO](https://github.com/CathIAS/TLIO). Third-party components and their licenses are
listed in `THIRD_PARTY.md`.
