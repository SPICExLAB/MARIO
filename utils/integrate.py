import torch
import tqdm

from utils import move_to

def integrate(
    integrator,
    loader,
    init,
    device="cpu",
    gtinit=False,
    save_full_traj=False,
    use_gt_rot=True,
):
    """
    Integration function with type consistency using float32
    """
    integrator.eval()

    # —— ensure init state is float32 ——
    init = {
        k: v.to(device).to(torch.float32)
        for k, v in init.items()
    }

    out_state = dict()
    poses, poses_gt = [init["pos"][None, :]], [init["pos"][None, :]]
    orientations, orientations_gt = [init["rot"][None, :]], [init["rot"][None, :]]
    vel, vel_gt = [init["vel"][None, :]], [init["vel"][None, :]]
    covs = [torch.zeros(9, 9, dtype=torch.float32, device=device)]
    
    # Convert integrator to float32
    integrator.to(torch.float32)
    integrator.gravity = integrator.gravity.to(dtype=torch.float32)
    
    for idx, data in tqdm.tqdm(enumerate(loader)):
        data = move_to(data, device)
        
        # Fix dimension mismatch
        if len(data["dt"].shape) < len(data["acc"].shape):
            data["dt"] = data["dt"].unsqueeze(-1)
        
        # Convert all sensor tensors to float32
        for key in ["dt", "gyro", "acc"]:
            data[key] = data[key].to(dtype=torch.float32)
        # Convert rotations to float32
        data["init_rot"] = data.get("init_rot", torch.tensor([], device=device)).to(dtype=torch.float32)
        data["gt_rot"] = data.get("gt_rot", torch.tensor([], device=device)).to(dtype=torch.float32)
        
        if gtinit:
            init_state = {
                "pos": data["init_pos"][:, :1, :].to(dtype=torch.float32),
                "vel": data["init_vel"][:, :1, :].to(dtype=torch.float32),
                "rot": data["init_rot"][:, :1, :].to(dtype=torch.float32),
            }
        else:
            init_state = None

        init_rot = data["init_rot"] if use_gt_rot else None
        
        state = integrator(
            init_state=init_state,
            dt=data["dt"],
            gyro=data["gyro"],
            acc=data["acc"],
            rot=init_rot,
        )

        if save_full_traj:
            vel.append(state["vel"].to(dtype=torch.float32, device=device))
            vel_gt.append(data["gt_vel"].to(dtype=torch.float32, device=device))
            orientations.append(state["rot"].to(dtype=torch.float32, device=device))
            orientations_gt.append(data["gt_rot"].to(dtype=torch.float32, device=device))
            poses_gt.append(data["gt_pos"].to(dtype=torch.float32, device=device))
            poses.append(state["pos"].to(dtype=torch.float32, device=device))
        else:
            vel.append(state["vel"][..., -1:, :].to(dtype=torch.float32, device=device))
            vel_gt.append(data["gt_vel"][..., -1:, :].to(dtype=torch.float32, device=device))
            orientations.append(state["rot"][..., -1:, :].to(dtype=torch.float32, device=device))
            orientations_gt.append(data["gt_rot"][..., -1:, :].to(dtype=torch.float32, device=device))
            poses_gt.append(data["gt_pos"][..., -1:, :].to(dtype=torch.float32, device=device))
            poses.append(state["pos"][..., -1:, :].to(dtype=torch.float32, device=device))

        covs.append(state["cov"][..., -1, :, :].to(device))

    out_state["vel"] = torch.cat(vel, dim=-2)
    out_state["vel_gt"] = torch.cat(vel_gt, dim=-2).to(dtype=torch.float32, device=device)

    out_state["orientations"] = torch.cat(orientations, dim=-2)
    out_state["orientations_gt"] = torch.cat(orientations_gt, dim=-2).to(dtype=torch.float32, device=device)

    out_state["poses"] = torch.cat(poses, dim=-2)
    out_state["poses_gt"] = torch.cat(poses_gt, dim=-2).to(dtype=torch.float32, device=device)

    out_state["covs"] = torch.stack(covs, dim=0)
    
    out_state["pos_dist"] = (
        out_state["poses"][:, 1:, :] - out_state["poses_gt"][:, 1:, :]
    ).norm(dim=-1)
    out_state["vel_dist"] = (
        out_state["vel"][:, 1:, :] - out_state["vel_gt"][:, 1:, :]
    ).norm(dim=-1)
    out_state["rot_dist"] = (
        (
            out_state["orientations_gt"][..., 1:, :].Inv()
            @ out_state["orientations"][..., 1:, :]
        ).Log()
    ).norm(dim=-1)
    return out_state
