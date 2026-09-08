import os
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import pypose as pp
import torch
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

def rotate_trajectory_2d(trajectory, angle_degrees, rotate_around_origin=False):
    """Rotate a 2D trajectory by the specified angle in degrees.
    
    Args:
        trajectory: Input trajectory
        angle_degrees: Rotation angle in degrees
        rotate_around_origin: If True, rotate around (0,0). If False, rotate around trajectory mean (default behavior)
    """
    # Convert angle to radians
    angle_rad = np.radians(angle_degrees)
    
    # Create rotation matrix
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    
    # Convert to numpy for easier manipulation if it's a tensor
    if torch.is_tensor(trajectory):
        is_tensor = True
        device = trajectory.device
        trajectory_np = trajectory.cpu().numpy()
    else:
        is_tensor = False
        trajectory_np = trajectory
    
    # Extract x and y coordinates
    x = trajectory_np[:, 0]
    y = trajectory_np[:, 1]
    
    if rotate_around_origin:
        # Rotate around origin (0,0) - preserves relative positioning
        points = np.column_stack([x, y])
        rotated_points = np.dot(points, rotation_matrix.T)
        rotated_x = rotated_points[:, 0]
        rotated_y = rotated_points[:, 1]
    else:
        # Original behavior: rotate around trajectory mean
        center_x = np.mean(x)
        center_y = np.mean(y)
        
        centered_x = x - center_x
        centered_y = y - center_y
        
        # Combine into a single array for rotation
        points = np.column_stack([centered_x, centered_y])
        
        # Apply rotation
        rotated_points = np.dot(points, rotation_matrix.T)
        
        # Move back to original center
        rotated_x = rotated_points[:, 0] + center_x
        rotated_y = rotated_points[:, 1] + center_y
    
    # Create new trajectory array with rotated x,y and original z
    if trajectory_np.shape[1] > 2:
        rotated_trajectory = np.column_stack([rotated_x, rotated_y, trajectory_np[:, 2:]])
    else:
        rotated_trajectory = np.column_stack([rotated_x, rotated_y])
    
    # Convert back to tensor if input was tensor
    if is_tensor:
        rotated_trajectory = torch.tensor(rotated_trajectory, device=device)
    
    return rotated_trajectory

def calculate_trajectory_error(trajectory1, trajectory2, time_intervals=[50, 100, 200, 500]):
    """Calculate relative trajectory error (RTE) for different time intervals."""
    min_length = min(len(trajectory1), len(trajectory2))
    trajectory1 = trajectory1[:min_length]
    trajectory2 = trajectory2[:min_length]
    
    # Convert to numpy if needed
    if torch.is_tensor(trajectory1):
        trajectory1 = trajectory1.cpu().numpy()
    if torch.is_tensor(trajectory2):
        trajectory2 = trajectory2.cpu().numpy()
    
    total_rte = 0.0
    interval_count = 0
    
    # Calculate RTE for different time intervals
    for duration in time_intervals:
        if duration >= min_length:
            continue
            
        # Calculate relative displacement over time intervals
        dp1 = trajectory1[duration:] - trajectory1[:-duration]  # Predicted trajectory displacements
        dp2 = trajectory2[duration:] - trajectory2[:-duration]  # GT trajectory displacements
        
        # Calculate error between relative displacements
        rte = np.sqrt(np.sum((dp1 - dp2)**2, axis=1))  # Euclidean distance for each interval
        mean_rte = np.mean(rte)
        
        total_rte += mean_rte
        interval_count += 1
    
    # Return average RTE across all time intervals
    if interval_count > 0:
        return total_rte / interval_count
    else:
        # Fallback to simple MSE if no valid intervals
        diff_x = trajectory1[:, 0] - trajectory2[:, 0]
        diff_y = trajectory1[:, 1] - trajectory2[:, 1]
        squared_dist = diff_x**2 + diff_y**2
        return np.mean(squared_dist)

def visualize_3D_motion(save_prefix, save_folder, outstate, infstate, label="AirIO"):
    """Visualize trajectories in 3D space with clean coordinate views."""
    print(f"🎯 Creating 3D visualization for {save_prefix}")
    
    # Extract trajectories
    gt_trajectory = outstate["poses_gt"][0].cpu()
    airio_trajectory = infstate["poses"][0].cpu()
    
    print(f"📊 Trajectory shapes - GT: {gt_trajectory.shape}, AirIO: {airio_trajectory.shape}")
    
    # Handle length mismatch - truncate to shorter length
    min_length = min(len(gt_trajectory), len(airio_trajectory))
    gt_trajectory = gt_trajectory[:min_length]
    airio_trajectory = airio_trajectory[:min_length]
    
    print(f"📊 After truncation - GT: {gt_trajectory.shape}, AirIO: {airio_trajectory.shape}")
    
    # Split coordinates
    gt_x, gt_y, gt_z = torch.split(gt_trajectory, 1, dim=1)
    air_x, air_y, air_z = torch.split(airio_trajectory, 1, dim=1)
    
    # Flatten for plotting
    gt_x, gt_y, gt_z = gt_x.flatten(), gt_y.flatten(), gt_z.flatten()
    air_x, air_y, air_z = air_x.flatten(), air_y.flatten(), air_z.flatten()
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, height_ratios=[2, 1, 1])
    
    # ========== 3D Trajectory Plot ==========
    ax3d = fig.add_subplot(gs[0, 0], projection='3d')
    
    # Plot trajectories
    ax3d.plot(gt_x, gt_y, gt_z, label='Ground Truth', color='green', linewidth=2, alpha=0.8)
    ax3d.plot(air_x, air_y, air_z, label=f'{label}', color='blue', linewidth=2, alpha=0.8)
    
    # Mark start points
    ax3d.scatter(gt_x[0], gt_y[0], gt_z[0], color='green', s=100, marker='o', label='GT Start')
    ax3d.scatter(air_x[0], air_y[0], air_z[0], color='blue', s=100, marker='s', label=f'{label} Start')
    
    ax3d.set_xlabel('X (m)')
    ax3d.set_ylabel('Y (m)')
    ax3d.set_zlabel('Z (m)')
    ax3d.set_title('3D Trajectory Comparison')
    ax3d.legend()
    ax3d.grid(True)
    
    # ========== X-Y Plane View ==========
    ax_xy = fig.add_subplot(gs[0, 1])
    ax_xy.plot(gt_x, gt_y, label='Ground Truth', color='green', linewidth=2, alpha=0.8)
    ax_xy.plot(air_x, air_y, label=f'{label}', color='blue', linewidth=2, alpha=0.8)
    
    # Mark start points
    ax_xy.scatter(gt_x[0], gt_y[0], color='green', s=100, marker='o', label='GT Start')
    ax_xy.scatter(air_x[0], air_y[0], color='blue', s=100, marker='s', label=f'{label} Start')
    
    ax_xy.set_xlabel('X (m)')
    ax_xy.set_ylabel('Y (m)')
    ax_xy.set_title('X-Y Plane View')
    ax_xy.legend()
    ax_xy.grid(True)
    ax_xy.axis('equal')
    
    # ========== Individual Coordinate Time Series ==========
    time_steps = np.arange(len(gt_x))  # Use the actual length after truncation
    
    # X coordinate over time
    ax_x = fig.add_subplot(gs[1, 0])
    ax_x.plot(time_steps, gt_x, label='Ground Truth', color='green', linewidth=2)
    ax_x.plot(time_steps, air_x, label=f'{label}', color='blue', linewidth=2)
    ax_x.set_xlabel('Time Steps')
    ax_x.set_ylabel('X Position (m)')
    ax_x.set_title('X Coordinate vs Time')
    ax_x.legend()
    ax_x.grid(True)
    
    # Y coordinate over time
    ax_y = fig.add_subplot(gs[1, 1])
    ax_y.plot(time_steps, gt_y, label='Ground Truth', color='green', linewidth=2)
    ax_y.plot(time_steps, air_y, label=f'{label}', color='blue', linewidth=2)
    ax_y.set_xlabel('Time Steps')
    ax_y.set_ylabel('Y Position (m)')
    ax_y.set_title('Y Coordinate vs Time')
    ax_y.legend()
    ax_y.grid(True)
    
    # Z coordinate over time
    ax_z = fig.add_subplot(gs[2, :])
    ax_z.plot(time_steps, gt_z, label='Ground Truth', color='green', linewidth=2)
    ax_z.plot(time_steps, air_z, label=f'{label}', color='blue', linewidth=2)
    ax_z.set_xlabel('Time Steps')
    ax_z.set_ylabel('Z Position (m)')
    ax_z.set_title('Z Coordinate vs Time')
    ax_z.legend()
    ax_z.grid(True)
    
    # ========== Calculate and Display Errors ==========
    # Convert to numpy for consistent operations if needed
    if isinstance(gt_x, torch.Tensor):
        gt_x_np, gt_y_np, gt_z_np = gt_x.numpy(), gt_y.numpy(), gt_z.numpy()
        air_x_np, air_y_np, air_z_np = air_x.numpy(), air_y.numpy(), air_z.numpy()
    else:
        gt_x_np, gt_y_np, gt_z_np = gt_x, gt_y, gt_z
        air_x_np, air_y_np, air_z_np = air_x, air_y, air_z
    
    # Calculate trajectory errors
    error_3d = np.sqrt((gt_x_np - air_x_np)**2 + (gt_y_np - air_y_np)**2 + (gt_z_np - air_z_np)**2)
    error_2d = np.sqrt((gt_x_np - air_x_np)**2 + (gt_y_np - air_y_np)**2)
    
    mean_error_3d = np.mean(error_3d)
    mean_error_2d = np.mean(error_2d)
    max_error_3d = np.max(error_3d)
    
    # Add error text to the figure
    error_text = f"""Trajectory Errors:
Mean 3D Error: {mean_error_3d:.3f} m
Mean 2D Error: {mean_error_2d:.3f} m
Max 3D Error: {max_error_3d:.3f} m
Final 3D Error: {error_3d[-1]:.3f} m
Trajectory Length: {len(gt_x)} points"""
    
    fig.text(0.02, 0.02, error_text, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    # Print errors
    print(f"📊 Trajectory Analysis for {save_prefix}:")
    print(f"   Mean 3D error: {mean_error_3d:.3f} m")
    print(f"   Mean 2D error: {mean_error_2d:.3f} m")
    print(f"   Max 3D error: {max_error_3d:.3f} m")
    print(f"   Final 3D error: {error_3d[-1]:.3f} m")
    
    plt.tight_layout()
    
    # Save the plot
    save_path = os.path.join(save_folder, f"{save_prefix}_3D_motion.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved 3D visualization to: {save_path}")
    plt.close()
    
    return mean_error_3d, mean_error_2d, max_error_3d

def visualize_3D_motion_with_xz(save_prefix, save_folder, outstate, infstate, label="AirIO"):
    """Visualize trajectories in 3D space with Y-up coordinate system including X-Z view."""
    print(f"🎯 Creating 3D visualization with X-Z view for {save_prefix}")
    
    # Extract trajectories
    # gt_trajectory = outstate["poses_gt"][0].cpu()
    gt_trajectory = infstate["poses_gt"][0].cpu()
    airio_trajectory = infstate["poses"][0].cpu()
    
    # Handle length mismatch
    min_length = min(len(gt_trajectory), len(airio_trajectory))
    gt_trajectory = gt_trajectory[:min_length]
    airio_trajectory = airio_trajectory[:min_length]
    
    print(f"📊 Trajectory shapes - GT: {gt_trajectory.shape}, AirIO: {airio_trajectory.shape}")
    
    # Split coordinates
    gt_x, gt_y, gt_z = torch.split(gt_trajectory, 1, dim=1)
    air_x, air_y, air_z = torch.split(airio_trajectory, 1, dim=1)
    
    # Flatten for plotting
    gt_x, gt_y, gt_z = gt_x.flatten(), gt_y.flatten(), gt_z.flatten()
    air_x, air_y, air_z = air_x.flatten(), air_y.flatten(), air_z.flatten()
    
    # Create figure with subplots
    fig = plt.figure(figsize=(24, 16))
    gs = GridSpec(4, 3, height_ratios=[2, 2, 1.5, 1.5])
    
    # ========== 3D Trajectory Plot ==========
    ax3d = fig.add_subplot(gs[0, 0], projection='3d')
    
    ax3d.plot(gt_x, gt_y, gt_z, label='Ground Truth', color='green', linewidth=2, alpha=0.8)
    ax3d.plot(air_x, air_y, air_z, label=f'{label}', color='blue', linewidth=2, alpha=0.8)
    
    ax3d.scatter(gt_x[0], gt_y[0], gt_z[0], color='green', s=100, marker='o', label='GT Start')
    ax3d.scatter(air_x[0], air_y[0], air_z[0], color='blue', s=100, marker='s', label=f'{label} Start')
    
    ax3d.set_xlabel('X (m)')
    ax3d.set_ylabel('Y (m) - UP')
    ax3d.set_zlabel('Z (m)')
    ax3d.set_title('3D Trajectory (Y-up World)')
    ax3d.legend()
    ax3d.grid(True)
    
    # ========== X-Z Plane View (HORIZONTAL in Y-up) ==========
    ax_xz = fig.add_subplot(gs[0, 1])
    ax_xz.plot(gt_x, gt_z, label='Ground Truth', color='green', linewidth=2, alpha=0.8)
    ax_xz.plot(air_x, air_z, label=f'{label}', color='blue', linewidth=2, alpha=0.8)
    
    ax_xz.scatter(gt_x[0], gt_z[0], color='green', s=100, marker='o')
    ax_xz.scatter(air_x[0], air_z[0], color='blue', s=100, marker='s')
    
    ax_xz.set_xlabel('X (m)')
    ax_xz.set_ylabel('Z (m)')
    ax_xz.set_title('X-Z Plane View')
    ax_xz.legend()
    ax_xz.grid(True)
    ax_xz.axis('equal')
    ax_xz.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax_xz.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    # ========== X-Y Plane View (Side view) ==========
    ax_xy = fig.add_subplot(gs[0, 2])
    ax_xy.plot(gt_x, gt_y, label='Ground Truth', color='green', linewidth=2, alpha=0.8)
    ax_xy.plot(air_x, air_y, label=f'{label}', color='blue', linewidth=2, alpha=0.8)
    
    ax_xy.scatter(gt_x[0], gt_y[0], color='green', s=100, marker='o')
    ax_xy.scatter(air_x[0], air_y[0], color='blue', s=100, marker='s')
    
    ax_xy.set_xlabel('X (m)')
    ax_xy.set_ylabel('Y (m) - Vertical')
    ax_xy.set_title('X-Y Plane View')
    ax_xy.legend()
    ax_xy.grid(True)
    ax_xy.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # ========== Y-Z Plane View ==========
    ax_yz = fig.add_subplot(gs[1, 0])
    ax_yz.plot(gt_z, gt_y, label='Ground Truth', color='green', linewidth=2, alpha=0.8)
    ax_yz.plot(air_z, air_y, label=f'{label}', color='blue', linewidth=2, alpha=0.8)
    
    ax_yz.set_xlabel('Z (m)')
    ax_yz.set_ylabel('Y (m) - Vertical')
    ax_yz.set_title('Y-Z Plane View (Front - Vertical Motion)')
    ax_yz.legend()
    ax_yz.grid(True)
    ax_yz.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # ========== DIRECT VELOCITY COMPARISON (FROM MODEL) ==========
    if 'net_vel' in infstate and 'vel_gt' in outstate:
        ax_vel_direct = fig.add_subplot(gs[1, 1:])
        
        # Get velocities directly from model output (downsample for visibility)
        step = 50  # Show every 50th sample
        gt_vel_direct = outstate['vel_gt'][0][::step].cpu().numpy()
        pred_vel_direct = infstate['net_vel'][0][::step].cpu().numpy()
        time_steps_vel = np.arange(len(gt_vel_direct)) * (step * 0.02)  # 50Hz base
        
        # Plot all three components
        ax_vel_direct.plot(time_steps_vel, gt_vel_direct[:, 0], 'g-', linewidth=2, alpha=0.8, label='GT X vel')
        ax_vel_direct.plot(time_steps_vel, gt_vel_direct[:, 1], 'g--', linewidth=2, alpha=0.8, label='GT Y vel')
        ax_vel_direct.plot(time_steps_vel, gt_vel_direct[:, 2], 'g:', linewidth=2, alpha=0.8, label='GT Z vel')
        
        ax_vel_direct.plot(time_steps_vel, pred_vel_direct[:, 0], 'b-', linewidth=1.5, alpha=0.7, label='Pred X vel')
        ax_vel_direct.plot(time_steps_vel, pred_vel_direct[:, 1], 'b--', linewidth=1.5, alpha=0.7, label='Pred Y vel')
        ax_vel_direct.plot(time_steps_vel, pred_vel_direct[:, 2], 'b:', linewidth=1.5, alpha=0.7, label='Pred Z vel')
        
        ax_vel_direct.set_xlabel('Time (s)')
        ax_vel_direct.set_ylabel('Velocity (m/s)')
        ax_vel_direct.set_title('Direct Velocity Comparison (Model Output vs GT)')
        ax_vel_direct.legend(ncol=2, loc='upper right')
        ax_vel_direct.grid(True, alpha=0.3)
        
        # Add zero line
        ax_vel_direct.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # ========== VELOCITY COMPONENTS (SEPARATE PLOTS) ==========
    time_steps = np.arange(len(gt_x)) * 0.02  # 50Hz
    
    # Calculate velocities from positions
    dt = 0.02  # 50Hz
    gt_vel_from_pos = np.diff(gt_trajectory.numpy(), axis=0) / dt
    air_vel_from_pos = np.diff(airio_trajectory.numpy(), axis=0) / dt
    time_steps_diff = time_steps[:-1]  # One less due to diff
    
    # X Velocity
    ax_vx = fig.add_subplot(gs[2, 0])
    ax_vx.plot(time_steps_diff, gt_vel_from_pos[:, 0], 'g-', linewidth=2, alpha=0.8, label='GT X vel')
    ax_vx.plot(time_steps_diff, air_vel_from_pos[:, 0], 'b-', linewidth=1.5, alpha=0.7, label='Pred X vel')
    ax_vx.set_xlabel('Time (s)')
    ax_vx.set_ylabel('X Velocity (m/s)')
    ax_vx.set_title('X Velocity Component')
    ax_vx.legend()
    ax_vx.grid(True, alpha=0.3)
    ax_vx.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Y Velocity (Vertical)
    ax_vy = fig.add_subplot(gs[2, 1])
    ax_vy.plot(time_steps_diff, gt_vel_from_pos[:, 1], 'g-', linewidth=2, alpha=0.8, label='GT Y vel')
    ax_vy.plot(time_steps_diff, air_vel_from_pos[:, 1], 'b-', linewidth=1.5, alpha=0.7, label='Pred Y vel')
    ax_vy.set_xlabel('Time (s)')
    ax_vy.set_ylabel('Y Velocity (m/s) - Vertical')
    ax_vy.set_title('Y Velocity Component (Vertical)')
    ax_vy.legend()
    ax_vy.grid(True, alpha=0.3)
    ax_vy.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Z Velocity
    ax_vz = fig.add_subplot(gs[2, 2])
    ax_vz.plot(time_steps_diff, gt_vel_from_pos[:, 2], 'g-', linewidth=2, alpha=0.8, label='GT Z vel')
    ax_vz.plot(time_steps_diff, air_vel_from_pos[:, 2], 'b-', linewidth=1.5, alpha=0.7, label='Pred Z vel')
    ax_vz.set_xlabel('Time (s)')
    ax_vz.set_ylabel('Z Velocity (m/s)')
    ax_vz.set_title('Z Velocity Component')
    ax_vz.legend()
    ax_vz.grid(True, alpha=0.3)
    ax_vz.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # ========== POSITION COMPONENTS ==========
    # X coordinate over time
    ax_x = fig.add_subplot(gs[3, 0])
    ax_x.plot(time_steps, gt_x, 'g-', linewidth=2, label='GT')
    ax_x.plot(time_steps, air_x, 'b-', linewidth=1.5, label='Pred')
    ax_x.set_xlabel('Time (s)')
    ax_x.set_ylabel('X Position (m)')
    ax_x.set_title('X Position vs Time')
    ax_x.legend()
    ax_x.grid(True, alpha=0.3)
    
    # Y coordinate over time (VERTICAL)
    ax_y = fig.add_subplot(gs[3, 1])
    ax_y.plot(time_steps, gt_y, 'g-', linewidth=2, label='GT')
    ax_y.plot(time_steps, air_y, 'b-', linewidth=1.5, label='Pred')
    ax_y.set_xlabel('Time (s)')
    ax_y.set_ylabel('Y Position (m) - Vertical')
    ax_y.set_title('Y Position vs Time (Vertical)')
    ax_y.legend()
    ax_y.grid(True, alpha=0.3)
    ax_y.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Z coordinate over time
    ax_z = fig.add_subplot(gs[3, 2])
    ax_z.plot(time_steps, gt_z, 'g-', linewidth=2, label='GT')
    ax_z.plot(time_steps, air_z, 'b-', linewidth=1.5, label='Pred')
    ax_z.set_xlabel('Time (s)')
    ax_z.set_ylabel('Z Position (m)')
    ax_z.set_title('Z Position vs Time')
    ax_z.legend()
    ax_z.grid(True, alpha=0.3)
    
    # ========== Simple Summary Stats ==========
    gt_x_np, gt_y_np, gt_z_np = gt_x.numpy(), gt_y.numpy(), gt_z.numpy()
    air_x_np, air_y_np, air_z_np = air_x.numpy(), air_y.numpy(), air_z.numpy()
    
    # Calculate basic stats
    final_gt = np.array([gt_x_np[-1], gt_y_np[-1], gt_z_np[-1]])
    final_pred = np.array([air_x_np[-1], air_y_np[-1], air_z_np[-1]])
    final_error = np.linalg.norm(final_pred - final_gt)
    
    gt_displacement = np.linalg.norm(final_gt - np.array([gt_x_np[0], gt_y_np[0], gt_z_np[0]]))
    pred_displacement = np.linalg.norm(final_pred - np.array([air_x_np[0], air_y_np[0], air_z_np[0]]))
    
    # Simple text summary
    summary_text = f"""Final Positions:
GT:   [{final_gt[0]:.2f}, {final_gt[1]:.2f}, {final_gt[2]:.2f}] m
Pred: [{final_pred[0]:.2f}, {final_pred[1]:.2f}, {final_pred[2]:.2f}] m
Error: {final_error:.2f} m

Total Displacement:
GT:   {gt_displacement:.2f} m
Pred: {pred_displacement:.2f} m"""
    
    fig.text(0.02, 0.02, summary_text, fontsize=11, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Print key issues
    print(f"\n📊 Key Issues:")
    print(f"   Final error: {final_error:.2f}m")
    print(f"   GT moved: {gt_displacement:.2f}m, Pred moved: {pred_displacement:.2f}m")
    print(f"   Y (vertical) drift: {final_pred[1] - final_gt[1]:.2f}m")
    
    plt.tight_layout()
    
    # Save the plot
    save_path = os.path.join(save_folder, f"{save_prefix}_3D_motion_xz_view.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved 3D visualization to: {save_path}")
    plt.close()
    
    return final_error, gt_displacement, pred_displacement

def visualize_motion(save_prefix, save_folder, outstate, infstate, label="AirIO"):
    ### visualize gt&netoutput velocity, 2d trajectory. 
    gt_x, gt_y, gt_z = torch.split(outstate["poses_gt"][0].cpu(), 1, dim=1)
    airTraj_x, airTraj_y, airTraj_z = torch.split(infstate["poses"][0].cpu(), 1, dim=1)
    
    v_gt_x, v_gt_y, v_gt_z = torch.split(outstate['vel_gt'][0][::50,:].cpu(), 1, dim=1)
    airVel_x, airVel_y, airVel_z = torch.split(infstate['net_vel'][0][::50,:].cpu(), 1, dim=1)
    
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(3, 2) 

    ax1 = fig.add_subplot(gs[:, 0]) 
    ax2 = fig.add_subplot(gs[0, 1]) 
    ax3 = fig.add_subplot(gs[1, 1]) 
    ax4 = fig.add_subplot(gs[2, 1]) 
   
    #visualize traj 
    ax1.plot(airTraj_x, airTraj_y, label=label)
    ax1.plot(gt_x, gt_y, label="Ground Truth")
    ax1.set_xlabel('X axis')
    ax1.set_ylabel('Y axis')
    ax1.legend()
    
    #visualize vel
    ax2.plot(airVel_x, label=label)
    ax2.plot(v_gt_x, label="Ground Truth",linestyle='--')
    
    ax3.plot(airVel_y, label=label)
    ax3.plot(v_gt_y, label="Ground Truth",linestyle='--')
    
    ax4.plot(airVel_z, label=label)
    ax4.plot(v_gt_z, label="Ground Truth",linestyle='--')
    
    ax2.set_xlabel('time')
    ax2.set_ylabel('velocity')
    ax2.legend()
    ax3.legend()
    ax4.legend()
    save_prefix += "_state.png"
    plt.savefig(os.path.join(save_folder, save_prefix), dpi=300)
    plt.close()

def visualize_rotated_motion(save_prefix, save_folder, outstate, infstate, label="AirIO", debug=False):
    """Visualize ground truth and rotated AirIO trajectories to find best match."""
    # Use infstate for both trajectories to ensure they start at the same point
    gt_trajectory = infstate["poses_gt"][0].cpu() if "poses_gt" in infstate else outstate["poses_gt"][0].cpu()
    airio_trajectory = infstate["poses"][0].cpu()
    
    # DEBUG: Check what data we're getting
    if debug:
        print("\n🔍 VISUALIZATION DEBUG - visualize_rotated_motion:")
        print(f"   GT trajectory shape: {gt_trajectory.shape}")
        print(f"   GT first 3 positions:\n{gt_trajectory[:3]}")
        print(f"   GT last 3 positions:\n{gt_trajectory[-3:]}")
        print(f"   GT start position: {gt_trajectory[0]}")
        print(f"   GT end position: {gt_trajectory[-1]}")
        print(f"   \n   AirIO trajectory shape: {airio_trajectory.shape}")
        print(f"   AirIO first 3 positions:\n{airio_trajectory[:3]}")
        print(f"   AirIO last 3 positions:\n{airio_trajectory[-3:]}")
        print(f"   AirIO start position: {airio_trajectory[0]}")
        print(f"   AirIO end position: {airio_trajectory[-1]}")
    
    # Try different rotation angles
    angles = [0, 90, 180, 270]
    errors = []
    rotated_trajectories = []
    
    for angle in angles:
        rotated_traj = rotate_trajectory_2d(airio_trajectory, angle)
        error = calculate_trajectory_error(rotated_traj, gt_trajectory)
        errors.append(error)
        rotated_trajectories.append(rotated_traj)
    
    # Find the best rotation angle
    best_idx = np.argmin(errors)
    best_angle = angles[best_idx]
    best_rotated_traj = rotated_trajectories[best_idx]
    
    # Print the results
    print(f"Rotation errors: {errors}")
    print(f"Best rotation angle: {best_angle} degrees with error: {errors[best_idx]:.4f}")
    
    # Split trajectories for plotting
    gt_x, gt_y, gt_z = torch.split(gt_trajectory, 1, dim=1)
    best_x, best_y, best_z = torch.split(best_rotated_traj, 1, dim=1)
    
    # Extract velocities for comparison
    v_gt_x, v_gt_y, v_gt_z = torch.split(outstate['vel_gt'][0][::50,:].cpu(), 1, dim=1)
    airVel_x, airVel_y, airVel_z = torch.split(infstate['net_vel'][0][::50,:].cpu(), 1, dim=1)
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(5, 2)
    
    # Main plot with original and rotated trajectories
    ax1 = fig.add_subplot(gs[:3, 0])
    ax1.plot(gt_x, gt_y, label="Ground Truth", color='green')
    ax1.plot(best_x, best_y, label=f"{label} (Rotated {best_angle}°)", color='blue')
    ax1.set_xlabel('X axis')
    ax1.set_ylabel('Y axis')
    ax1.set_title('Trajectory Comparison with Best Rotation')
    ax1.legend()
    ax1.grid(True)
    
    # Plot with all rotation angles
    ax5 = fig.add_subplot(gs[3:, 0])
    ax5.plot(gt_x, gt_y, label="Ground Truth", color='green')
    
    for i, angle in enumerate(angles):
        if angle == best_angle:
            continue  # Skip the best angle as it's already in the main plot
        rotated_x, rotated_y, _ = torch.split(rotated_trajectories[i], 1, dim=1)
        ax5.plot(rotated_x, rotated_y, label=f"{label} ({angle}°)", alpha=0.5)
    
    ax5.set_xlabel('X axis')
    ax5.set_ylabel('Y axis')
    ax5.set_title('All Rotation Angles')
    ax5.legend()
    ax5.grid(True)
    
    # Velocity plots
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, 1])
    
    ax2.plot(airVel_x, label=label)
    ax2.plot(v_gt_x, label="Ground Truth",linestyle='--')
    ax3.plot(airVel_y, label=label)
    ax3.plot(v_gt_y, label="Ground Truth",linestyle='--')
    ax4.plot(airVel_z, label=label)
    ax4.plot(v_gt_z, label="Ground Truth",linestyle='--')
    
    ax2.set_title('X Velocity')
    ax3.set_title('Y Velocity')
    ax4.set_title('Z Velocity')
    
    ax2.legend()
    ax3.legend()
    ax4.legend()
    
    # Error plot
    ax6 = fig.add_subplot(gs[3:, 1])
    ax6.bar([str(a)+"°" for a in angles], errors)
    ax6.set_xlabel('Rotation Angle (degrees)')
    ax6.set_ylabel('Mean Squared Error')
    ax6.set_title('Error for Each Rotation Angle')
    ax6.grid(True)
    
    plt.tight_layout()
    save_path = os.path.join(save_folder, f"{save_prefix}_rotated_{best_angle}deg.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.savefig(save_path, dpi=300)
    print(f"Saved visualization to {save_path}")
    plt.close()
    
    return best_angle, errors[best_idx]

def visualize_rotations(save_prefix, gt_rot, out_rot, inf_rot=None, save_folder=None):
    gt_euler = np.unwrap(pp.SO3(gt_rot).euler(), axis=0, discont=np.pi/2) * 180.0 / np.pi
    outstate_euler = np.unwrap(pp.SO3(out_rot).euler(), axis=0, discont=np.pi/2) * 180.0 / np.pi

    legend_list = ["roll", "pitch","yaw"]
    fig, axs = plt.subplots(3)
    fig.suptitle("integrated orientation")
    for i in range(3):
        axs[i].plot(outstate_euler[:, i], color="b", linewidth=0.9)
        axs[i].plot(gt_euler[:, i], color="mediumseagreen", linewidth=0.9)
        axs[i].legend(["raw_" + legend_list[i], "gt_" + legend_list[i]])
        axs[i].grid(True)

    if inf_rot is not None:
        infstate_euler = np.unwrap(pp.SO3(inf_rot).euler(), axis=0, discont=np.pi/2) * 180.0 / np.pi
        for i in range(3):
            axs[i].plot(infstate_euler[:, i], color="red", linewidth=0.9)
            axs[i].legend(
                [
                    "raw_" + legend_list[i],
                    "gt_" + legend_list[i],
                    "AirIMU_" + legend_list[i],
                ]
            )
    plt.tight_layout()
    if save_folder is not None:
        plt.savefig(
            os.path.join(save_folder, save_prefix + "_orientation_compare.png"), dpi=300
        )
    plt.close()
    
def visualize_velocity(save_prefix, gtstate, outstate, refstate=None, save_folder=None):
    legend_list = ["x", "y", "z"]
    fig, axs = plt.subplots(
        3,
    )
    fig.suptitle("Velocity Comparison")
    for i in range(3):
        axs[i].plot(outstate[:, i], color="b", linewidth=0.9)
        axs[i].plot(gtstate[:, i], color="mediumseagreen", linewidth=0.9)
        axs[i].legend(["AirIO_" + legend_list[i], "gt_" + legend_list[i]])
        axs[i].grid(True)
    
    if refstate is not None:
        for i in range(3):
            axs[i].plot(refstate[:, i], color="red", linewidth=0.9)
            axs[i].legend(
                [
                "AirIO_" + legend_list[i], 
                "gt_" + legend_list[i],
                "IOnet" + legend_list[i],
                ]
            )

    plt.tight_layout()
    if save_folder is not None:
        plt.savefig(
            os.path.join(save_folder, save_prefix + ".png"), dpi=300
        )
    plt.show()
    plt.close()


def scale_trajectory(trajectory, scale_factor):
    """Scale a trajectory by the specified factor."""
    if torch.is_tensor(trajectory):
        trajectory_np = trajectory.cpu().numpy()
        is_tensor = True
        device = trajectory.device
    else:
        trajectory_np = trajectory
        is_tensor = False
    
    scaled_trajectory = trajectory_np.copy()
    scaled_trajectory[:, :2] = trajectory_np[:, :2] * scale_factor
    
    if is_tensor:
        scaled_trajectory = torch.tensor(scaled_trajectory, device=device)
    
    return scaled_trajectory

def normalize_trajectories_to_origin(gt_trajectory, pred_trajectory):
    """Normalize both trajectories to start at (0,0)."""
    if torch.is_tensor(gt_trajectory):
        gt_np = gt_trajectory.cpu().numpy()
        pred_np = pred_trajectory.cpu().numpy()
        is_tensor = True
        device = gt_trajectory.device
    else:
        gt_np = gt_trajectory
        pred_np = pred_trajectory
        is_tensor = False
    
    # Use GT starting point as reference for both trajectories
    gt_start = gt_np[0, :2].copy()
    
    gt_normalized = gt_np.copy()
    pred_normalized = pred_np.copy()
    
    # Normalize both trajectories using GT's starting point
    gt_normalized[:, :2] -= gt_start
    pred_normalized[:, :2] -= gt_start
    
    if is_tensor:
        gt_normalized = torch.tensor(gt_normalized, device=device)
        pred_normalized = torch.tensor(pred_normalized, device=device)
    
    return gt_normalized, pred_normalized

def calculate_trajectory_error(trajectory1, trajectory2, time_intervals=[50, 100, 200, 500]):
    """Calculate relative trajectory error (RTE) for different time intervals."""
    min_length = min(len(trajectory1), len(trajectory2))
    trajectory1 = trajectory1[:min_length]
    trajectory2 = trajectory2[:min_length]
    
    # Convert to numpy if needed
    if torch.is_tensor(trajectory1):
        trajectory1 = trajectory1.cpu().numpy()
    if torch.is_tensor(trajectory2):
        trajectory2 = trajectory2.cpu().numpy()
    
    total_rte = 0.0
    interval_count = 0
    
    # Calculate RTE for different time intervals
    for duration in time_intervals:
        if duration >= min_length:
            continue
            
        # Calculate relative displacement over time intervals
        dp1 = trajectory1[duration:] - trajectory1[:-duration]  # Predicted trajectory displacements
        dp2 = trajectory2[duration:] - trajectory2[:-duration]  # GT trajectory displacements
        
        # Calculate error between relative displacements
        rte = np.sqrt(np.sum((dp1 - dp2)**2, axis=1))  # Euclidean distance for each interval
        mean_rte = np.mean(rte)
        
        total_rte += mean_rte
        interval_count += 1
    
    # Return average RTE across all time intervals
    if interval_count > 0:
        return total_rte / interval_count
    else:
        # Fallback to simple MSE if no valid intervals
        diff_x = trajectory1[:, 0] - trajectory2[:, 0]
        diff_y = trajectory1[:, 1] - trajectory2[:, 1]
        squared_dist = diff_x**2 + diff_y**2
        return np.mean(squared_dist)

def optimize_trajectory_alignment(gt_trajectory, pred_trajectory, 
                    rotation_angles=[0, 45, 90, 135, 180, 225, 270, 315],
                    scale_factors=[0.1, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]):
    """Find optimal rotation and scaling for trajectory alignment."""
    best_error = float('inf')
    best_angle = 0
    best_scale = 1.0
    
    for angle in rotation_angles:
        for scale in scale_factors:
            scaled_traj = scale_trajectory(pred_trajectory, scale)
            rotated_traj = rotate_trajectory_2d(scaled_traj, angle)
            
            # Normalize both trajectories to start at origin before calculating error
            gt_normalized, pred_normalized = normalize_trajectories_to_origin(gt_trajectory, rotated_traj)
            error = calculate_trajectory_error(pred_normalized, gt_normalized)
            
            if error < best_error:
                best_error = error
                best_angle = angle
                best_scale = scale
    
    return best_angle, best_scale, best_error

class EfficientTrajectoryAnimator:
    def __init__(self, gt_trajectory, pred_trajectory, save_path, 
                 rotation_angle=None, scale_factor=None, 
                 imu_sample_rate=100, fps=30, skip_seconds=0, skip_end_seconds=0,
                 speed_multiplier=1):
        
        self.save_path = save_path
        self.fps = fps
        self.imu_sample_rate = imu_sample_rate
        
        # Skip first N seconds of data
        skip_start_points = int(skip_seconds * imu_sample_rate)
        skip_end_points = int(skip_end_seconds * imu_sample_rate)
        
        if skip_start_points > 0:
            gt_trajectory = gt_trajectory[skip_start_points:]
            pred_trajectory = pred_trajectory[skip_start_points:]
            print(f"⏭️  Skipped first {skip_seconds}s ({skip_start_points} points)")
        
        # Skip last N seconds of data
        if skip_end_points > 0:
            gt_trajectory = gt_trajectory[:-skip_end_points]
            pred_trajectory = pred_trajectory[:-skip_end_points]
            print(f"⏮️  Skipped last {skip_end_seconds}s ({skip_end_points} points)")
        
        # Normalize both trajectories to start at (0,0)
        self.gt_trajectory, pred_normalized = normalize_trajectories_to_origin(gt_trajectory, pred_trajectory)
        
        # Convert to numpy for easier processing
        if torch.is_tensor(self.gt_trajectory):
            self.gt_trajectory = self.gt_trajectory.cpu().numpy()
            pred_normalized = pred_normalized.cpu().numpy()
        
        # ROTATE THE VIEW: Swap X and Y coordinates for vertical display
        self.gt_trajectory = self.rotate_view(self.gt_trajectory)
        pred_normalized = self.rotate_view(pred_normalized)
        
        # Calculate real video duration from trajectory length and IMU sample rate
        min_length = min(len(self.gt_trajectory), len(pred_normalized))
        self.real_duration = min_length / imu_sample_rate
        self.video_duration = self.real_duration / speed_multiplier  # Faster video
        self.total_video_frames = int(self.video_duration * fps)
        
        print(f"📊 Trajectory length: {min_length} points (after trimming)")
        print(f"⏱️  Real duration: {self.real_duration:.1f}s at {imu_sample_rate}Hz")
        print(f"🎬 Video: {self.video_duration:.1f}s ({self.total_video_frames} frames at {fps} fps, {speed_multiplier}x speed)")
        
        # Determine rotation and scale
        if rotation_angle is not None and scale_factor is not None:
            self.rotation_angle = rotation_angle
            self.scale_factor = scale_factor
            print(f"🎯 Using fixed parameters: Rotation={rotation_angle}°, Scale={scale_factor:.3f}")
        else:
            print("🔍 Optimizing trajectory alignment...")
            self.rotation_angle, self.scale_factor, error = optimize_trajectory_alignment(
                self.gt_trajectory, pred_normalized
            )
            print(f"✅ Optimized alignment: Rotation={self.rotation_angle}°, Scale={self.scale_factor:.3f}, Error={error:.4f}")
        
        # Apply transformations to prediction
        scaled_pred = scale_trajectory(pred_normalized, self.scale_factor)
        self.pred_trajectory = rotate_trajectory_2d(scaled_pred, self.rotation_angle)
        
        if torch.is_tensor(self.pred_trajectory):
            self.pred_trajectory = self.pred_trajectory.cpu().numpy()
        
        # Ensure same length
        self.gt_trajectory = self.gt_trajectory[:min_length]
        self.pred_trajectory = self.pred_trajectory[:min_length]
        
        # Create mapping from video frames to trajectory indices
        self.frame_to_traj_mapping = []
        for frame in range(self.total_video_frames):
            progress = frame / (self.total_video_frames - 1) if self.total_video_frames > 1 else 0
            traj_idx = int(progress * (min_length - 1))
            self.frame_to_traj_mapping.append(traj_idx)
        
        self.setup_animation()
    
    def rotate_view(self, trajectory):
        """Swap X and Y coordinates for vertical display."""
        rotated = trajectory.copy()
        rotated[:, 0] = trajectory[:, 1]   # New X = old Y 
        rotated[:, 1] = -trajectory[:, 0]   # New Y = -old X
        return rotated
    
    def setup_animation(self):
        """Setup the animation figure showing the FULL trajectory range."""
        # Use dark style for modern look
        plt.style.use('dark_background')
        
        # Standard figure size with black background
        self.fig, self.ax = plt.subplots(figsize=(12, 10), facecolor='black')
        self.ax.set_facecolor('black')
        
        # Remove margins for maximum space utilization
        self.fig.subplots_adjust(left=0.05, right=0.98, top=0.98, bottom=0.08)
        
        # Modern color palette
        self.gt_color = '#00FF7F'      # Spring green
        self.pred_color = '#1E90FF'    # Dodger blue
        self.current_gt_color = '#32CD32'    # Lime green
        self.current_pred_color = '#FF6347'  # Tomato
        
        self.ax.set_aspect('equal')
        
        # Subtle grid
        self.ax.grid(True, alpha=0.2, color='gray', linewidth=0.5)
        
        # Clean axis styling
        self.ax.set_xlabel('X (m)', fontsize=14, color='white', fontweight='light')
        self.ax.set_ylabel('Y (m)', fontsize=14, color='white', fontweight='light')
        
        # Modern axis styling
        self.ax.spines['bottom'].set_color('gray')
        self.ax.spines['top'].set_color('gray') 
        self.ax.spines['right'].set_color('gray')
        self.ax.spines['left'].set_color('gray')
        self.ax.spines['bottom'].set_linewidth(0.5)
        self.ax.spines['top'].set_linewidth(0.5)
        self.ax.spines['right'].set_linewidth(0.5)
        self.ax.spines['left'].set_linewidth(0.5)
        
        # Tick styling
        self.ax.tick_params(colors='gray', which='both', labelsize=11)
        
        # Initialize sleek trajectory lines
        self.gt_line, = self.ax.plot([], [], color=self.gt_color, linewidth=2.5, 
                                    alpha=0.9, label='Ground Truth')
        self.pred_line, = self.ax.plot([], [], color=self.pred_color, linewidth=2.5, 
                                      alpha=0.9, label='Prediction')
        
        # Modern current position markers
        self.gt_point, = self.ax.plot([], [], 'o', color=self.current_gt_color, 
                                     markersize=10, markeredgewidth=2, 
                                     markeredgecolor='white', zorder=10, alpha=0.9)
        self.pred_point, = self.ax.plot([], [], 's', color=self.current_pred_color, 
                                       markersize=10, markeredgewidth=2, 
                                       markeredgecolor='white', zorder=10, alpha=0.9)
        
        # Subtle legend
        legend = self.ax.legend(fontsize=12, loc='upper right', 
                               frameon=True, fancybox=True, shadow=True,
                               facecolor='black', edgecolor='gray', framealpha=0.8)
        legend.get_frame().set_linewidth(0.5)
        for text in legend.get_texts():
            text.set_color('white')
        
        # Calculate axis limits to show COMPLETE trajectory
        all_x = np.concatenate([self.gt_trajectory[:, 0], self.pred_trajectory[:, 0]])
        all_y = np.concatenate([self.gt_trajectory[:, 1], self.pred_trajectory[:, 1]])
        
        # Get full range of data
        x_min, x_max = np.min(all_x), np.max(all_x)
        y_min, y_max = np.min(all_y), np.max(all_y)
        
        # Add margin to ensure nothing is cut off
        margin_x = 0.05 * np.ptp(all_x)  # 5% of X range
        margin_y = 0.05 * np.ptp(all_y)  # 5% of Y range
        
        # x_min -= margin_x
        # x_max += margin_x
        y_min -= margin_y
        y_max += margin_y
        
        x_range = np.ptp(all_x)
        extra_x_margin = 0.3 * x_range  # 30% extra width on each side
        x_min -= extra_x_margin
        x_max += extra_x_margin

        # SET LIMITS TO SHOW COMPLETE TRAJECTORY
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        
        print(f"📊 Full trajectory range:")
        print(f"   X: {np.min(all_x):.2f} to {np.max(all_x):.2f}")
        print(f"   Y: {np.min(all_y):.2f} to {np.max(all_y):.2f}")
        print(f"📊 Axis limits: X({x_min:.2f} to {x_max:.2f}), Y({y_min:.2f} to {y_max:.2f})")
        
        # No title for clean look
        self.ax.set_title('')
    
    def animate(self, video_frame):
        """Animation function with smooth trajectory building."""
        traj_idx = self.frame_to_traj_mapping[video_frame]
        
        # Progressive trajectory building with smooth lines
        if traj_idx > 0:
            # Main trajectory lines
            self.gt_line.set_data(self.gt_trajectory[:traj_idx+1, 0], 
                                 self.gt_trajectory[:traj_idx+1, 1])
            self.pred_line.set_data(self.pred_trajectory[:traj_idx+1, 0], 
                                   self.pred_trajectory[:traj_idx+1, 1])
            
            # Current position markers with smooth update
            self.gt_point.set_data([self.gt_trajectory[traj_idx, 0]], 
                                  [self.gt_trajectory[traj_idx, 1]])
            self.pred_point.set_data([self.pred_trajectory[traj_idx, 0]], 
                                    [self.pred_trajectory[traj_idx, 1]])
        
        return self.gt_line, self.pred_line, self.gt_point, self.pred_point
    
    def create_animation(self, interval=None):
        """Create smooth animation."""
        if interval is None:
            interval = 1000 // self.fps
            
        self.ani = animation.FuncAnimation(
            self.fig, self.animate, frames=self.total_video_frames,
            interval=interval, blit=True, repeat=True
        )
        return self.ani
    
    def save_animation(self, format='mp4'):
        """Save high-quality animation."""
        ani = self.create_animation()
        
        # Ensure .mp4 extension
        output_path = self.save_path
        if not output_path.endswith('.mp4'):
            output_path = output_path + '.mp4'
        
        try:
            Writer = animation.writers['ffmpeg']
            # Higher bitrate for better quality
            writer = Writer(fps=self.fps, metadata=dict(artist='TrajectoryVisualizer'), 
                          bitrate=8000, extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
            ani.save(output_path, writer=writer, dpi=150)
            print(f"✅ Saved complete trajectory MP4 to: {output_path}")
        except Exception as e:
            print(f"⚠️ Could not save MP4: {e}")
            print(f"Make sure ffmpeg is installed and available in PATH")
    
        return ani

def visualize_2d_trajectory(save_prefix, save_folder, outstate, infstate, 
                                    rotation_angle=None, scale_factor=None,
                                    animation_fps=30, imu_sample_rate=100, 
                                    skip_seconds=0, skip_end_seconds=0, 
                                    speed_multiplier=1, format='mp4', debug=False):
    """Create 2D trajectory animation with optional trimming from start and end."""
    print(f"🎯 Creating trajectory animation for {save_prefix}")
    
    # Use infstate for both trajectories to ensure they start at the same point
    gt_trajectory = infstate["poses_gt"][0].cpu() if "poses_gt" in infstate else outstate["poses_gt"][0].cpu()
    pred_trajectory = infstate["poses"][0].cpu()
    
    # DEBUG: Check what data we're getting
    if debug:
        print("\n🔍 VISUALIZATION DEBUG - visualize_2d_trajectory:")
        print(f"   Using GT from: {'infstate' if 'poses_gt' in infstate else 'outstate'}")
        print(f"   GT trajectory shape: {gt_trajectory.shape}")
        print(f"   GT first position: {gt_trajectory[0]}")
        print(f"   GT last position: {gt_trajectory[-1]}")
        print(f"   \n   Pred trajectory shape: {pred_trajectory.shape}")
        print(f"   Pred first position: {pred_trajectory[0]}")
        print(f"   Pred last position: {pred_trajectory[-1]}")
        print(f"   \n   Both trajectories should now start at [0,0,0]: {torch.allclose(gt_trajectory[0], pred_trajectory[0], atol=1e-6)}")
    
    # Align trajectory lengths
    min_length = min(len(gt_trajectory), len(pred_trajectory))
    gt_trajectory = gt_trajectory[:min_length]
    pred_trajectory = pred_trajectory[:min_length]
    
    if debug:
        print(f"   After length alignment:")
        print(f"   GT start: {gt_trajectory[0]}, Pred start: {pred_trajectory[0]}")
        print(f"   Start positions match: {torch.allclose(gt_trajectory[0], pred_trajectory[0], atol=1e-6)}")
        print(f"   Aligned length: {min_length}")
    
    print(f"📊 Original trajectory length: {min_length} points")
    if skip_seconds > 0:
        print(f"📊 Will skip first {skip_seconds}s ({int(skip_seconds * imu_sample_rate)} points)")
    if skip_end_seconds > 0:
        print(f"📊 Will skip last {skip_end_seconds}s ({int(skip_end_seconds * imu_sample_rate)} points)")
    
    save_path = os.path.join(save_folder, f"{save_prefix}_2d_animation")
    os.makedirs(save_folder, exist_ok=True)
    
    animator = EfficientTrajectoryAnimator(
        gt_trajectory, pred_trajectory, save_path,
        rotation_angle=rotation_angle, scale_factor=scale_factor,
        imu_sample_rate=imu_sample_rate, fps=animation_fps,
        skip_seconds=skip_seconds, skip_end_seconds=skip_end_seconds,
        speed_multiplier=speed_multiplier
    )
    
    animator.save_animation(format=format)
    plt.close()
    
    # Reset matplotlib style
    plt.style.use('default')
    
    return animator.rotation_angle, animator.scale_factor

def create_2d_trajectory_animation(gt_trajectory, pred_trajectory, save_path, 
                                 duration=10, fps=30, label="AirIO"):
    """
    Create an animated GIF showing 2D (X-Y plane) trajectories being drawn progressively.
    
    Args:
        gt_trajectory: Ground truth trajectory (N x 3)
        pred_trajectory: Predicted trajectory (N x 3)
        save_path: Path to save the animation
        duration: Duration of animation in seconds
        fps: Frames per second
        label: Label for predicted trajectory
    """
    print(f"🎬 Creating animated 2D trajectory visualization...")
    
    # Convert to numpy if torch tensors
    if torch.is_tensor(gt_trajectory):
        gt_trajectory = gt_trajectory.cpu().numpy()
    if torch.is_tensor(pred_trajectory):
        pred_trajectory = pred_trajectory.cpu().numpy()
    
    # Handle length mismatch
    min_length = min(len(gt_trajectory), len(pred_trajectory))
    gt_trajectory = gt_trajectory[:min_length]
    pred_trajectory = pred_trajectory[:min_length]
    
    # Extract only X and Y coordinates (ignore Z)
    gt_x, gt_y = gt_trajectory[:, 0], gt_trajectory[:, 1]
    pred_x, pred_y = pred_trajectory[:, 0], pred_trajectory[:, 1]
    
    # Calculate total frames and ensure we show all points
    total_frames = duration * fps
    # Calculate how many points to add per frame to show full trajectory
    points_per_frame = len(gt_trajectory) / total_frames
    
    # Create figure with subplots
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[2, 1])
    
    # Main X-Y trajectory plot
    ax_main = fig.add_subplot(gs[0, :])
    ax_main.set_xlabel('X (m)', fontsize=12)
    ax_main.set_ylabel('Y (m)', fontsize=12)
    ax_main.set_title('X-Y Plane Trajectory Animation', fontsize=16)
    ax_main.set_xlim([min(gt_x.min(), pred_x.min())-2, max(gt_x.max(), pred_x.max())+2])
    ax_main.set_ylim([min(gt_y.min(), pred_y.min())-2, max(gt_y.max(), pred_y.max())+2])
    ax_main.grid(True, alpha=0.3)
    ax_main.set_aspect('equal')
    
    # 2D Error plot over time
    ax_error = fig.add_subplot(gs[1, 0])
    ax_error.set_xlabel('Time Steps', fontsize=12)
    ax_error.set_ylabel('2D Error (m)', fontsize=12)
    ax_error.set_title('X-Y Plane Error Over Time', fontsize=14)
    ax_error.grid(True, alpha=0.3)
    
    # Error distribution histogram
    ax_hist = fig.add_subplot(gs[1, 1])
    ax_hist.set_xlabel('2D Error (m)', fontsize=12)
    ax_hist.set_ylabel('Frequency', fontsize=12)
    ax_hist.set_title('Error Distribution', fontsize=14)
    ax_hist.grid(True, alpha=0.3, axis='y')
    
    # Calculate 2D errors (only X-Y plane)
    errors_2d = np.sqrt((gt_x - pred_x)**2 + (gt_y - pred_y)**2)
    ax_error.set_xlim([0, len(errors_2d)])
    ax_error.set_ylim([0, errors_2d.max() * 1.1])
    
    # Initialize empty line objects for main plot
    gt_line, = ax_main.plot([], [], 'g-', linewidth=3, label='Ground Truth', alpha=0.8)
    pred_line, = ax_main.plot([], [], 'b-', linewidth=3, label=label, alpha=0.8)
    
    # Initialize trail lines (thinner, more transparent)
    gt_trail, = ax_main.plot([], [], 'g-', linewidth=1, alpha=0.3)
    pred_trail, = ax_main.plot([], [], 'b-', linewidth=1, alpha=0.3)
    
    # Error line
    error_line, = ax_error.plot([], [], 'r-', linewidth=2, label='2D Error')
    
    # Start points
    gt_start = ax_main.scatter([], [], c='green', s=200, marker='o', 
                              edgecolors='darkgreen', linewidths=3, zorder=5)
    pred_start = ax_main.scatter([], [], c='blue', s=200, marker='s', 
                                edgecolors='darkblue', linewidths=3, zorder=5)
    
    # Current points (moving markers)
    gt_current = ax_main.scatter([], [], c='lime', s=150, marker='o', 
                                edgecolors='darkgreen', linewidths=2, zorder=6)
    pred_current = ax_main.scatter([], [], c='cyan', s=150, marker='o', 
                                  edgecolors='darkblue', linewidths=2, zorder=6)
    
    # End points (will appear at the end)
    gt_end = ax_main.scatter([], [], c='darkgreen', s=200, marker='*', 
                            edgecolors='green', linewidths=2, zorder=5)
    pred_end = ax_main.scatter([], [], c='darkblue', s=200, marker='*', 
                              edgecolors='blue', linewidths=2, zorder=5)
    
    # Add legends
    ax_main.legend(loc='upper right', fontsize=12)
    ax_error.legend(loc='upper right', fontsize=10)
    
    # Text elements
    progress_text = fig.text(0.5, 0.02, '', ha='center', fontsize=12, weight='bold')
    error_text = fig.text(0.02, 0.96, '', ha='left', va='top', fontsize=11,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    # Initialize histogram
    n_bins = 30
    ax_hist.set_xlim([0, errors_2d.max() * 1.1])
    
    def init():
        """Initialize animation"""
        gt_line.set_data([], [])
        pred_line.set_data([], [])
        gt_trail.set_data([], [])
        pred_trail.set_data([], [])
        error_line.set_data([], [])
        
        return (gt_line, pred_line, gt_trail, pred_trail, error_line, 
                progress_text, error_text)
    
    def animate(frame):
        """Animation function"""
        # Calculate how many points to show (ensures full trajectory is drawn)
        current_idx = int(min(frame * points_per_frame, len(gt_trajectory) - 1))
        
        # Main trajectory lines (last portion for visibility)
        trail_length = min(100, current_idx)  # Show last 100 points as bold
        if current_idx > trail_length:
            gt_line.set_data(gt_x[current_idx-trail_length:current_idx], 
                           gt_y[current_idx-trail_length:current_idx])
            pred_line.set_data(pred_x[current_idx-trail_length:current_idx], 
                             pred_y[current_idx-trail_length:current_idx])
            
            # Full trail (thin line)
            gt_trail.set_data(gt_x[:current_idx-trail_length], 
                            gt_y[:current_idx-trail_length])
            pred_trail.set_data(pred_x[:current_idx-trail_length], 
                              pred_y[:current_idx-trail_length])
        else:
            gt_line.set_data(gt_x[:current_idx], gt_y[:current_idx])
            pred_line.set_data(pred_x[:current_idx], pred_y[:current_idx])
        
        # Update error line
        error_line.set_data(range(current_idx), errors_2d[:current_idx])
        
        # Update markers
        if current_idx > 0:
            # Start points
            gt_start.set_offsets([[gt_x[0], gt_y[0]]])
            pred_start.set_offsets([[pred_x[0], pred_y[0]]])
            
            # Current point markers
            gt_current.set_offsets([[gt_x[current_idx-1], gt_y[current_idx-1]]])
            pred_current.set_offsets([[pred_x[current_idx-1], pred_y[current_idx-1]]])
            
            # End points (show when complete)
            if current_idx >= len(gt_trajectory) - 1:
                gt_end.set_offsets([[gt_x[-1], gt_y[-1]]])
                pred_end.set_offsets([[pred_x[-1], pred_y[-1]]])
        
        # Update histogram
        if current_idx > 10:
            ax_hist.clear()
            ax_hist.hist(errors_2d[:current_idx], bins=n_bins, color='red', alpha=0.7, edgecolor='darkred')
            ax_hist.set_xlabel('2D Error (m)', fontsize=12)
            ax_hist.set_ylabel('Frequency', fontsize=12)
            ax_hist.set_title('Error Distribution', fontsize=14)
            ax_hist.grid(True, alpha=0.3, axis='y')
            ax_hist.set_xlim([0, errors_2d.max() * 1.1])
        
        # Update progress text
        progress = (current_idx / len(gt_trajectory)) * 100
        time_elapsed = (frame / fps)
        progress_text.set_text(f'Progress: {progress:.1f}% | Time: {time_elapsed:.1f}s / {duration}s | Points: {current_idx}/{len(gt_trajectory)}')
        
        # Update error statistics
        if current_idx > 0:
            current_error = errors_2d[current_idx-1]
            mean_error = np.mean(errors_2d[:current_idx])
            max_error = np.max(errors_2d[:current_idx])
            std_error = np.std(errors_2d[:current_idx])
            
            # Calculate final displacement error
            final_displacement = np.sqrt((gt_x[current_idx-1] - pred_x[current_idx-1])**2 + 
                                       (gt_y[current_idx-1] - pred_y[current_idx-1])**2)
            
            error_text.set_text(
                f'2D Trajectory Statistics:\n'
                f'Current Error: {current_error:.3f} m\n'
                f'Mean Error: {mean_error:.3f} m\n'
                f'Max Error: {max_error:.3f} m\n'
                f'Std Dev: {std_error:.3f} m\n'
                f'Current Displacement: {final_displacement:.3f} m'
            )
        
        return (gt_line, pred_line, gt_trail, pred_trail, error_line, 
                progress_text, error_text)
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, init_func=init, 
                                 frames=total_frames, interval=1000/fps, 
                                 blit=False, repeat=True)
    
    # Save as GIF
    print(f"💾 Saving animation to {save_path}...")
    anim.save(save_path, writer='pillow', fps=fps, dpi=100)
    print(f"✅ Animation saved successfully!")
    
    plt.close()
    
    return save_path

def visualize_2d_trajectory_animation(gt_trajectory, pred_trajectories, labels, save_path, 
                                    imu_sample_rate=100, fps=30, skip_seconds=2, skip_end_seconds=2, 
                                    speed_multiplier=5):
    """
    Create animated visualization of multiple trajectories with 5x speed boost.
    
    Args:
        gt_trajectory: Ground truth trajectory (N x 3)
        pred_trajectories: List of predicted trajectories
        labels: List of labels for each trajectory
        save_path: Path to save animation (without extension)
        imu_sample_rate: IMU sampling rate in Hz
        fps: Animation frames per second
        skip_seconds: Skip first N seconds
        skip_end_seconds: Skip last N seconds
        speed_multiplier: Speed multiplier for animation
    """
    print(f"🎬 Creating multi-trajectory animation with {speed_multiplier}x speed boost...")
    
    # Convert to numpy if needed
    if torch.is_tensor(gt_trajectory):
        gt_trajectory = gt_trajectory.cpu().numpy()
    
    pred_trajectories_np = []
    for traj in pred_trajectories:
        if traj is not None:
            if torch.is_tensor(traj):
                pred_trajectories_np.append(traj.cpu().numpy())
            else:
                pred_trajectories_np.append(traj)
        else:
            pred_trajectories_np.append(None)
    
    # Apply skipping
    skip_start_points = int(skip_seconds * imu_sample_rate)
    skip_end_points = int(skip_end_seconds * imu_sample_rate)
    
    if skip_start_points > 0:
        gt_trajectory = gt_trajectory[skip_start_points:]
        pred_trajectories_np = [traj[skip_start_points:] if traj is not None else None 
                               for traj in pred_trajectories_np]
    
    if skip_end_points > 0:
        gt_trajectory = gt_trajectory[:-skip_end_points]
        pred_trajectories_np = [traj[:-skip_end_points] if traj is not None else None 
                               for traj in pred_trajectories_np]
    
    # Find minimum length
    lengths = [len(gt_trajectory)]
    for traj in pred_trajectories_np:
        if traj is not None:
            lengths.append(len(traj))
    
    min_length = min(lengths)
    gt_trajectory = gt_trajectory[:min_length]
    pred_trajectories_np = [traj[:min_length] if traj is not None else None 
                           for traj in pred_trajectories_np]
    
    # SIMPLE FIX: Normalize ALL trajectories to start at (0,0)
    if len(gt_trajectory) > 0:
        # Normalize GT to start at (0,0)
        gt_start = gt_trajectory[0, :2].copy()
        gt_trajectory[:, :2] -= gt_start
        
        # Normalize each prediction trajectory to start at (0,0)
        for i, traj in enumerate(pred_trajectories_np):
            if traj is not None and len(traj) > 0:
                pred_start = traj[0, :2].copy()
                pred_trajectories_np[i][:, :2] -= pred_start
    
    # Calculate animation parameters with speed multiplier
    video_duration = (min_length / imu_sample_rate) / speed_multiplier
    total_frames = int(video_duration * fps)
    
    print(f"📊 Animation: {min_length} points, {video_duration:.1f}s duration, {total_frames} frames")
    
    # Setup figure
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 12), facecolor='black')
    ax.set_facecolor('black')
    
    # Colors
    gt_color = '#00FF7F'  # Spring green
    pred_colors = ['#1E90FF', '#FF6347', '#FFD700', '#FF69B4']  # Blue, Red, Gold, Pink
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, color='gray', linewidth=0.5)
    ax.set_xlabel('X (m)', fontsize=14, color='white')
    ax.set_ylabel('Y (m)', fontsize=14, color='white')
    ax.set_title('Multi-Trajectory Comparison', fontsize=16, color='white', pad=20)
    
    # Calculate plot bounds
    all_x = [gt_trajectory[:, 0]]
    all_y = [gt_trajectory[:, 1]]
    for traj in pred_trajectories_np:
        if traj is not None:
            all_x.append(traj[:, 0])
            all_y.append(traj[:, 1])
    
    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    
    margin = 0.1
    x_range = all_x.max() - all_x.min()
    y_range = all_y.max() - all_y.min()
    
    ax.set_xlim(all_x.min() - margin * x_range, all_x.max() + margin * x_range)
    ax.set_ylim(all_y.min() - margin * y_range, all_y.max() + margin * y_range)
    
    # Animation elements
    lines = []
    points = []
    
    # GT trajectory
    gt_label = labels[0] if len(labels) > 0 else 'Ground Truth'
    gt_line, = ax.plot([], [], color=gt_color, linewidth=2, label=gt_label)
    gt_point, = ax.plot([], [], 'o', color=gt_color, markersize=8)
    lines.append(gt_line)
    points.append(gt_point)
    
    # Predicted trajectories
    for i, (traj, label) in enumerate(zip(pred_trajectories_np, labels[1:])):
        if traj is not None:
            color = pred_colors[i % len(pred_colors)]
            line, = ax.plot([], [], color=color, linewidth=2, label=label)
            point, = ax.plot([], [], 'o', color=color, markersize=8)
            lines.append(line)
            points.append(point)
    
    ax.legend(loc='upper right', fontsize=12)
    
    # Animation function
    def animate(frame):
        progress = frame / (total_frames - 1) if total_frames > 1 else 0
        traj_idx = int(progress * (min_length - 1))
        
        # Update GT
        gt_line.set_data(gt_trajectory[:traj_idx+1, 0], gt_trajectory[:traj_idx+1, 1])
        gt_point.set_data([gt_trajectory[traj_idx, 0]], [gt_trajectory[traj_idx, 1]])
        
        # Update predictions
        line_idx = 1
        point_idx = 1
        for traj in pred_trajectories_np:
            if traj is not None:
                lines[line_idx].set_data(traj[:traj_idx+1, 0], traj[:traj_idx+1, 1])
                points[point_idx].set_data([traj[traj_idx, 0]], [traj[traj_idx, 1]])
                line_idx += 1
                point_idx += 1
        
        return lines + points
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=total_frames, 
                                 interval=1000/fps, blit=True, repeat=True)
    
    # Save animation
    try:
        print(f"💾 Saving animation to {save_path}.mp4...")
        anim.save(f"{save_path}.mp4", writer='ffmpeg', fps=fps, dpi=100)
        print(f"✅ Animation saved successfully!")
    except Exception as e:
        print(f"❌ Failed to save MP4: {e}")
        print("📝 Trying to save as GIF...")
        try:
            anim.save(f"{save_path}.gif", writer='pillow', fps=fps, dpi=100)
            print(f"✅ GIF saved successfully!")
        except Exception as e2:
            print(f"❌ Failed to save GIF: {e2}")
    
    plt.close()
    plt.style.use('default')
    
    return f"{save_path}.mp4"

# Example usage within your existing code
def visualize_2D_motion_animated(save_prefix, save_folder, outstate, infstate, label="AirIO"):
    """Create 2D animated visualization focusing on X-Y plane only"""
    
    # Extract trajectories
    gt_trajectory = outstate["poses_gt"][0].cpu()
    airio_trajectory = infstate["poses"][0].cpu()
    
    # Create animation save path
    animation_path = os.path.join(save_folder, f"{save_prefix}_2D_animation.gif")
    
    # Create the animation
    create_2d_trajectory_animation(
        gt_trajectory, 
        airio_trajectory, 
        animation_path,
        duration=10,  # 10 seconds
        fps=30,       # 30 frames per second
        label=label
    )