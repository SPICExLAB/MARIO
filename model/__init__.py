from .code import *

# config `network:` name -> class
net_dict = {
    'codewithrot':          CodeNetMotionwithRot,           # AirIO backbone (single IMU)
    'codewithrot_pose':     CodeNetMotionwithRot_Pose,      # + PoseNet prior
    'codewithrot_baro':     CodeNetMotionwithRot_Baro,      # + barometer
    'codewithrot_mag':      CodeNetMotionwithRot_Mag,       # + magnetometer
    'codewithrot_leftIMU':  CodeNetMotionwithRot_LeftIMU,   # + secondary IMU
    'codewithrot_combined': CodeNetMotionwithRot_Combined,  # + all (pose, baro, mag, secondary IMU)
    'codewithrot_pose_v4':  CodeNetMotionwithPose_v4,       # PoseNet itself
}
