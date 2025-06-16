import numpy as np
import torch
from typing import List, Optional, Tuple, Dict
import importlib
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
from mujoco.glfw import glfw
from mano_utils import hand_rotvet_to_pose
import pickle
import time

class MujocoKinematics:
    def __init__(self, xml_path: str, joint_names_subset: Optional[List[str]] = None, hand_type: str = "inspire"):
        self.mujoco = importlib.import_module("mujoco")
        self.mujoco_viewer = importlib.import_module("mujoco_viewer")
        
        self.model = self.mujoco.MjModel.from_xml_path(xml_path)
        self.data = self.mujoco.MjData(self.model)
        self.viewer = self.mujoco_viewer.MujocoViewer(self.model, self.data)
        self.viewer.cam.distance = 5.0
        self.viewer._paused = False

        self.marker_pos = None
        self.marker_rgba = None

        # base joint type
        self.has_free_joint = self.model.jnt_type[0] == self.mujoco.mjtJoint.mjJNT_FREE
        self.qpos_offset = 7 if self.has_free_joint else 0

        if joint_names_subset is not None:
            self.num_joints = len(joint_names_subset)
            self.joint_names = joint_names_subset
        else:
            self.num_joints = 0
            self.joint_names = []
            for i in range(self.model.njnt):
                # skip root joint
                if self.has_free_joint and i == 0:
                    continue
                joint_name = self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, i)
                self.joint_names.append(joint_name)
                self.num_joints += 1

        self.joint_qpos_indices = []
        for joint_name in self.joint_names:
            joint_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id == -1:
                raise ValueError(f"Joint {joint_name} not found in the model.")
            qpos_addr = self.model.jnt_qposadr[joint_id]
            self.joint_qpos_indices.append(qpos_addr)

        print(f"[MujocoKinematics] Loaded {self.num_joints} joints from {xml_path}")
        self.hand_type = hand_type
        match self.hand_type:
            case "inspire":
                self.selected_idx = [4,6,8,10,12,16,18,20,22,24]
            case "dex3":
                self.selected_idx = [6,4,10,18,16,22]
            case _:
                raise ValueError(f"Unknown hand type {hand_type}")


    def forward(
        self,
        joint_pos: np.ndarray,
        base_pos: Optional[np.ndarray] = None,
        base_quat: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.hand_type == "inspire":
            joint_pos = self.dof12_to_24dof(joint_pos)
        elif self.hand_type == "dex3":
            joint_pos = self.dex_jpos_preprocess(joint_pos)
        else:
            raise ValueError(f"Unknown hand type {self.hand_type}")
        
        joint_pos = np.clip(joint_pos, self.model.jnt_range[:, 0], self.model.jnt_range[:, 1])

        assert joint_pos.shape[0] == self.num_joints, (
            f"Expected joint_pos of shape ({self.num_joints},), got {joint_pos.shape}"
        )

        qpos_full = np.zeros(self.model.nq, dtype=np.float64)

        if self.has_free_joint:
            if base_pos is not None:
                assert base_pos.shape == (3,), "base_pos must be of shape (3,)"
                qpos_full[0:3] = base_pos

            if base_quat is not None:
                assert base_quat.shape == (4,), "base_quat must be of shape (4,)"
                qpos_full[3:7] = base_quat[[3, 0, 1, 2]]

        qpos_full[self.joint_qpos_indices] = joint_pos
        self.data.qpos[:] = qpos_full

        self.mujoco.mj_forward(self.model, self.data)

        nbody = self.model.nbody
        offset = 1 if self.has_free_joint else 0
        body_pos = np.zeros((nbody - offset, 3), dtype=np.float64)
        body_quat = np.zeros((nbody - offset, 4), dtype=np.float64)

        return self.data.site_xpos, None
        for i in range(0, nbody - offset):
            body_pos[i] = self.data.xpos[i + offset]
            body_quat[i] = self.data.xquat[i + offset][[1, 2, 3, 0]]  # [x, y, z, w]

        return body_pos[self.selected_idx], body_quat
    
    def add_marker(self, marker_pos, rgba=(1, 0, 0, 1)):
        if self.marker_pos is None:
            self.marker_pos = marker_pos
            self.marker_rgba = [rgba] * len(marker_pos)
        else:
            self.marker_pos = np.concatenate([self.marker_pos, marker_pos], axis=0)
            self.marker_rgba += [rgba] * len(marker_pos)

    def draw_marker(self):
        for j in range(self.marker_pos.shape[0]):
            # need to modify mujoco_viewer to support this
            self.viewer.add_marker(
                pos=self.marker_pos[j],
                size=0.005,
                rgba=self.marker_rgba[j],
                type=self.mujoco.mjtGeom.mjGEOM_SPHERE,
                label="",
                id=j,
            )

    def show(self):
        import time
        while True:
            try:
                # self.data.qpos[:3] = 
                # self.mujoco.mj_forward(self.model, self.data)
                self.viewer.cam.lookat = self.data.qpos.astype(np.float32)[:3]
                self.draw_marker()
                if self.viewer.is_alive:
                    self.viewer.render()
                time.sleep(0.02)
            except KeyboardInterrupt:
                break
            
    
    @staticmethod
    def dof12_to_24dof(hand_pose: np.ndarray) -> np.ndarray:
        outpose = np.zeros((24))
        outpose[0] = hand_pose[0]
        outpose[[1,2]] = hand_pose[1]
        outpose[[4,5]] = hand_pose[2]
        outpose[[6,7]] = hand_pose[3]
        outpose[[8,9]] = hand_pose[4]
        outpose[[10,11]] = hand_pose[5]
        
        outpose[12] = hand_pose[6]
        outpose[[13,14]] = hand_pose[7]
        outpose[[16,17]] = hand_pose[8]
        outpose[[18,19]] = hand_pose[9]
        outpose[[20,21]] = hand_pose[10]
        outpose[[22,23]] = hand_pose[11]

        return outpose
    
    @staticmethod
    def dex_jpos_preprocess(hand_pose: np.ndarray) -> np.ndarray:
        outpose = hand_pose
        return outpose

class Retargetor:
    def __init__(self, f: callable, loss: str = "l2", hand_type: str = "inspire"):
        self.f = f

        match hand_type:
            case "inspire":
                self.joint_idx = [0,1,4,6,8,10,12,13,16,18,20,22]
            case "dex3":
                self.joint_idx = list(range(14))
            case _:
                raise ValueError(f"Unknown hand type: {hand_type}")

        match loss:
            case "l2":
                self.loss = self.l2_loss
            case "l1":
                self.loss = self.l1_loss
            case _:
                raise ValueError(f"Unknown loss function: {loss}")

    def ik_solve(self, target_pos: np.ndarray, init_guess, bounds=None):
        """
        解算逆运动学
        target_pos: 目标位置，shape为(Joint, 3)
        """
        def objective(x):
            pred_pos, _ = self.f(x)
            return np.sum(self.loss(pred_pos, target_pos))
        
        result = minimize(objective, init_guess, bounds=bounds, method='L-BFGS-B', tol=1e-5)
        return result.x

    @staticmethod
    def l2_loss(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.sum((x - y) ** 2, axis=-1)
    
    @staticmethod
    def l1_loss(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.sum(np.abs(x - y), axis=-1)

def retarget_inspire_hand(finger_pos: np.ndarray, hand_type="inspire"):
    t, joint, _ = finger_pos.shape
    kine = MujocoKinematics(xml_path="assets/robot/inspire/scene.xml", hand_type=hand_type)
    solver = Retargetor(kine.forward, hand_type=hand_type)
    # joint_idx = [0,1,4,6,8,10,12,13,16,18,20,22]

    if hand_type == "inspire":
        hand_pose = np.zeros((t, 12))
        init_guess = np.ones((12,))
    elif hand_type == "dex3":
        hand_pose = np.zeros((t, 14))
        init_guess = np.array([0,0,1,-1,-1,-1,-1, 0,0,-1,1,1,1,1])* 0.8

    for i in range(t):
        hand_pose[i] = solver.ik_solve(finger_pos[i], init_guess=init_guess)#  % (2 * np.pi)
        # print(solver.ik_solve(data, np.ones((12,)) * 0.8))
        hand_pose[i] = np.clip(hand_pose[i], kine.model.jnt_range[solver.joint_idx, 0], kine.model.jnt_range[solver.joint_idx, 1])
        init_guess = hand_pose[i]
    kine.add_marker(finger_pos[-1])
    kine.add_marker(np.zeros((1, 3)), rgba=(0, 1, 0, 1))
    kine.show()
    return hand_pose

def test():

    phc_robot_config = CFG.policy.phc.robot_config
    xml_path = phc_robot_config.asset.assetFileName
    kine = MujocoKinematics(xml_path="scene.xml", hand_type="dex3")


    data = np.load("joints.npy")[0]
    data, _ = kine.forward(
        joint_pos=np.array([0,0,1,-1,-1,-1,-1, 0,0,-1,1,1,1,1]),
        base_pos=np.zeros((3,)),
        base_quat=np.array([0, 0, 0, 1]),
    )

    kine.add_marker(data[:])
    kine.add_marker(np.zeros((1, 3)), rgba=(0, 1, 0, 1))
    import time
    start = time.time()
    solver = Retargetor(kine.forward, hand_type="dex3")
    # print(solver.ik_solve(data, np.ones((12,)) * 0.8))
    print(np.clip(solver.ik_solve(data, np.array([0,0,1,-1,-1,-1,-1, 0,0,-1,1,1,1,1]),), kine.model.jnt_range[solver.joint_idx, 0], kine.model.jnt_range[solver.joint_idx, 1]))
    
    print(time.time() - start)
    kine.show()
    
if __name__ == "__main__":
    # test()
    hand_type = "inspire"
    # origin_data = np.load("assets/resources/motions/test/0_out.npy")[0]
    t = 1 # origin_data.shape[0]

    data = R.from_matrix(pickle.load(open("assets/data/hand_pose.pkl", "rb"))[1]['pred_mano_params']['hand_pose'].cpu().numpy().reshape(-1, 3, 3)).as_rotvec()
    hand_data = np.tile(data[None, :, :], (t, 1, 1))
    # hand_data: (t, 30, 3) left hand and right hand all have 15 joints x 3 rotvec dim
    print(hand_data.shape)
    finger_pos = hand_rotvet_to_pose(hand_data, hand_type=hand_type)

    start = time.time()
    hand_jpos = retarget_inspire_hand(finger_pos, hand_type=hand_type)

    print(time.time() - start)
    print(hand_jpos[0])
    print(hand_jpos[-1])

    # save_data = np.concatenate([origin_data, hand_jpos.reshape(-1, 4, 3)], axis=1)
    # print(save_data.shape)
    # np.save("processed_data.npy", save_data)

    # print(retarget_inspire_hand(data))
