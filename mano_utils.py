
import torch
import mano
from mano.utils import Mesh
import pickle
from scipy.spatial.transform import Rotation as R
import numpy as np

model_path = 'models/mano'
n_comps = 45
batch_size = 1

lh_model = mano.load(model_path=model_path,
                     is_rhand= False,
                     num_pca_comps=n_comps,
                     batch_size=batch_size,
                     flat_hand_mean=False)

rh_model = mano.load(model_path=model_path,
                     is_rhand= True,
                     num_pca_comps=n_comps,
                     batch_size=batch_size,
                     flat_hand_mean=False)

def hand_rotvet_to_pose(hand_rotvec: np.ndarray, hand_type="inspire"):
    """
    Convert hand rotation vector to pose matrix.
    hand_rotvec: (Batch, 30, 3),
    return: (Batch, 10, 3)
    """
    batch_size, num_joints, _ = hand_rotvec.shape
    betas = torch.zeros(batch_size, 10)
    hand_rotvec = torch.from_numpy(hand_rotvec).float()
    global_orient = torch.zeros(batch_size, 3)
    transl        = torch.zeros(batch_size, 3)

    loutput = lh_model(betas=betas,
                    global_orient=global_orient,
                    hand_pose=hand_rotvec[:, :15].reshape(batch_size, -1),
                    transl=transl,
                    return_verts=True,
                    return_tips = True)
    
    routput = rh_model(betas=betas,
                    global_orient=global_orient,
                    hand_pose=hand_rotvec[:, 15:].reshape(batch_size, -1),
                    transl=transl,
                    return_verts=True,
                    return_tips = True)
    
    match hand_type:
        case "inspire":
            finger_index = [16, 17, 18, 19, 20]
            offset = np.array([[0.1, 0, 0]])
            scale = 1
        case "dex3":
            finger_index = [16, 17, 19]
            offset = np.array([[0.08, 0, 0]])
            scale = 1
    lpoint = loutput.joints.cpu().detach().numpy()
    lpoint = (lpoint - lpoint[:, 0:1])[:, finger_index].reshape(batch_size, -1, 3)
    rpoint = routput.joints.cpu().detach().numpy()
    rpoint = (rpoint - rpoint[:, 0:1])[:, finger_index].reshape(batch_size, -1, 3) * np.array([-1, -1, 1])
    output = (np.concatenate((lpoint, rpoint), axis=1) + offset)* scale
    return output