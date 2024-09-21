import ast
import math
import os
from argparse import ArgumentParser
from copy import deepcopy
from os import makedirs

import cv2
import imageio
import torch
import torch.optim as optim

from arguments import (ModelParams, PipelineParams, get_combined_args,
                       iComMaParams)
from gaussian_renderer import GaussianModel, render
from LightGlue.lightglue import DISK, LightGlue, SuperPoint, viz2d
from LightGlue.lightglue.utils import load_image, rbd
from LoFTR.src.loftr import LoFTR, default_cfg
from scene import Scene
from scene.cameras import Camera_Pose
from utils.calculate_error_utils import cal_campose_error
from utils.general_utils import safe_state
from utils.icomma_helper import get_pose_estimation_input
from utils.image_utils import to8b
from utils.loss_utils import loss_loftr, loss_mse


# Load the pre-trained LoFTR model. For more details, please refer to https://github.com/zju3dv/LoFTR.
def load_LoFTR(ckpt_path:str, temp_bug_fix:bool):
    _default_cfg = deepcopy(default_cfg)
    _default_cfg['coarse']['temp_bug_fix'] = temp_bug_fix  # set to False when using the old ckpt
   
    LoFTR_model = LoFTR(config=_default_cfg)
    LoFTR_model.load_state_dict(torch.load(ckpt_path)['state_dict'])
    LoFTR_model = LoFTR_model.eval().cuda()
    return LoFTR_model

def loss_glue(kpc0, kpc1):
    match_pt_count = kpc0.shape[0]
    x1 = kpc0[:,0] / 376
    y1 = kpc0[:,1] / 1408
    x2 = kpc1[:,0] / 376
    y2 = kpc1[:,1] / 1408
    error_i = (x1 - x2) ** 2 + (y1 - y2) ** 2
    error_i = error_i.float().cuda()
    loss = torch.sum(error_i) / match_pt_count
    return loss

def match_lightglue(image0, image1, save_match: bool=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'

    extractor = SuperPoint(max_num_keypoints=4096).eval().to(device)  # load the extractor
    matcher = LightGlue(features="superpoint").eval().to(device)

    feats0 = extractor.extract(image0.to(device))
    feats1 = extractor.extract(image1.to(device))
    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [
        rbd(x) for x in [feats0, feats1, matches01]
    ]  # remove batch dimension

    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    if save_match:
        axes = viz2d.plot_images([image0, image1])
        viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
        viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
        viz2d.save_plot("./match_lightglue_lines.png")

        kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
        viz2d.plot_images([image0, image1])
        viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)
        viz2d.save_plot("./match_lightglue_pts.png")

    return loss_glue(m_kpts0, m_kpts1)


def camera_pose_estimation(gaussians:GaussianModel, background:torch.tensor, LoFTR_model,
                           pipeline:PipelineParams, icommaparams:iComMaParams, icomma_info, output_path):
    # start pose & gt pose
    gt_pose_c2w = icomma_info.gt_pose_c2w
    start_pose_w2c = icomma_info.start_pose_w2c.cuda()
    # query_image for comparing 
    query_image = icomma_info.query_image.cuda()

    # initialize camera pose object
    camera_pose = Camera_Pose(start_pose_w2c, FoVx=icomma_info.FoVx, FoVy=icomma_info.FoVy,
                            image_width=icomma_info.image_width, image_height=icomma_info.image_height)
    camera_pose.cuda()
    # store gif elements
    imgs=[]

    matching_flag = not icommaparams.deprecate_matching

    # start optimizing
    optimizer = optim.Adam(camera_pose.parameters(), lr=icommaparams.camera_pose_lr)
    iter = icommaparams.pose_estimation_iter
    num_iter_matching = 0
    for k in range(iter):

        rendering = render(camera_pose, gaussians, pipeline, 
                           background, compute_grad_cov2d=icommaparams.compute_grad_cov2d)["render"]
        
        if matching_flag:
            loss_matching = loss_loftr(query_image, rendering, LoFTR_model, 
                                       icommaparams.confidence_threshold_LoFTR, icommaparams.min_matching_points)
            # loss_matching = match_lightglue(query_image, rendering)
            loss_comparing = loss_mse(rendering, query_image)
            
            if loss_matching is None:
                loss = loss_comparing
            else:  
                loss = icommaparams.lambda_LoFTR * loss_matching + (1 - icommaparams.lambda_LoFTR) * loss_comparing
                if loss_matching < 0.001:
                    matching_flag = False
                    
            num_iter_matching += 1
        else:
            loss_comparing = loss_mse(rendering, query_image)
            loss = loss_comparing
            
            new_lrate = icommaparams.camera_pose_lr * (0.6 ** ((k - num_iter_matching + 1) / 50))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate
        
        # output intermediate results
        if (k + 1) % 20 == 0 or k == 0:
            print('Step: ', k)
            if matching_flag and loss_matching is not None:
                print('Matching Loss: ', loss_matching.item())
            print('Comparing Loss: ', loss_comparing.item())
            print('Loss: ', loss.item())

            # record error
            with torch.no_grad():
                cur_pose_c2w = camera_pose.current_campose_c2w()
                # print("cur_pose_c2w: ", cur_pose_c2w)
                rot_error, translation_error = cal_campose_error(cur_pose_c2w, gt_pose_c2w)
                print('Rotation error: ', rot_error)
                print('Translation error: ', translation_error)
                # print('-----------------------------------')
               
            # output images
            if icommaparams.OVERLAY is True:
                with torch.no_grad():
                    rgb = rendering.clone().permute(1, 2, 0).cpu().detach().numpy()
                    rgb8 = to8b(rgb)
                    ref = to8b(query_image.permute(1, 2, 0).cpu().detach().numpy())
                    filename = os.path.join(output_path, str(k) + '.png')
                    dst = cv2.addWeighted(rgb8, 0.7, ref, 0.3, 0)
                    imageio.imwrite(filename, dst)
                    filename = os.path.join(output_path, 'render_' + str(k) + '.png')
                    imageio.imwrite(filename, rgb8)
                    imgs.append(dst)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        camera_pose(start_pose_w2c)

    # output gif
    if icommaparams.OVERLAY is True:
        imageio.mimwrite(os.path.join(output_path, 'video.gif'), imgs, fps=4)
  
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Camera pose estimation parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    icommaparams = iComMaParams(parser)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output_path", default='output', type=str, help="output path")
    parser.add_argument("--obs_img_index", default=20, type=int)
    parser.add_argument("--delta", default="[30, 10, 5, 0.1, 0.2, 0.3]", type=str)
    # parser.add_argument("--delta", default="[1,1,1,0.1,0.1,1.1]", type=str)
    parser.add_argument("--iteration", default=-1, type=int)
    args = get_combined_args(parser)
    # Initialize system state (RNG)
    safe_state(args.quiet)
    makedirs(args.output_path, exist_ok=True)
    # load LoFTR_model
    LoFTR_model=load_LoFTR(icommaparams.LoFTR_ckpt_path, icommaparams.LoFTR_temp_bug_fix)
    # load gaussians
    dataset = model.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    # get camera info from Scene
    # Reused 3DGS code to obtain camera information. 
    # You can customize the iComMa_input_info in practical applications.
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    obs_view = scene.getTestCameras()[args.obs_img_index]
    #obs_view = scene.getTrainCameras()[args.obs_img_index]
    print("obs_view: ", obs_view)
    icomma_info = get_pose_estimation_input(obs_view, ast.literal_eval(args.delta))
    # pose estimation
    camera_pose_estimation(gaussians, background, LoFTR_model, pipeline, icommaparams, icomma_info, args.output_path)






    # image0 = load_image('/workspace/vegs/bash_scripts/output/pose_estimated_results/ref.png')
    # image1 = load_image('/workspace/vegs/bash_scripts/output/pose_estimated_results/rendered_179.png')
    # loss = match_lightglue(image0, image1)
    # print(loss)