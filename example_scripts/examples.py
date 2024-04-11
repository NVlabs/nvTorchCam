import numpy as np
import imageio
from scipy.spatial.transform import Rotation
from struct import unpack
import os
import torch
import nvtorchcam.cameras as cameras
import nvtorchcam.warpings as warpings
import nvtorchcam.utils as utils
import example_scripts.write_ply as write_ply
import cv2
import argparse

examples_output = 'examples_output'

def write_image(save_path, image):
    if image.size(0) == 3:
        image = (image.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
    elif image.size(0) == 1:
        image = (image.squeeze().cpu().numpy()*255).astype(np.uint8)
    imageio.imwrite(save_path, image)

def load_matterport360_example(path, device='cpu'):
    coord_change = np.eye(4)
    coord_change[1,1] = -1
    coord_change[2,2] = -1
    im = imageio.imread(path+'_rgb.png').astype(np.float32)/255
    depth =  read_dpt(path+'_depth.dpt')
    pose = np.loadtxt(path+'_pose.txt')
    rot = Rotation.from_quat(pose[3:])
    R = rot.as_matrix()
    t = pose[:3]
    to_world = np.eye(4)
    to_world[:3,:3] = R
    to_world[:3,3] = t
    to_world =np.matmul(to_world,coord_change)
    im = torch.from_numpy(im).permute(2,0,1)
    depth = torch.from_numpy(depth).unsqueeze(0)
    to_world = torch.from_numpy(to_world).float()
    cam = cameras.EquirectangularCamera.make(batch_shape=())
    return im.to(device), depth.to(device), to_world.to(device), cam.to(device)

def load_scannet_example(path = 'example_scripts/data/scannet', view_num = 0, device='cpu'):
    intrinsics = np.loadtxt(os.path.join(path, 'intrinsics.txt'))
    world_to_cam = np.loadtxt(os.path.join(path, 'pose', '%06d.txt' % view_num))
    to_world = np.linalg.inv(world_to_cam)
    im = imageio.imread(os.path.join(path,'rgb', '%06d.jpg' % view_num)).astype(np.float32)/255
    depth = cv2.imread(os.path.join(path,'depth', '%06d.png' % view_num), 2) / 1000.0

    to_world = torch.from_numpy(to_world).float()
    im = torch.from_numpy(im).float().permute(2,0,1)
    depth = torch.from_numpy(depth).float().unsqueeze(0)

    intrinsics = torch.from_numpy(intrinsics).float()
    n_intrinsics = utils.normalized_intrinsics_from_pixel_intrinsics(intrinsics, im.shape[-2:])
    cam = cameras.PinholeCamera.make(n_intrinsics)
    return im.to(device), depth.to(device), to_world.to(device), cam.to(device)
   
def read_dpt(dpt_file_path):
    #from https://github.com/manurare/360monodepth/blob/main/code/python/src/utility/depthmap_utils.py
    """read depth map from *.dpt file.
    :param dpt_file_path: the dpt file path
    :type dpt_file_path: str
    :return: depth map data
    :rtype: numpy
    """
    TAG_FLOAT = 202021.25  # check for this when READING the file

    ext = os.path.splitext(dpt_file_path)[1]

    assert len(ext) > 0, ('readFlowFile: extension required in fname %s' % dpt_file_path)
    assert ext == '.dpt', exit('readFlowFile: fname %s should have extension ''.flo''' % dpt_file_path)

    fid = None
    try:
        fid = open(dpt_file_path, 'rb')
    except IOError:
        print('readFlowFile: could not open %s', dpt_file_path)

    tag = unpack('f', fid.read(4))[0]
    width = unpack('i', fid.read(4))[0]
    height = unpack('i', fid.read(4))[0]

    assert tag == TAG_FLOAT, ('readFlowFile(%s): wrong tag (possibly due to big-endian machine?)' % dpt_file_path)
    assert 0 < width and width < 100000, ('readFlowFile(%s): illegal width %d' % (dpt_file_path, width))
    assert 0 < height and height < 100000, ('readFlowFile(%s): illegal height %d' % (dpt_file_path, height))

    # arrange into matrix form
    depth_data = np.fromfile(fid, np.float32)
    depth_data = depth_data.reshape(height, width)

    fid.close()

    return depth_data

def save_pointcloud(save_name, im, depth, to_world, cam, depth_is_along_ray):
    origin, dirs, valid = cam.get_camera_rays(im.shape[1:3], depth_is_along_ray)
    pc_image = origin + dirs*depth.squeeze(0).unsqueeze(-1) #(h,w,3)
    pc_image = utils.apply_affine(to_world, pc_image)
    pc_color = im.permute(1,2,0)

    pc_image = pc_image.reshape(-1,3)
    pc_color = pc_color.reshape(-1,3)
    pc_valid = valid.reshape(-1) & torch.all(torch.isfinite(pc_image), dim=-1).reshape(-1)
    write_ply.write_ply_standard(save_name, pc_image[pc_valid,:], colors=pc_color[pc_valid,:])

def example_save_pointclouds():
    """ save unprojected depth maps from matterport ERPs and scannet pinholes
        as pointclouds """
    out_dir = os.path.join(examples_output,'save_pointclouds')
    os.makedirs(out_dir, exist_ok=True)
    for i, path in enumerate(['example_scripts/data/matterport360/0c3c242b5567468889d3c66eb931d6e8','example_scripts/data/matterport360/bf85972c0d1c47c0b2c90a0c41cb4c62']):
        im, depth, to_world, cam = load_matterport360_example(path)
        save_pointcloud(os.path.join(out_dir, 'matterport_%d.ply' % i), im, depth, to_world, cam, True)
    
    for i in range(0,5):
        im, depth, to_world, cam = load_scannet_example(view_num=i)
        save_pointcloud(os.path.join(out_dir, 'scannet_%d.ply' % i), im, depth, to_world, cam, False)

def example_warp_to_other_models():
    """ Demonstrates warping one camera model to another with resample by intrinsics. Possibly with
        an extrinsic rotation.
        Also demonstrates heterogeneous batching.
    """
    out_dir = os.path.join(examples_output, 'warp_to_other_model')
    os.makedirs(out_dir, exist_ok=True)
    im, depth, to_world, cam = load_matterport360_example('example_scripts/data/matterport360/0c3c242b5567468889d3c66eb931d6e8')
    pin_cam = cameras.PinholeCamera.make(torch.eye(3))
    
    intrin = torch.eye(3)
    intrin[0,0] = 1.5
    intrin[1,1] = 1.5
    k1 = 1.6798e-2
    k2 = 1.6548
    xi = 2.213
    fisheye_cam = cameras.Kitti360FisheyeCamera.make(intrin,k1,k2,xi,3.14/2)

    both_cams = torch.stack([pin_cam, fisheye_cam],dim=0)
    
    resampled_images_and_depths,_ = warpings.resample_by_intrinsics([im.unsqueeze(0).expand(2,-1,-1,-1), depth.unsqueeze(0).expand(2,-1,-1,-1,)],
                                                       cam.unsqueeze(0).expand(2), 
                                                       both_cams,
                                                       (512,512),
                                                        interp_mode=['bilinear','nearest'])
    resampled_images = resampled_images_and_depths[0]
    resampled_depths = resampled_images_and_depths[1]

    write_image(os.path.join(out_dir, 'original_image.png'), im)
    for i, save_name in enumerate(['pin','fish']):
        write_image(os.path.join(out_dir, 'resampled_image_%s.png' % save_name), resampled_images[i])
        imageio.imwrite(os.path.join(out_dir, 'resampled_depth_%s.png' % save_name), (resampled_depths[i].squeeze()*25).numpy().astype(np.uint8))

    #now do the same thing with a 180 degree extrinsic rotation
    rotation_trg_to_src = torch.tensor([[-1, 0, 0],
                                       [0, 1, 0],
                                        [0,  0 , -1]]).unsqueeze(0).expand(2,-1,-1)
    resampled_images_and_depths,_ = warpings.resample_by_intrinsics([im.unsqueeze(0).expand(2,-1,-1,-1), depth.unsqueeze(0).expand(2,-1,-1,-1,)],
                                                                  cam.unsqueeze(0).expand(2), 
                                                                  both_cams,
                                                                  (512,512),
                                                                  interp_mode=['bilinear','nearest'],
                                                                  rotation_trg_to_src=rotation_trg_to_src, depth_is_along_ray=False)
    resampled_images = resampled_images_and_depths[0]
    resampled_depths = resampled_images_and_depths[1]

    for i, save_name in enumerate(['pin','fish']):
        write_image(os.path.join(out_dir, 'resampled_image_rot_%s.png' % save_name), resampled_images[i])
        imageio.imwrite(os.path.join(out_dir, 'resampled_depth_rot_%s.png' % save_name), (resampled_depths[i].squeeze()*25).numpy().astype(np.uint8))

def example_backward_warp():
    """ Demonstrates backward warping with ERPs"""
    out_dir = os.path.join(examples_output, 'backward_warp')
    os.makedirs(out_dir, exist_ok=True)
    trg_image, trg_depth, trg_to_world, trg_cam = load_matterport360_example('example_scripts/data/matterport360/0c3c242b5567468889d3c66eb931d6e8')
    src_image, src_depth, src_to_world, src_cam = load_matterport360_example('example_scripts/data/matterport360/bf85972c0d1c47c0b2c90a0c41cb4c62')
    trg_cam_to_src_cam = torch.mm(torch.inverse(src_to_world), trg_to_world)
    src_image_warped, src_pts, valid_mask = warpings.backward_warp(trg_cam, trg_depth, src_cam, src_image, trg_cam_to_src_cam, depth_is_along_ray=True)
    src_image_warped[:, ~valid_mask] = 0
    write_image(os.path.join(out_dir, 'trg_image.png'), trg_image)
    write_image(os.path.join(out_dir, 'src_image.png'), src_image)
    write_image(os.path.join(out_dir, 'src_image_warped.png'), src_image_warped)

def example_mvsnet_fusion():
    """ Saves three point clouds from scannet then filters out the points in the first point cloud that do not appear in both the
        second and third point cloud
    """
    out_dir = os.path.join(examples_output, 'mvsnet_fusion')
    os.makedirs(out_dir, exist_ok=True)

    images = []
    to_worlds = []
    depths = []
    cameras = []
    for i in range(0,3):
        im, depth, to_world, cam = load_scannet_example(view_num=i)
        save_pointcloud(os.path.join(out_dir, 'pc_%d.ply' % i), im, depth, to_world, cam, depth_is_along_ray=False)
        images.append(im)
        depths.append(depth)
        to_worlds.append(to_world)
        cameras.append(cam)

    images = torch.stack(images)
    depths = torch.stack(depths)
    to_worlds = torch.stack(to_worlds)
    cameras = torch.stack(cameras)

    
    trg_to_world = to_worlds[0:1].expand(2,-1,-1)
    src_to_world = to_worlds[1:]
    trg_to_src = torch.bmm(torch.inverse(src_to_world), trg_to_world)
    fused_depth, fused_valid = warpings.fuse_depths_mvsnet(cameras[0], depths[0], cameras[1:], depths[1:], trg_to_src, depth_is_along_ray=False, num_image_threshold=3)
    fused_depth[~fused_valid] = torch.nan
    save_pointcloud(os.path.join(out_dir, 'pc_0_filtered.ply'), images[0],  fused_depth, to_worlds[0], cameras[0], depth_is_along_ray=False)

def example_stereo_rectify():
    """ demonstrates stereo rectifying on to pinholes and the analogous rectification on ERPs"""
    out_dir = os.path.join(examples_output, 'stereo_rectify')
    os.makedirs(out_dir, exist_ok=True)
    image0, depth0, to_world0, cam0 = load_matterport360_example('example_scripts/data/matterport360/0c3c242b5567468889d3c66eb931d6e8', device='cuda')
    image1, depth1, to_world1, cam1 = load_matterport360_example('example_scripts/data/matterport360/bf85972c0d1c47c0b2c90a0c41cb4c62', device='cuda')
    images = torch.stack((image0, image1), dim=0)
    to_worlds = torch.stack((to_world0, to_world1), dim=0)
    cams = torch.stack((cam0, cam1), dim=0)
    new_images, new_to_worlds, valid_mask = warpings.stereo_rectify(images, cams, to_worlds, (512,1024))
    
    write_image(os.path.join(out_dir,'original_top.png'), images[0])
    write_image(os.path.join(out_dir,'original_bottom.png'), images[1])
    write_image(os.path.join(out_dir,'rectified_top.png'), new_images[0])
    write_image(os.path.join(out_dir,'rectified_bottom.png'), new_images[1])

    trg_cams = cameras.PinholeCamera.make(torch.eye(3).unsqueeze(0).expand(2,-1,-1))
    new_images, new_to_worlds, valid_mask = warpings.stereo_rectify(images, cams, to_worlds, (512,1024), trg_cams=trg_cams, top_bottom=False)
   
    write_image(os.path.join(out_dir,'original_left.png'), images[0])
    write_image(os.path.join(out_dir,'original_right.png'), images[1])
    write_image(os.path.join(out_dir,'rectified_left.png'), new_images[0])
    write_image(os.path.join(out_dir,'rectified_right.png'), new_images[1])

def example_backward_warp_sphere():
    out_dir = os.path.join(examples_output, 'backward_warp_sphere')
    os.makedirs(out_dir, exist_ok=True)
    cam = cameras.OrthographicCamera.make(torch.eye(3))
    trg_to_world = torch.eye(4)
    trg_to_world[2,3] = -4
    res = (1024,1024)
    trg_image, trg_distance = warpings.render_sphere_image(cam, trg_to_world, res, 1)
    

    write_image(os.path.join(out_dir, 'trg_image.png'), (trg_image+1)/2)
    write_image(os.path.join(out_dir,'trg_distance.png'), trg_distance)

    src_to_world = torch.tensor([[0, 0, -1, 4],
                               [0, 1,  0, 0 ],
                               [1, 0,  0, 0 ],
                               [0, 0,  0, 1.0 ]])

    src_image, src_distance = warpings.render_sphere_image(cam, src_to_world, res, 1)
  
    write_image(os.path.join(out_dir, 'src_image.png'), (src_image+1)/2)
    write_image(os.path.join(out_dir,'src_distance.png'), src_distance)

    trg_to_src = torch.mm(torch.inverse(src_to_world), trg_to_world)
    src_warped_and_dist, _, valid_mask = warpings.backward_warp(cam, trg_distance, cam, [src_image, src_distance], trg_to_src, depth_is_along_ray=True, interp_mode='nearest')
    src_warped = src_warped_and_dist[0]
    src_distance_warped = src_warped_and_dist[1]

    write_image(os.path.join(out_dir,'src_warped.png'), (src_warped+1)/2)
    write_image(os.path.join(out_dir, 'src_distance_warped.png'), src_distance_warped)
    diff = torch.abs(trg_image - src_warped)
    write_image(os.path.join(out_dir,'diff.png'), diff)

def list_available_examples():
    print("Available example names:")
    for name in function_mapping.keys():
        print(name)

if __name__ == '__main__':
    function_mapping = {
        'save_pointclouds': example_save_pointclouds,
        'warp_to_other_models': example_warp_to_other_models,
        'backward_warp': example_backward_warp,
        'mvsnet_fusion': example_mvsnet_fusion,
        'stereo_rectify': example_stereo_rectify,
        'backward_warp_sphere': example_backward_warp_sphere,
    }

    parser = argparse.ArgumentParser(description='Run a function based on the example name.')
    parser.add_argument('example_name', nargs='?', help='Name of the example to run')
    args = parser.parse_args()

    if args.example_name:
        if args.example_name in function_mapping:
            function_mapping[args.example_name]()
        else:
            print(f"No example named '{args.example_name}'.")
            list_available_examples()
    else:
        list_available_examples()
