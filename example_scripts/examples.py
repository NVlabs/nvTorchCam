import numpy as np
import imageio.v2 as imageio
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
        image = (image.permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)
    elif image.size(0) == 1:
        image = (image.squeeze().cpu().numpy()*255).astype(np.uint8)
    imageio.imwrite(save_path, image)


def load_matterport360_example(path, device='cpu'):
    coord_change = np.eye(4)
    coord_change[1, 1] = -1
    coord_change[2, 2] = -1
    im = imageio.imread(path+'_rgb.png').astype(np.float32)/255
    depth = read_dpt(path+'_depth.dpt')
    pose = np.loadtxt(path+'_pose.txt')
    rot = Rotation.from_quat(pose[3:])
    R = rot.as_matrix()
    t = pose[:3]
    to_world = np.eye(4)
    to_world[:3, :3] = R
    to_world[:3, 3] = t
    to_world = np.matmul(to_world, coord_change)
    im = torch.from_numpy(im).permute(2, 0, 1)
    depth = torch.from_numpy(depth).unsqueeze(0)
    to_world = torch.from_numpy(to_world).float()
    cam = cameras.EquirectangularCamera.make(batch_shape=())
    return im.to(device), depth.to(device), to_world.to(device), cam.to(device)


def load_scannet_example(path='example_scripts/data/scannet', view_num=0, device='cpu'):
    intrinsics = np.loadtxt(os.path.join(path, 'intrinsics.txt'))
    world_to_cam = np.loadtxt(os.path.join(
        path, 'pose', '%06d.txt' % view_num))
    to_world = np.linalg.inv(world_to_cam)
    im = imageio.imread(os.path.join(path, 'rgb', '%06d.jpg' %
                        view_num)).astype(np.float32)/255
    depth = cv2.imread(os.path.join(
        path, 'depth', '%06d.png' % view_num), 2) / 1000.0

    to_world = torch.from_numpy(to_world).float()
    im = torch.from_numpy(im).float().permute(2, 0, 1)
    depth = torch.from_numpy(depth).float().unsqueeze(0)

    intrinsics = torch.from_numpy(intrinsics).float()
    n_intrinsics = utils.normalized_intrinsics_from_pixel_intrinsics(
        intrinsics, im.shape[-2:])
    cam = cameras.PinholeCamera.make(n_intrinsics)
    return im.to(device), depth.to(device), to_world.to(device), cam.to(device)


def read_dpt(dpt_file_path):
    # from https://github.com/manurare/360monodepth/blob/main/code/python/src/utility/depthmap_utils.py
    """read depth map from *.dpt file.
    :param dpt_file_path: the dpt file path
    :type dpt_file_path: str
    :return: depth map data
    :rtype: numpy
    """
    TAG_FLOAT = 202021.25  # check for this when READING the file

    ext = os.path.splitext(dpt_file_path)[1]

    assert len(
        ext) > 0, ('readFlowFile: extension required in fname %s' % dpt_file_path)
    assert ext == '.dpt', exit(
        'readFlowFile: fname %s should have extension ''.flo''' % dpt_file_path)

    fid = None
    try:
        fid = open(dpt_file_path, 'rb')
    except IOError:
        print('readFlowFile: could not open %s', dpt_file_path)

    tag = unpack('f', fid.read(4))[0]
    width = unpack('i', fid.read(4))[0]
    height = unpack('i', fid.read(4))[0]

    assert tag == TAG_FLOAT, (
        'readFlowFile(%s): wrong tag (possibly due to big-endian machine?)' % dpt_file_path)
    assert 0 < width and width < 100000, ('readFlowFile(%s): illegal width %d' % (
        dpt_file_path, width))
    assert 0 < height and height < 100000, ('readFlowFile(%s): illegal height %d' % (
        dpt_file_path, height))

    # arrange into matrix form
    depth_data = np.fromfile(fid, np.float32)
    depth_data = depth_data.reshape(height, width)

    fid.close()

    return depth_data


def save_pointcloud(save_name, im, depth, to_world, cam, depth_is_along_ray):
    pc_image, valid = cam.unproject_depth(
        depth, depth_is_along_ray=depth_is_along_ray, to_world=to_world)
    pc_image = pc_image
    pc_color = im.permute(1, 2, 0)

    pc_image = pc_image.reshape(-1, 3)
    pc_color = pc_color.reshape(-1, 3)
    pc_valid = valid.reshape(-1) & torch.all(torch.isfinite(pc_image),
                                             dim=-1).reshape(-1)
    write_ply.write_ply_standard(
        save_name, pc_image[pc_valid, :], colors=pc_color[pc_valid, :])


def example_save_pointclouds():
    """ save unprojected depth maps from matterport ERPs and scannet pinholes
        as pointclouds """
    out_dir = os.path.join(examples_output, 'save_pointclouds')
    os.makedirs(out_dir, exist_ok=True)
    for i, path in enumerate(['example_scripts/data/matterport360/0c3c242b5567468889d3c66eb931d6e8', 'example_scripts/data/matterport360/bf85972c0d1c47c0b2c90a0c41cb4c62']):
        im, depth, to_world, cam = load_matterport360_example(path)
        save_pointcloud(os.path.join(out_dir, 'matterport_%d.ply' %
                        i), im, depth, to_world, cam, True)

    for i in range(0, 5):
        im, depth, to_world, cam = load_scannet_example(view_num=i)
        save_pointcloud(os.path.join(out_dir, 'scannet_%d.ply' %
                        i), im, depth, to_world, cam, False)


def example_warp_to_other_models():
    """ Demonstrates warping one camera model to another with resample by intrinsics. Possibly with
        an extrinsic rotation.
        Also demonstrates heterogeneous batching.
    """
    out_dir = os.path.join(examples_output, 'warp_to_other_model')
    os.makedirs(out_dir, exist_ok=True)
    im, depth, to_world, cam = load_matterport360_example(
        'example_scripts/data/matterport360/0c3c242b5567468889d3c66eb931d6e8')
    pin_cam = cameras.PinholeCamera.make(torch.eye(3))

    intrin = torch.eye(3)
    intrin[0, 0] = 1.5
    intrin[1, 1] = 1.5
    k1 = 1.6798e-2
    k2 = 1.6548
    xi = 2.213
    fisheye_cam = cameras.Kitti360FisheyeCamera.make(
        intrin, k1, k2, xi, 3.14/2)

    both_cams = torch.stack([pin_cam, fisheye_cam], dim=0)

    resampled_images_and_depths, _ = warpings.resample_by_intrinsics([im.unsqueeze(0).expand(2, -1, -1, -1), depth.unsqueeze(0).expand(2, -1, -1, -1,)],
                                                                     cam.unsqueeze(
                                                                         0).expand(2),
                                                                     both_cams,
                                                                     (512, 512),
                                                                     interp_mode=['bilinear', 'nearest'])
    resampled_images = resampled_images_and_depths[0]
    resampled_depths = resampled_images_and_depths[1]

    write_image(os.path.join(out_dir, 'original_image.png'), im)
    for i, save_name in enumerate(['pin', 'fish']):
        write_image(os.path.join(out_dir, 'resampled_image_%s.png' %
                    save_name), resampled_images[i])
        imageio.imwrite(os.path.join(out_dir, 'resampled_depth_%s.png' % save_name),
                        (resampled_depths[i].squeeze()*25).numpy().astype(np.uint8))

    # now do the same thing with a 180 degree extrinsic rotation
    rotation_trg_to_src = torch.tensor([[-1, 0, 0],
                                       [0, 1, 0],
                                        [0,  0, -1]]).unsqueeze(0).expand(2, -1, -1)
    resampled_images_and_depths, _ = warpings.resample_by_intrinsics([im.unsqueeze(0).expand(2, -1, -1, -1), depth.unsqueeze(0).expand(2, -1, -1, -1,)],
                                                                     cam.unsqueeze(
                                                                         0).expand(2),
                                                                     both_cams,
                                                                     (512, 512),
                                                                     interp_mode=[
                                                                         'bilinear', 'nearest'],
                                                                     rotation_trg_to_src=rotation_trg_to_src, depth_is_along_ray=False)
    resampled_images = resampled_images_and_depths[0]
    resampled_depths = resampled_images_and_depths[1]

    for i, save_name in enumerate(['pin', 'fish']):
        write_image(os.path.join(out_dir, 'resampled_image_rot_%s.png' %
                    save_name), resampled_images[i])
        imageio.imwrite(os.path.join(out_dir, 'resampled_depth_rot_%s.png' %
                        save_name), (resampled_depths[i].squeeze()*25).numpy().astype(np.uint8))


def example_warp_to_cubemap():
    """ Similar to example_warp_to_other_models, warp ERP to cubemap and cubemap back to ERP.
        Demonstrates format of nvtorchcam cubemap format i.e. (*, channels, 6w, w) and utility
        function utils.flatten_cubemap_for_visual
    """
    out_dir = os.path.join(examples_output, 'resample_to_cubemap')
    os.makedirs(out_dir, exist_ok=True)
    device = 'cuda'
    im, depth, to_world, cam = load_matterport360_example(
        'example_scripts/data/matterport360/0c3c242b5567468889d3c66eb931d6e8', device=device)

    cube_cam = cameras.CubeCamera.make((), device=device)
    cube_images_and_depths, _ = warpings.resample_by_intrinsics([im, depth],
                                                                cam,
                                                                cube_cam,
                                                                (6*512, 512),
                                                                interp_mode=['bilinear', 'nearest'])
    cube_image = cube_images_and_depths[0]
    cube_depth = cube_images_and_depths[1]
    write_image(os.path.join(out_dir, 'image.png'), im)
    write_image(os.path.join(out_dir, 'cube_image.png'),
                utils.flatten_cubemap_for_visual(cube_image, mode=0))

    erp_images_and_depths, _ = warpings.resample_by_intrinsics([cube_image, cube_depth],
                                                               cube_cam,
                                                               cam,
                                                               im.shape[-2:],
                                                               interp_mode=['bilinear', 'nearest'])
    erp_image = erp_images_and_depths[0]
    erp_depth = erp_images_and_depths[1]
    write_image(os.path.join(out_dir, 'erp_image.png'), erp_image)


def example_backward_warp():
    """ Demonstrates backward warping a source to a target based on target depth with ERP and pinholes"""
    out_dir = os.path.join(examples_output, 'backward_warp')
    os.makedirs(out_dir, exist_ok=True)
    trg_image, trg_depth, trg_to_world, trg_cam = load_matterport360_example(
        'example_scripts/data/matterport360/0c3c242b5567468889d3c66eb931d6e8')
    src_image, src_depth, src_to_world, src_cam = load_matterport360_example(
        'example_scripts/data/matterport360/bf85972c0d1c47c0b2c90a0c41cb4c62')
    trg_cam_to_src_cam = torch.mm(torch.inverse(src_to_world), trg_to_world)
    src_image_warped, src_pts, valid_mask = warpings.backward_warp(
        trg_cam, trg_depth, src_cam, src_image, trg_cam_to_src_cam, depth_is_along_ray=True)
    src_image_warped[:, ~valid_mask] = 0
    write_image(os.path.join(out_dir, 'trg_image_erp.png'), trg_image)
    write_image(os.path.join(out_dir, 'src_image_erp.png'), src_image)
    write_image(os.path.join(out_dir, 'src_image_warped_erp.png'),
                src_image_warped.squeeze(1))

    trg_image, trg_depth, trg_to_world, trg_cam = load_scannet_example(
        view_num=0)
    src_image, src_depth, src_to_world, src_cam = load_scannet_example(
        view_num=1)
    trg_cam_to_src_cam = torch.mm(torch.inverse(src_to_world), trg_to_world)
    src_image_warped, src_pts, valid_mask = warpings.backward_warp(
        trg_cam, trg_depth, src_cam, src_image, trg_cam_to_src_cam, depth_is_along_ray=False)
    src_image_warped[:, ~valid_mask] = 0
    write_image(os.path.join(out_dir, 'trg_image_pin.png'), trg_image)
    write_image(os.path.join(out_dir, 'src_image_pin.png'), src_image)
    write_image(os.path.join(out_dir, 'src_image_warped_pin.png'),
                src_image_warped.squeeze(1))


def example_backward_warp_sphere():
    """ Demonstrates backward warping based on target depth of two synthetic sphere images rendered with normal map texture.
        Also demonstrates mixed interp_mode when warping multiple images simultaneously: bilinear for image and nearest for distance
    """
    out_dir = os.path.join(examples_output, 'backward_warp_sphere')
    os.makedirs(out_dir, exist_ok=True)
    cam = cameras.OrthographicCamera.make(torch.eye(3))
    trg_to_world = torch.eye(4)
    trg_to_world[2, 3] = -4
    res = (1024, 1024)
    trg_image, trg_distance, _ = warpings.render_sphere_image(
        cam, trg_to_world, res, 1)

    write_image(os.path.join(out_dir, 'trg_image.png'), (trg_image+1)/2)
    write_image(os.path.join(out_dir, 'trg_distance.png'), trg_distance-3)

    src_to_world = torch.tensor([[0, 0, -1, 4],
                                 [0, 1,  0, 0],
                                 [1, 0,  0, 0],
                                 [0, 0,  0, 1.0]])

    src_image, src_distance, _ = warpings.render_sphere_image(
        cam, src_to_world, res, 1)

    write_image(os.path.join(out_dir, 'src_image.png'), (src_image+1)/2)
    write_image(os.path.join(out_dir, 'src_distance.png'), src_distance-3)

    trg_to_src = torch.mm(torch.inverse(src_to_world), trg_to_world)
    src_warped_and_dist, _, valid_mask = warpings.backward_warp(cam, trg_distance.squeeze(0), cam, [
                                                                src_image, src_distance], trg_to_src, depth_is_along_ray=True, interp_mode=['bilinear', 'nearest'])
    src_warped = src_warped_and_dist[0]
    src_distance_warped = src_warped_and_dist[1]

    write_image(os.path.join(out_dir, 'src_warped.png'), (src_warped+1)/2)
    write_image(os.path.join(out_dir, 'src_distance_warped.png'),
                src_distance_warped-3)
    image_abs_diff = torch.abs(trg_image - src_warped)
    write_image(os.path.join(out_dir, 'image_abs_diff.png'),
                image_abs_diff.clamp(min=0, max=1))


def example_mvsnet_fusion():
    """ 
        Loads one reference view from scannet and two overlapping source views and their ground truth depthmaps. Adds noise to
        the depth maps and saves each noisy unprojected depth as a pointcloud. Then filters the reference noisy depth based on the
        source depths and saves the filtered/fused pointcloud.
    """
    out_dir = os.path.join(examples_output, 'mvsnet_fusion')
    os.makedirs(out_dir, exist_ok=True)

    images = []
    to_worlds = []
    depths = []
    cameras = []
    for i in range(0, 3):
        im, depth, to_world, cam = load_scannet_example(view_num=i)
        noise_mask = torch.rand(depth.shape) > 0.99
        depth = depth + torch.randn(depth.shape)*noise_mask  # add noise

        save_pointcloud(os.path.join(out_dir, 'pc_%d.ply' % i),
                        im, depth, to_world, cam, depth_is_along_ray=False)
        images.append(im)
        depths.append(depth)
        to_worlds.append(to_world)
        cameras.append(cam)

    images = torch.stack(images)
    depths = torch.stack(depths)
    to_worlds = torch.stack(to_worlds)
    cameras = torch.stack(cameras)

    trg_to_world = to_worlds[0:1].expand(2, -1, -1)
    src_to_world = to_worlds[1:]
    trg_to_src = torch.bmm(torch.inverse(src_to_world), trg_to_world)
    fused_depth, fused_valid = warpings.fuse_depths_mvsnet(cameras[0], depths[0].squeeze(
        0), cameras[1:], depths[1:].squeeze(1), trg_to_src, depth_is_along_ray=False, num_image_threshold=3)
    fused_depth[~fused_valid] = torch.nan
    save_pointcloud(os.path.join(out_dir, 'pc_0_filtered.ply'),
                    images[0],  fused_depth, to_worlds[0], cameras[0], depth_is_along_ray=False)


def example_stereo_rectify():
    """ demonstrates stereo rectifying of onto pinholes and the analogous rectification onto ERPs """
    out_dir = os.path.join(examples_output, 'stereo_rectify')
    os.makedirs(out_dir, exist_ok=True)
    image0, depth0, to_world0, cam0 = load_matterport360_example(
        'example_scripts/data/matterport360/0c3c242b5567468889d3c66eb931d6e8', device='cuda')
    image1, depth1, to_world1, cam1 = load_matterport360_example(
        'example_scripts/data/matterport360/bf85972c0d1c47c0b2c90a0c41cb4c62', device='cuda')
    images = torch.stack((image0, image1), dim=0)
    to_worlds = torch.stack((to_world0, to_world1), dim=0)
    cams = torch.stack((cam0, cam1), dim=0)
    new_images, new_to_worlds, valid_mask = warpings.stereo_rectify(
        images, cams, to_worlds, (512, 1024))

    write_image(os.path.join(out_dir, 'original_top.png'), images[0])
    write_image(os.path.join(out_dir, 'original_bottom.png'), images[1])
    write_image(os.path.join(out_dir, 'rectified_top.png'), new_images[0])
    write_image(os.path.join(out_dir, 'rectified_bottom.png'), new_images[1])

    trg_cams = cameras.PinholeCamera.make(
        torch.eye(3).unsqueeze(0).expand(2, -1, -1))
    new_images, new_to_worlds, valid_mask = warpings.stereo_rectify(
        images, cams, to_worlds, (512, 1024), trg_cams=trg_cams, top_bottom=False)

    write_image(os.path.join(out_dir, 'original_left.png'), images[0])
    write_image(os.path.join(out_dir, 'original_right.png'), images[1])
    write_image(os.path.join(out_dir, 'rectified_left.png'), new_images[0])
    write_image(os.path.join(out_dir, 'rectified_right.png'), new_images[1])


def example_make_cost_volume_synthetic_sphere():
    """ Demonstrates cost-volume construction via backward warping base on a number of hypothesis depths (i.e. plane sweep).
        Uses three images of a synthetic sphere with normal map texture. One from the front to serve as a
        reference and two sources from the left and right.

    """
    out_dir = os.path.join(
        examples_output, 'make_cost_volume_sphere_synthetic_sphere')
    os.makedirs(out_dir, exist_ok=True)

    # setup poses and cameras
    ref_to_world = torch.eye(4)
    ref_to_world[2, 3] = -2
    R = torch.tensor([[0.7071,  0.0000,  0.7071,  0.0000],
                      [0.0000,  1.0000,  0.0000,  0.0000],
                      [-0.7071,  0.0000,  0.7071,  0.0000],
                      [0.0000,  0.0000,  0.0000,  1.0000]])  # 45 degree rotation

    src_right_to_world = torch.mm(R.T, ref_to_world)
    src_left_to_world = torch.mm(R, ref_to_world)
    to_worlds = torch.stack(
        (ref_to_world, src_right_to_world, src_left_to_world))
    trg_cam_to_src_cam = torch.bmm(torch.inverse(
        to_worlds[1:]), to_worlds[0:1].expand(2, -1, -1))
    cams = cameras.PinholeCamera.make(
        intrinsics=torch.eye(3).unsqueeze(0).expand(3, -1, -1))

    # render synthetic images and save
    images, _, depth = warpings.render_sphere_image(
        cams, to_worlds, (512, 512))
    write_image(os.path.join(out_dir, 'image_ref.png'), (images[0]+1)/2)
    write_image(os.path.join(out_dir, 'image_right.png'), (images[1]+1)/2)
    write_image(os.path.join(out_dir, 'image_left.png'), (images[2]+1)/2)

    # make depth hypotheses
    depth_hypotheses = 1/torch.linspace(1/0.8, 1/2, 48)

    # expand by reference image shape. Note that one could uses per-pixel hypothesis distances
    # if your implementing something like CasMVSNet
    # (num_hypos=32, h, w)
    depth_hypotheses = depth_hypotheses.reshape(
        -1, 1, 1).expand(-1, *images[0].shape[-2:])

    # backward warp based on depth_hypotheses
    warped_srcs, _, _ = warpings.backward_warp(
        cams[0], depth_hypotheses, cams[1:], images[1:], trg_cam_to_src_cam, depth_is_along_ray=False)  # (num_src,c,num_hypos,h,w)
    ref_volume = images[0].unsqueeze(0).unsqueeze(2)  # (1,c,1,512,512)

    # measure dot product similiarity of images. Only at the correct depth should the normals exactly match i.e.
    # the dot product should be 1
    # (num_src,num_hypos,512,512)
    dot = torch.sum(ref_volume*warped_srcs, dim=1)

    for src_num in range(dot.size(0)):
        for dh in range(dot.size(1)):
            dot_im = dot[src_num, dh, :, :].unsqueeze(0)
            dot_im = dot_im.clamp(min=0)**512  # soft version of dot_im == 1
            write_image(os.path.join(out_dir, 'similarity_src%d_depth%f.png' % (
                src_num, depth_hypotheses[dh, 0, 0])), dot_im)


def example_make_cost_volume_sphere_sweep_erp():
    """ Demonstrates cost-volume construction via backward warping base on a number of hypothesis distances (i.e. sphere sweep)
        For ERP camera model. See example_make_cost_volume_synthetic_sphere for more details
    """
    device = 'cuda'
    out_dir = os.path.join(
        examples_output, 'make_cost_volume_sphere_sweep_erp')
    os.makedirs(out_dir, exist_ok=True)

    trg_image, trg_dist, trg_to_world, trg_cam = load_matterport360_example(
        'example_scripts/data/matterport360/0c3c242b5567468889d3c66eb931d6e8', device=device)
    src_image, src_dist, src_to_world, src_cam = load_matterport360_example(
        'example_scripts/data/matterport360/bf85972c0d1c47c0b2c90a0c41cb4c62', device=device)
    trg_image = torch.nn.functional.interpolate(
        trg_image.unsqueeze(0), (512, 1024)).squeeze(0)
    src_image = torch.nn.functional.interpolate(
        src_image.unsqueeze(0), (512, 1024)).squeeze(0)

    trg_cam_to_src_cam = torch.mm(torch.inverse(src_to_world), trg_to_world)
    # save_images
    write_image(os.path.join(out_dir, 'image_ref.png'), trg_image)
    write_image(os.path.join(out_dir, 'image_src.png'), src_image)

    dist_hypotheses = 1/torch.linspace(1/0.5, 1/5, 48, device=device)
    # (num_hypos=32, h, w)
    dist_hypotheses = dist_hypotheses.reshape(-1,
                                              1, 1).expand(-1, *trg_image.shape[-2:])
    # backward warp depth_is_along_ray means we do sphere sweep
    warped_srcs, _, _ = warpings.backward_warp(
        trg_cam, dist_hypotheses, src_cam, src_image, trg_cam_to_src_cam, depth_is_along_ray=True)  # (c,num_hypos,h,w)

    for dh in range(dist_hypotheses.size(0)):
        im = warped_srcs[:, dh, :, :]
        write_image(os.path.join(out_dir, 'warped_src_dist%f.png' %
                    dist_hypotheses[dh, 0, 0]), im)


def example_random_crop_flip():
    """ Demonstrate random crop and flip module """
    out_dir = os.path.join(examples_output, 'random_crop_flip')
    os.makedirs(out_dir, exist_ok=True)

    #load example views
    images = []
    to_worlds = []
    depths = []
    cameras = []
    for i in range(0, 3):
        im, depth, to_world, cam = load_scannet_example(view_num=i)
        images.append(im)
        depths.append(depth)
        to_worlds.append(to_world)
        cameras.append(cam)

    images = torch.stack(images).unsqueeze(0)
    depths = torch.stack(depths).unsqueeze(0)
    to_worlds = torch.stack(to_worlds).unsqueeze(0)
    cameras = torch.stack(cameras).unsqueeze(0)

    def save_images_depths_point_clouds(path, images, depths, to_worlds, cameras):
        os.makedirs(path, exist_ok=True)
        for b in range(images.size(0)):
            for i in range(images.size(1)):
                write_image(os.path.join(path, 'image_batch_%d_view_%d.jpg' % (b,i)), images[b,i])
                write_image(os.path.join(path, 'depth_batch_%d_view_%d.jpg' % (b,i)), depths[b,i]/5)
                save_pointcloud(os.path.join(path, 'pc_batch_%d_view_%d.ply' % (b,i)), 
                                images[b,i], 
                                depths[b,i], to_worlds[b,i], 
                                cameras[b,i], 
                                False)

    def demonstate_random_resize_crop_by_params(out_folder, rrcf_params):
        #demonstrate world flipping
        rrcf = warpings.RandomResizedCropFlip(**rrcf_params)
    
        new_images_depths, new_camera, new_to_world, _ = rrcf([images, depths], cameras, to_world=to_worlds)
        new_images = new_images_depths[0]
        new_depths = new_images_depths[1]
        new_to_worlds = new_to_world
        new_cameras = new_camera
        save_images_depths_point_clouds(out_folder, new_images, new_depths, new_to_worlds, new_cameras)

    #save originals and unprojected pointclouds
    save_images_depths_point_clouds(os.path.join(out_dir, 'original'), images, depths, to_worlds, cameras)
    
    #demonstrate flipping with world flipping
    out_folder = os.path.join(out_dir, 'flipped_world')
    params = {'scale': (1.0,1.0),
              'ratio': (1.0,1.0),
              'flip_probability': 1.0,
              'mode': 'width_aspect',
              'share_crop_across_views': True,
              'interp_mode': ['bilinear','nearest'],
              'world_flip': True,
              'out_shape': None}
    demonstate_random_resize_crop_by_params(out_folder, params)


    #demonstrate flipping without world flipping 
    out_folder = os.path.join(out_dir, 'flipped_without_world')
    params = {'scale': (1.0,1.0),
              'ratio': (1.0,1.0),
              'flip_probability': 1.0,
              'mode': 'width_aspect',
              'share_crop_across_views': True,
              'interp_mode': ['bilinear','nearest'],
              'world_flip': False,
              'out_shape': None}
    demonstate_random_resize_crop_by_params(out_folder, params)

    #demonstrate share_crop_across_views=True
    out_folder = os.path.join(out_dir, 'shared_cropping')
    params = {'scale': (0.5,0.5),
              'ratio': (1.0,1.0),
              'flip_probability': 0.0,
              'mode': 'width_aspect',
              'share_crop_across_views': True,
              'interp_mode': ['bilinear','nearest'],
              'world_flip': False,
              'out_shape': None}
    demonstate_random_resize_crop_by_params(out_folder, params)

    #demonstrate share_crop_across_views=False
    out_folder = os.path.join(out_dir, 'not_shared_cropping')
    params = {'scale': (0.5,0.5),
              'ratio': (1.0,1.0),
              'flip_probability': 0.0,
              'mode': 'width_aspect',
              'share_crop_across_views': False,
              'interp_mode': ['bilinear','nearest'],
              'world_flip': False,
              'out_shape': None}
    demonstate_random_resize_crop_by_params(out_folder, params)

    #demonstrate torchvision cropping
    out_folder = os.path.join(out_dir, 'torchvision')
    params = {'scale': (0.08, 1.0),
              'ratio': (3/4, 4/3),
              'flip_probability': 0.0,
              'mode': 'torchvision',
              'share_crop_across_views': False,
              'interp_mode': ['bilinear','nearest'],
              'world_flip': True,
              'out_shape': None}
    demonstate_random_resize_crop_by_params(out_folder, params)


def list_available_examples():
    print("Available example names:")
    for name in function_mapping.keys():
        print(name)


if __name__ == '__main__':
    function_mapping = {
        'save_pointclouds': example_save_pointclouds,
        'warp_to_other_models': example_warp_to_other_models,
        'resample_to_cubemap': example_warp_to_cubemap,
        'backward_warp': example_backward_warp,
        'backward_warp_sphere': example_backward_warp_sphere,
        'mvsnet_fusion': example_mvsnet_fusion,
        'stereo_rectify': example_stereo_rectify,
        'make_cost_volume_synthetic_sphere': example_make_cost_volume_synthetic_sphere,
        'make_cost_volume_sphere_sweep_erp': example_make_cost_volume_sphere_sweep_erp,
        'random_crop_flip': example_random_crop_flip
    }

    parser = argparse.ArgumentParser(
        description='Run a function based on the example name.')
    parser.add_argument('example_name', nargs='?',
                        help='Name of the example to run')
    args = parser.parse_args()

    if args.example_name:
        if args.example_name in function_mapping:
            function_mapping[args.example_name]()
        else:
            print(f"No example named '{args.example_name}'.")
            list_available_examples()
    else:
        list_available_examples()
