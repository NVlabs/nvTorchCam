import unittest
import torch

from nvtorchcam import cameras
from nvtorchcam import warpings



class TestRenderSphereImage(unittest.TestCase):
    def setUp(self):
        pass

    # @unittest.skip('Uncomment to skip this test') 
    def test_ray_sphere_intersection(self):
        origin = torch.tensor([ [0,0,-2], #hit at lower time
                                [0,0,0], #hit at higher time
                                [0,0,2], #invalid b/c hit at negative time
                                [2,0,0.0] ]) #invalid b/c negative discriminant
        dirs = torch.tensor([0,0,1.0]).unsqueeze(0).expand(4,-1)
        intersection, distance, valid = warpings.ray_sphere_intersection(origin, dirs)
        torch.testing.assert_close( valid, torch.tensor([ True,  True, False, False]) )
        torch.testing.assert_close( distance, torch.tensor([ 1.,  1., -1.,  0.]) )
        torch.testing.assert_close( intersection[valid,:], torch.tensor([[ 0.,  0., -1.],
                                                                         [ 0.,  0.,  1.]]))

        #test batching
        origin = origin.unsqueeze(0).unsqueeze(0).expand(2,3,-1,-1)
        dirs = dirs.unsqueeze(0).unsqueeze(0).expand(2,3,-1,-1)
        intersection_b, distance_b, valid_b = warpings.ray_sphere_intersection(origin, dirs)
        
        torch.testing.assert_close(intersection.unsqueeze(0).unsqueeze(0).expand(2,3,-1,-1), intersection_b)
        torch.testing.assert_close(valid.unsqueeze(0).unsqueeze(0).expand(2,3,-1), valid_b)
        torch.testing.assert_close(distance.unsqueeze(0).unsqueeze(0).expand(2,3,-1), distance_b)

    
    # @unittest.skip('Uncomment to skip this test')    
    def test_render_sphere_image(self):
        k1 = torch.tensor(1.6798e-2)
        k2 = torch.tensor(1.6548)
        xi =  torch.tensor(2.213)
        theta_max = 3.14/2
        distance_min = 0.0

        intrinsics = torch.eye(3)
        intrinsics[0,0] = 6
        intrinsics[1,1] = 6
        cam_fish = cameras.Kitti360FisheyeCamera.make(intrinsics,k1,k2,xi,theta_max, distance_min)
       
        to_world = torch.eye(4)
        to_world[2,3] = -4
        res = (8,8)
        image_fish, distance_fish = warpings.render_sphere_image(cam_fish, to_world, res, 1)
  
        nan = torch.nan
        expected_image = torch.tensor([[[    nan,     nan,     nan,     nan,     nan,     nan,     nan,
              nan],
         [    nan,     nan,     nan,     nan,     nan,     nan,     nan,
              nan],
         [    nan,     nan,     nan, -0.2239,  0.2239,     nan,     nan,
              nan],
         [    nan,     nan, -0.6718, -0.2043,  0.2043,  0.6718,     nan,
              nan],
         [    nan,     nan, -0.6718, -0.2043,  0.2043,  0.6718,     nan,
              nan],
         [    nan,     nan,     nan, -0.2239,  0.2239,     nan,     nan,
              nan],
         [    nan,     nan,     nan,     nan,     nan,     nan,     nan,
              nan],
         [    nan,     nan,     nan,     nan,     nan,     nan,     nan,
              nan]],

        [[    nan,     nan,     nan,     nan,     nan,     nan,     nan,
              nan],
         [    nan,     nan,     nan,     nan,     nan,     nan,     nan,
              nan],
         [    nan,     nan,     nan, -0.6718, -0.6718,     nan,     nan,
              nan],
         [    nan,     nan, -0.2239, -0.2043, -0.2043, -0.2239,     nan,
              nan],
         [    nan,     nan,  0.2239,  0.2043,  0.2043,  0.2239,     nan,
              nan],
         [    nan,     nan,     nan,  0.6718,  0.6718,     nan,     nan,
              nan],
         [    nan,     nan,     nan,     nan,     nan,     nan,     nan,
              nan],
         [    nan,     nan,     nan,     nan,     nan,     nan,     nan,
              nan]],

        [[    nan,     nan,     nan,     nan,     nan,     nan,     nan,
              nan],
         [    nan,     nan,     nan,     nan,     nan,     nan,     nan,
              nan],
         [    nan,     nan,     nan, -0.7061, -0.7061,     nan,     nan,
              nan],
         [    nan,     nan, -0.7061, -0.9574, -0.9574, -0.7061,     nan,
              nan],
         [    nan,     nan, -0.7061, -0.9574, -0.9574, -0.7061,     nan,
              nan],
         [    nan,     nan,     nan, -0.7061, -0.7061,     nan,     nan,
              nan],
         [    nan,     nan,     nan,     nan,     nan,     nan,     nan,
              nan],
         [    nan,     nan,     nan,     nan,     nan,     nan,     nan,
              nan]]])
        expected_distance = torch.tensor([[[   nan,    nan,    nan,    nan,    nan,    nan,    nan,    nan],
         [   nan,    nan,    nan,    nan,    nan,    nan,    nan,    nan],
         [   nan,    nan,    nan, 3.3692, 3.3692,    nan,    nan,    nan],
         [   nan,    nan, 3.3692, 3.0563, 3.0563, 3.3692,    nan,    nan],
         [   nan,    nan, 3.3692, 3.0563, 3.0563, 3.3692,    nan,    nan],
         [   nan,    nan,    nan, 3.3692, 3.3692,    nan,    nan,    nan],
         [   nan,    nan,    nan,    nan,    nan,    nan,    nan,    nan],
         [   nan,    nan,    nan,    nan,    nan,    nan,    nan,    nan]]])

        torch.testing.assert_close(torch.nan_to_num(image_fish), torch.nan_to_num(expected_image), atol=1e-4, rtol=1e-3)
        torch.testing.assert_close(torch.nan_to_num(distance_fish), torch.nan_to_num(expected_distance), atol=1e-4, rtol=1e-3)
        
        #test with batching
        cam_fish = cam_fish.unsqueeze(0).unsqueeze(0).expand(2,3)
        to_world = to_world.unsqueeze(0).unsqueeze(0).expand(2,3,-1,-1)
        image_fish_b, distance_fish_b = warpings.render_sphere_image(cam_fish, to_world, res, 1)

        torch.testing.assert_close(torch.nan_to_num(image_fish_b), torch.nan_to_num(expected_image.unsqueeze(0).unsqueeze(0).expand(2,3,-1,-1,-1)), atol=1e-4, rtol=1e-3)
        torch.testing.assert_close(torch.nan_to_num(distance_fish_b), torch.nan_to_num(expected_distance.unsqueeze(0).unsqueeze(0).expand(2,3,-1,-1,-1)), atol=1e-4, rtol=1e-3)
        

class TestBackwardWarping(unittest.TestCase):
    def setUp(self):
        pass

    # @unittest.skip('Uncomment to skip this test')    
    def test_backwarp_warp_pts(self):
        cam = cameras.OrthographicCamera.make(torch.eye(3))
        trg_to_world = torch.eye(4)
        trg_to_world[2,3] = -4
        src_to_world = torch.tensor([[0, 0, -1, 4],
                               [0, 1,  0, 0 ],
                               [1, 0,  0, 0 ],
                               [0, 0,  0, 1.0 ]])
        trg_to_src = torch.mm(torch.inverse(src_to_world), trg_to_world)



        res = (8,8)
        trg_image, trg_distance = warpings.render_sphere_image(cam, trg_to_world, res, 1)
        src_pts, src_pts_depth, valid_mask = warpings.backwarp_warp_pts(cam, trg_distance, cam, trg_to_src, depth_is_along_ray=True)
        
        nan = torch.nan
        expected_src_pts = torch.tensor([[[    nan,     nan],
         [    nan,     nan],
         [-0.3062, -0.8750],
         [-0.4677, -0.8750],
         [-0.4677, -0.8750],
         [-0.3062, -0.8750],
         [    nan,     nan],
         [    nan,     nan]],

        [[    nan,     nan],
         [-0.4677, -0.6250],
         [-0.6847, -0.6250],
         [-0.7706, -0.6250],
         [-0.7706, -0.6250],
         [-0.6847, -0.6250],
         [-0.4677, -0.6250],
         [    nan,     nan]],

        [[-0.3062, -0.3750],
         [-0.6847, -0.3750],
         [-0.8478, -0.3750],
         [-0.9186, -0.3750],
         [-0.9186, -0.3750],
         [-0.8478, -0.3750],
         [-0.6847, -0.3750],
         [-0.3062, -0.3750]],

        [[-0.4677, -0.1250],
         [-0.7706, -0.1250],
         [-0.9186, -0.1250],
         [-0.9843, -0.1250],
         [-0.9843, -0.1250],
         [-0.9186, -0.1250],
         [-0.7706, -0.1250],
         [-0.4677, -0.1250]],

        [[-0.4677,  0.1250],
         [-0.7706,  0.1250],
         [-0.9186,  0.1250],
         [-0.9843,  0.1250],
         [-0.9843,  0.1250],
         [-0.9186,  0.1250],
         [-0.7706,  0.1250],
         [-0.4677,  0.1250]],

        [[-0.3062,  0.3750],
         [-0.6847,  0.3750],
         [-0.8478,  0.3750],
         [-0.9186,  0.3750],
         [-0.9186,  0.3750],
         [-0.8478,  0.3750],
         [-0.6847,  0.3750],
         [-0.3062,  0.3750]],

        [[    nan,     nan],
         [-0.4677,  0.6250],
         [-0.6847,  0.6250],
         [-0.7706,  0.6250],
         [-0.7706,  0.6250],
         [-0.6847,  0.6250],
         [-0.4677,  0.6250],
         [    nan,     nan]],

        [[    nan,     nan],
         [    nan,     nan],
         [-0.3062,  0.8750],
         [-0.4677,  0.8750],
         [-0.4677,  0.8750],
         [-0.3062,  0.8750],
         [    nan,     nan],
         [    nan,     nan]]])
        

        expected_src_pts_depth = torch.tensor([[   nan,    nan, 4.3750, 4.1250, 3.8750, 3.6250,    nan,    nan],
         [   nan, 4.6250, 4.3750, 4.1250, 3.8750, 3.6250, 3.3750,    nan],
         [4.8750, 4.6250, 4.3750, 4.1250, 3.8750, 3.6250, 3.3750, 3.1250],
         [4.8750, 4.6250, 4.3750, 4.1250, 3.8750, 3.6250, 3.3750, 3.1250],
         [4.8750, 4.6250, 4.3750, 4.1250, 3.8750, 3.6250, 3.3750, 3.1250],
         [4.8750, 4.6250, 4.3750, 4.1250, 3.8750, 3.6250, 3.3750, 3.1250],
         [   nan, 4.6250, 4.3750, 4.1250, 3.8750, 3.6250, 3.3750,    nan],
         [   nan,    nan, 4.3750, 4.1250, 3.8750, 3.6250,    nan,    nan]])

        expected_valid_mask = torch.tensor([[False, False,  True,  True,  True,  True, False, False],
        [False,  True,  True,  True,  True,  True,  True, False],
        [ True,  True,  True,  True,  True,  True,  True,  True],
        [ True,  True,  True,  True,  True,  True,  True,  True],
        [ True,  True,  True,  True,  True,  True,  True,  True],
        [ True,  True,  True,  True,  True,  True,  True,  True],
        [False,  True,  True,  True,  True,  True,  True, False],
        [False, False,  True,  True,  True,  True, False, False]])

        torch.testing.assert_close(torch.nan_to_num(src_pts), torch.nan_to_num(expected_src_pts), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(torch.nan_to_num(src_pts_depth), torch.nan_to_num(expected_src_pts_depth))
        torch.testing.assert_close(valid_mask, expected_valid_mask)
        
        #test multiple target images
        src_pts_t2, src_pts_depth_t2, valid_mask_t2 = warpings.backwarp_warp_pts(cam, trg_distance, cam.unsqueeze(0).expand(2), trg_to_src.unsqueeze(0).expand(2,-1,-1), depth_is_along_ray=True)
        
        torch.testing.assert_close(torch.nan_to_num(src_pts.unsqueeze(0).expand(2,-1,-1,-1)), torch.nan_to_num(src_pts_t2))
        torch.testing.assert_close(torch.nan_to_num(src_pts_depth.unsqueeze(0).expand(2,-1,-1)), torch.nan_to_num(src_pts_depth_t2))
        torch.testing.assert_close(valid_mask.unsqueeze(0).expand(2,-1,-1), valid_mask_t2)
      


class TestResampleByIntrinsics(unittest.TestCase):
    def setUp(self):
        pass
    
    @unittest.skip('Uncomment to skip this test')    
    def test_resample_by_intrinsics(self):
        cam_pin = cameras.PinholeCamera.make(torch.eye(3))
        k1 = torch.tensor(1.6798e-2)
        k2 = torch.tensor(1.6548)
        xi =  torch.tensor(2.213)
        theta_max = (3.14/2)
        distance_min = 0.0

        intrinsics = torch.eye(3)
        intrinsics[0,0] = 6
        intrinsics[1,1] = 6
        cam_fish = cameras.Kitti360FisheyeCamera.make(intrinsics,k1,k2,xi,theta_max, distance_min)
       
        to_world = torch.eye(4)
        to_world[2,3] = -4
        res = (8,8)
        image_fish, distance_fish = warpings.render_sphere_image(cam_fish, to_world, res, 1)

class TestCropImageResize(unittest.TestCase):
    def setUp(self):
        pass
     
    # @unittest.skip('Uncomment to skip this test')
    def test_crop_resize_image(self):
        image = torch.arange(20*40).reshape(1,20,40).float()
        lrtb = torch.tensor([3,36,5,17])
        crop_image_expected = image[:, lrtb[2]:lrtb[3], lrtb[0]:lrtb[1]]
        crop_image, valid = warpings.crop_resize_image(image, lrtb, normalized=False, out_shape = crop_image_expected.shape[-2:])
        torch.testing.assert_close(crop_image, crop_image_expected)

        #test list
        crop_image2, valid = warpings.crop_resize_image([image, image], lrtb, normalized=False, out_shape = crop_image_expected.shape[-2:], interp_mode=['bilinear','nearest'])
        assert isinstance(crop_image2, list)
        assert len(crop_image2) == 2
        torch.testing.assert_close(crop_image2[0], crop_image)
        torch.testing.assert_close(crop_image2[1], crop_image)


        #test that camera.crop is consistent with crop_resize_image
        cam = cameras.PinholeCamera.make(torch.eye(3))
        origin, dirs, _ = cam.get_camera_rays((20, 40), True)

        #interpret origin and dirs as image i.e. more to channel first
        origin = origin.permute(2,0,1) 
        dirs = dirs.permute(2,0,1)
        [origin_cropped, dirs_cropped], _ = warpings.crop_resize_image([origin, dirs], lrtb, normalized=False, out_shape = crop_image_expected.shape[-2:])
        
        cam_cropped = cam.crop(lrtb, normalized=False, image_shape = dirs.shape[-2:])
        origin_from_cropped_cam, dirs_from_cropped_cam, _ = cam_cropped.get_camera_rays(dirs_cropped.shape[-2:], True)
        
        #back to channels last
        origin_cropped = origin_cropped.permute(1,2,0)
        dirs_cropped = dirs_cropped.permute(1,2,0)
        
        torch.testing.assert_close(dirs_cropped, dirs_from_cropped_cam)
        torch.testing.assert_close(origin_cropped, origin_from_cropped_cam)

        


if __name__ == '__main__':
    unittest.main()
    from io import StringIO
    with unittest.mock.patch('sys.stdout', new=StringIO()) as std_out:
        unittest.main()
