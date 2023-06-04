from pytorch3d.renderer import (
    BlendParams, blending, look_at_view_transform, FoVOrthographicCameras,
    PointLights, RasterizationSettings, PointsRasterizationSettings,
    PointsRenderer, AlphaCompositor, PointsRasterizer, MeshRenderer,
    MeshRasterizer, SoftPhongShader, SoftSilhouetteShader, TexturesVertex)
from pytorch3d.renderer.mesh import TexturesVertex
from pytorch3d.structures import Meshes, Pointclouds
import torch
import numpy as np


class cleanShader(torch.nn.Module):
    def __init__(self, device="cpu", cameras=None, blend_params=None):
        super().__init__()
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams(
        )

    def forward(self, fragments, meshes, **kwargs):
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of TexturedSoftPhongShader"

            raise ValueError(msg)

        # get renderer output
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        images = blending.softmax_rgb_blend(texels,
                                            fragments,
                                            blend_params,
                                            znear=-256,
                                            zfar=256)

        return images


class Render:
    def __init__(self, size=512, device=torch.device("cuda:0")):
        self.device = device
        self.mesh_y_center = 100.0
        self.dis = 100.0
        self.scale = 1.0
        self.size = size
        self.cam_pos = [(0, 100, 100)]

        self.mesh = None
        self.pcd = None
        self.renderer = None
        self.meshRas = None

    def get_camera(self, cam_id):
        # at

        R, T = look_at_view_transform(eye=[self.cam_pos[cam_id]],
                                      at=((0, self.mesh_y_center, 0),),
                                      up=((0, 1, 0),))

        camera = FoVOrthographicCameras(device=self.device,
                                        R=R,
                                        T=T,
                                        znear=100.0,
                                        zfar=-100.0,
                                        max_y=100.0,
                                        min_y=-100.0,
                                        max_x=100.0,
                                        min_x=-100.0,
                                        scale_xyz=(self.scale * np.ones(3),))
        return camera

    def init_renderer(self, camera, type='clean_mesh', bg='gray'):

        if 'mesh' in type:
            # rasterizer
            self.raster_settings_mesh = RasterizationSettings(
                image_size=self.size,
                blur_radius=np.log(1.0 / 1e-4) * 1e-7,
                faces_per_pixel=30,
            )
            self.meshRas = MeshRasterizer(cameras=camera,
                                          raster_settings=self.raster_settings_mesh)

        if bg == 'black':
            blendparam = BlendParams(1e-4, 1e-4, (0.0, 0.0, 0.0))
        elif bg == 'white':
            blendparam = BlendParams(1e-4, 1e-8, (1.0, 1.0, 1.0))
        elif bg == 'gray':
            blendparam = BlendParams(1e-4, 1e-8, (0.5, 0.5, 0.5))

        if type == 'ori_mesh':
            lights = PointLights(device=self.device,
                                 ambient_color=((0.8, 0.8, 0.8),),
                                 diffuse_color=((0.2, 0.2, 0.2),),
                                 specular_color=((0.0, 0.0, 0.0),),
                                 location=[[0.0, 200.0, 0.0]])

            self.renderer = MeshRenderer(rasterizer=self.meshRas,
                                         shader=SoftPhongShader(
                                             device=self.device,
                                             cameras=camera,
                                             lights=lights,
                                             blend_params=blendparam))

        if type == 'silhouette':
            self.raster_settings_silhouette = RasterizationSettings(
                image_size=self.size,
                blur_radius=np.log(1. / 1e-4 - 1.) * 5e-5,
                faces_per_pixel=50,
                cull_backfaces=True,
            )

            self.silhouetteRas = MeshRasterizer(
                cameras=camera, raster_settings=self.raster_settings_silhouette)
            self.renderer = MeshRenderer(rasterizer=self.silhouetteRas,
                                         shader=SoftSilhouetteShader())

        if type == 'pointcloud':
            self.raster_settings_pcd = PointsRasterizationSettings(
                image_size=self.size,
                radius=0.006,
                points_per_pixel=10)

            self.pcdRas = PointsRasterizer(cameras=camera,
                                           raster_settings=self.raster_settings_pcd)
            self.renderer = PointsRenderer(
                rasterizer=self.pcdRas,
                compositor=AlphaCompositor(background_color=(0, 0, 0)))

        if type == 'clean_mesh':
            self.renderer = MeshRenderer(
                rasterizer=self.meshRas,
                shader=cleanShader(device=self.device,
                                   cameras=camera,
                                   blend_params=blendparam))

    def set_camera(self, verts, scale=0.95):
        self.mesh_y_center = (verts.max(dim=1)[0][0, 1] + verts.min(dim=1)[0][0, 1]).item() / 2.
        self.cam_pos = [(0, self.mesh_y_center, self.dis),
                        (self.dis, self.mesh_y_center, 0),
                        (0, self.mesh_y_center, -self.dis),
                        (-self.dis, self.mesh_y_center, 0)]

        hight = (verts.max(dim=1)[0][0, 1] - verts.min(dim=1)[0][0, 1]).item() / 2.
        self.scale = scale * 100 / (hight + 1e-10)

    def load_mesh(self, verts, faces, verts_rgb=None):
        """load mesh into the pytorch3d renderer

        Args:
            verts ([N,3]): verts
            faces ([N,3]): faces
            verts_rgb ([N,3]): rgb
        """

        if not torch.is_tensor(verts):
            verts = torch.tensor(verts)
        if not torch.is_tensor(faces):
            faces = torch.tensor(faces)

        if verts.ndimension() == 2:
            verts = verts.unsqueeze(0).float()
        if faces.ndimension() == 2:
            faces = faces.unsqueeze(0).long()

        self.set_camera(verts)
        self.mesh = Meshes(verts, faces)


        if verts_rgb is not None:
            if not torch.is_tensor(verts_rgb):
                verts_rgb = torch.as_tensor(verts_rgb) * 2 - 1
            if verts_rgb.ndimension() == 2:
                verts_rgb = verts_rgb.unsqueeze(0).float()
            verts_rgb = verts_rgb
        else:
            verts_rgb = self.mesh.verts_normals_padded()[..., 2:].expand(-1, -1, 3)
            # rgb = self.mesh.verts_normals_padded()
        textures = TexturesVertex(verts_rgb)
        self.mesh.textures = textures
        self.mesh = self.mesh.to(device=self.device)
        return self.mesh

    def load_mesh_with_text(self, mesh):
        self.set_camera(mesh.verts_packed()[None])
        self.mesh = mesh.to(self.device)

    def load_pcd(self, verts, verts_rgb):
        """load pointcloud into the pytorch3d renderer

        Args:
            verts ([B, N,3]): verts
            verts_rgb ([B, N,3]): verts colors
        """
        assert verts.shape == verts_rgb.shape and len(verts.shape) == 3
        # data format convert
        if not torch.is_tensor(verts):
            verts = torch.as_tensor(verts)
        if not torch.is_tensor(verts_rgb):
            verts_rgb = torch.as_tensor(verts_rgb)

        verts = verts.float().to(self.device)
        verts_rgb = verts_rgb.float().to(self.device)

        # camera setting
        self.set_camera(verts)
        pcd = Pointclouds(points=verts, features=verts_rgb).to(self.device)
        return pcd

    def get_image(self, cam_id=[0, 1, 2, 3]):
        images = torch.zeros(
            (self.size, self.size * len(cam_id), 3)).to(self.device)
        for i, cam_id in enumerate(cam_id):
            self.init_renderer(self.get_camera(cam_id), 'ori_mesh', 'gray')
            images[:, self.size * i:self.size *
                                    (i + 1), :] = self.renderer(self.mesh)[0, :, :, :3]

        return images.cpu().numpy()

    def get_clean_image(self, cam_ids=[0, 2], type='clean_mesh', bg='gray'):
        images = []
        for cam_id in range(len(self.cam_pos)):
            if cam_id in cam_ids:
                self.init_renderer(self.get_camera(cam_id), type, bg)
                if len(cam_ids) == 4:
                    rendered_img = (self.renderer(
                        self.mesh)[0:1, :, :, :3].permute(0, 3, 1, 2) -
                                    0.5) * 2.0
                else:
                    rendered_img = (self.renderer(
                        self.mesh)[0:1, :, :, :3].permute(0, 3, 1, 2) - 0.5) * 2.0
                if cam_id == 2 and len(cam_ids) == 2:
                    rendered_img = torch.flip(rendered_img, dims=[3])
                images.append(rendered_img)

        return images

    def get_silhouette_image(self, cam_ids=[0, 2]):
        images = []
        for cam_id in range(len(self.cam_pos)):
            if cam_id in cam_ids:
                self.init_renderer(self.get_camera(cam_id), 'silhouette')
                rendered_img = self.renderer(self.mesh)[0:1, :, :, 3]
                if cam_id == 2 and len(cam_ids) == 2:
                    rendered_img = torch.flip(rendered_img, dims=[2])
                images.append(rendered_img)

        return images

    def get_image_pcd(self, pcd, cam_ids=[0, 1, 2, 3]):
        images = torch.zeros((self.size, self.size * len(cam_ids), 3)).to(self.device)
        for i, cam_id in enumerate(cam_ids):
            self.init_renderer(self.get_camera(cam_id), 'pointcloud')
            images[:, self.size * i:self.size * (i + 1), :] = self.renderer(pcd)[0, :, :, :3]

        return images.cpu().numpy()

