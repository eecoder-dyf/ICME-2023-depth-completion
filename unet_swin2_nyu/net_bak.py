from cv2 import norm
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from swin import SwinTrans
from swin import SwinSeg
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
import numpy as np
from swin_util import *

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PatchExpanding(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = norm_layer(dim)

    def forward(self, x, H, W):
        """
        x: B, H*W, C -> B, 2H*2W, C/2
        """

        x = self.norm(x)
        x = self.expand(x)
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        # assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        assert C % 4 == 0
        x = x.view(B, H, W, C)
        L = H * W
        # print(x.shape)
        y = torch.zeros([B, 2*H, 2*W, C//4], device=x.device)
        y[:, 0::2, 0::2, :] = x[:,:,:,0:C//4]
        y[:, 1::2, 0::2, :] = x[:,:,:,C//4:C//2]
        y[:, 0::2, 1::2, :] = x[:,:,:,C//2:3*C//4]
        y[:, 1::2, 1::2, :] = x[:,:,:,3*C//4:C]

        y = y.view(B, 4*L, C//4)
        

        return y, 2*H, 2*W


class BasicLayerUp(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size,
                mlp_ratio=4, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                drop_path=0, norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.window_size = window_size
        self.shift_size = window_size // 2

        self.blocks = nn.ModuleList([
            SwinSeg.SwinTransformerBlock(dim=dim//2, num_heads=num_heads, window_size=window_size,
                                shift_size=0 if (i % 2 == 0) else window_size // 2,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop, attn_drop=attn_drop,
                                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                norm_layer=norm_layer)
            for i in range(depth)])
        
        if upsample is not None:
            self.upsample = upsample(dim, norm_layer)
        else:
            self.upsample = None

        self.cat_linear = nn.Linear(in_features=dim, out_features=dim//2, bias=False)

        self.norm = norm_layer(dim)

    def forward(self, x, H, W, y=None): # 注意这里计算attn时(H,W)应该减半
        x = self.norm(x)

        if self.upsample is not None:
            x, H, W = self.upsample(x, H, W)    #这里H, W都x2了
        
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = SwinSeg.window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))


        if y is not None:
            # print(x.shape, y.shape)
            assert y.shape == x.shape
            x = torch.cat((x, y), dim=-1)
            x = self.norm(x)
            x = self.cat_linear(x)

        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        
        return x, H, W


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.double_conv(x)
        return x


class ConvPatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=1, embed_dim=48, norm_layer=None):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.convlayer1 = DoubleConv(in_chans, 4*in_chans)
        self.downlayer1 = nn.Conv2d(4*in_chans, 4*in_chans, kernel_size=patch_size//2, stride=patch_size//2)
        self.convlayer2 = DoubleConv(4*in_chans, embed_dim)
        self.downlayer2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=patch_size//2, stride=patch_size//2)

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape

        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
        # print(x.shape)
        x = self.convlayer1(x)
        x = self.downlayer1(x)
        x = self.convlayer2(x)
        x = self.downlayer2(x)
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
        return x

class ConvPatchEmbedRGB(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=48, norm_layer=None):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.convlayer1 = DoubleConv(in_chans, 4*in_chans)
        self.downlayer1 = nn.Conv2d(4*in_chans, 4*in_chans, kernel_size=patch_size//2, stride=patch_size//2)
        self.convlayer2 = nn.Sequential(DoubleConv(4*in_chans, 8*in_chans), DoubleConv(8*in_chans, 8*in_chans))
        self.downlayer2 = nn.Conv2d(8*in_chans, 8*in_chans, kernel_size=patch_size//2, stride=patch_size//2)
        self.convlayer3 = nn.Sequential(DoubleConv(8*in_chans, embed_dim), DoubleConv(embed_dim, embed_dim))

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape

        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
        # print(x.shape)
        x = self.convlayer1(x)
        x = self.downlayer1(x)
        x = self.convlayer2(x)
        x = self.downlayer2(x)
        x = self.convlayer3(x)
        # x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
        return x

class ConvFinalExpanding(nn.Module):
    def __init__(self, dim, patch_size=4, out_chans=1, norm_layer=None):
        super().__init__()
        self.dim = dim
        self.patch_size = float(patch_size)
        self.out_chans = out_chans

        self.uplayer1 = nn.ConvTranspose2d(in_channels=dim, out_channels=dim, kernel_size=patch_size//2, stride=patch_size//2)
        self.convlayer1 = DoubleConv(dim, dim//2)
        self.uplayer2 = nn.ConvTranspose2d(in_channels=dim//2, out_channels=dim//2, kernel_size=patch_size//2, stride=patch_size//2)
        self.convlayer2 = DoubleConv(dim//2, out_chans)

        if norm_layer is not None:
            self.norm = norm_layer(dim)
        else:
            self.norm = None
    
    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = x.view(B, H, W, C)
        if self.norm is not None:
            x = self.norm(x)
        
        x = x.permute(0,3,1,2)
        x = self.uplayer1(x)
        x = self.convlayer1(x)
        x = self.uplayer2(x)
        x = self.convlayer2(x)

        return x

class FuseLayer(nn.Module):
    def __init__(self, dim, num_heads, window_size, mlp_ratio=4, qkv_bias=True, qk_scale=None, drop=0, attn_drop=0.1, drop_path=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.fuse_down_depth = CrossSwinTransformerBlock(dim=dim, num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                                                        qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer)
        self.fuse_down_rgb = CrossSwinTransformerBlock(dim=dim, num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                                                        qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path, norm_layer=norm_layer)
        self.cat_down = nn.Linear(in_features=2*dim, out_features=dim)
        self.H = None
        self.W = None
    def forward(self, x, y):
        # x : depth modality
        # y : rgb modality
        self.fuse_down_depth.H, self.fuse_down_depth.W = self.H, self.W
        self.fuse_down_rgb.H, self.fuse_down_rgb.W = self.H, self.W
        x_attn = self.fuse_down_depth(x, y)
        y_attn = self.fuse_down_rgb(y, x)
        output = torch.cat((x_attn, y_attn), dim=-1)
        output = self.cat_down(output)

        return output

class UnionComp(nn.Module):
    def __init__(self, patch_size=4, in_chans=1, in_chans_rgb=3, embed_dim=48, 
            down_depths=[2,2,6,2], catstep=2, up_depths=[6,2,2], num_heads_down=[2,4,8,16], num_heads_up=[8,4,2], 
            window_size=8, mlp_ratio=4,
            qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.1, drop_path_rate=0.1,
            norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
            use_checkpoint=False, **kwargs):

        super().__init__()
        self.num_downlayers = len(down_depths) # 有多少个stage
        self.num_uplayers = len(up_depths)
        self.catstep = catstep
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features_down = int(embed_dim * 2 ** (self.num_downlayers - 1))
        self.mlp_ratio = mlp_ratio
        self.in_chans = in_chans

        self.patch_embed_depth = ConvPatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None )

        self.patch_embed_rgb = ConvPatchEmbedRGB(
            patch_size=patch_size, in_chans=in_chans_rgb, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None )

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(down_depths))]  # stochastic depth decay rule
        self.layers_down_depth = nn.ModuleList()
        self.layers_down_rgb = nn.ModuleList()
        self.fuse_down = nn.ModuleList()

        for i_layer in range(self.num_downlayers):
            if i_layer <= catstep:
                fuse_down = FuseLayer(dim=embed_dim * 2 ** i_layer, 
                                    num_heads=num_heads_down[i_layer], window_size=window_size, 
                                    mlp_ratio=self.mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop_rate, attn_drop=attn_drop_rate, 
                                    drop_path=0,
                                    norm_layer=norm_layer)
                self.fuse_down.append(fuse_down)

            layer = SwinSeg.BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               depth=down_depths[i_layer],
                               num_heads=num_heads_down[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(down_depths[:i_layer]):sum(down_depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=SwinSeg.PatchMerging if (i_layer < self.num_downlayers - 1) else None,
                               use_checkpoint=use_checkpoint)

            self.layers_down_depth.append(layer)
            self.layers_down_rgb.append(layer)
            


        layerC = self.num_downlayers - 1
        # dimC = embed_dim * 2 ** layerC * 2  #在中间层之前添加Concat
        dimC = embed_dim * 2 ** layerC
        # print(resolutionC)
        self.mid_layer = SwinSeg.BasicLayer(dim=int(dimC), depth = 2, num_heads=16, window_size=window_size, mlp_ratio=self.mlp_ratio, 
                                        qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                                        drop_path=0, norm_layer=norm_layer, downsample=None, use_checkpoint=use_checkpoint)


        self.layers_up = nn.ModuleList()
        for i_layer in range(self.num_uplayers):
            # print(i_layer)
            # print(int(dimC // (2**i_layer)))
            layer = BasicLayerUp(dim=int(dimC // (2**i_layer)), 
                                depth=up_depths[i_layer], num_heads=num_heads_up[i_layer], window_size=window_size,
                                mlp_ratio=4, qkv_bias=True, qk_scale=None,
                                drop=0, attn_drop=0,
                                drop_path=0, norm_layer=norm_layer, upsample=PatchExpanding,
                                use_checkpoint=False)
            self.layers_up.append(layer)        

        dimF = dimC // (2**self.num_uplayers)
        FinalLayer = ConvFinalExpanding(dim=int(dimF), patch_size=patch_size, out_chans=1, norm_layer=None)
        
        self.outlayer = FinalLayer
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)


    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, depth, rgb):
        depth = self.patch_embed_depth(depth)
        rgb = self.patch_embed_rgb(rgb)
        Wh, Ww = depth.size(2), depth.size(3)
        Wh2, Ww2 = Wh, Ww
        depth = depth.flatten(2).transpose(1, 2)
        rgb = rgb.flatten(2).transpose(1,2)

        depth = self.pos_drop(depth)
        rgb = self.pos_drop(rgb)

        i = 0
        f = []
        for i in range(self.num_downlayers):
            if i < self.catstep: 
                self.fuse_down[i].H, self.fuse_down[i].W = Wh, Ww
                fuse = self.fuse_down[i](depth, rgb)
                f.append(fuse)
                _, _, _, depth, Wh, Ww = self.layers_down_depth[i](depth, Wh, Ww)
                _, _, _, rgb, Wh2, Ww2 = self.layers_down_rgb[i](rgb, Wh2, Ww2)
            elif i == self.catstep:
                self.fuse_down[i].H, self.fuse_down[i].W = Wh, Ww
                fusion = self.fuse_down[i](depth, rgb)
                f.append(fusion)
                _, _, _, fusion, Wh, Ww = self.layers_down_depth[i](fusion, Wh, Ww)
            else:
                _, _, _, fusion, Wh, Ww = self.layers_down_depth[i](fusion, Wh, Ww)
            i = i + 1

        # fusion = self.mid_layer(torch.cat((depth, rgb), dim=-1))

        i = 0
        for i in range(self.num_uplayers):
            fusion, Wh, Ww = self.layers_up[i](fusion, Wh, Ww, f[self.num_uplayers-1-i])
            i = i + 1
        
        # print(fusion.shape)
        fusion = self.outlayer(fusion, Wh, Ww)

        return fusion
