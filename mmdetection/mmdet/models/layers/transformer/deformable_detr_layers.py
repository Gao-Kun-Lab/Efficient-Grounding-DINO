# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, Union

import torch
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.ops import MultiScaleDeformableAttention
from mmengine.model import ModuleList
from torch import Tensor, nn
import torch.nn.functional as F

from .detr_layers import (DetrTransformerDecoder, DetrTransformerDecoderLayer,
                          DetrTransformerEncoder, DetrTransformerEncoderLayer)
from .utils import inverse_sigmoid

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except Exception:
    checkpoint_wrapper = None


class DeformableDetrTransformerEncoder(DetrTransformerEncoder):
    """Transformer encoder of Deformable DETR."""

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList([
            DeformableDetrTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])

        if self.num_cp > 0:
            if checkpoint_wrapper is None:
                raise NotImplementedError(
                    'If you want to reduce GPU memory usage, \
                    please install fairscale by executing the \
                    following command: pip install fairscale.')
            for i in range(self.num_cp):
                self.layers[i] = checkpoint_wrapper(self.layers[i])

        self.embed_dims = self.layers[0].embed_dims

    def forward(self, query: Tensor, query_pos: Tensor,
                key_padding_mask: Tensor, spatial_shapes: Tensor,
                level_start_index: Tensor, valid_ratios: Tensor,
                **kwargs) -> Tensor:
        """Forward function of Transformer encoder.

        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            query_pos (Tensor): The positional encoding for query, has shape
                (bs, num_queries, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (bs, num_queries).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).

        Returns:
            Tensor: Output queries of Transformer encoder, which is also
            called 'encoder output embeddings' or 'memory', has shape
            (bs, num_queries, dim)
        """
        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=query.device)
        for layer in self.layers:
            query = layer(
                query=query,
                query_pos=query_pos,
                key_padding_mask=key_padding_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points,
                **kwargs)
        return query

    @staticmethod
    def get_encoder_reference_points(
            spatial_shapes: Tensor, valid_ratios: Tensor,
            device: Union[torch.device, str]) -> Tensor:
        """Get the reference points used in encoder.

        Args:
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            device (obj:`device` or str): The device acquired by the
                `reference_points`.

        Returns:
            Tensor: Reference points used in decoder, has shape (bs, length,
            num_levels, 2).
        """

        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        # [bs, sum(hw), num_level, 2]
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points


class DeformableDetrTransformerDecoder(DetrTransformerDecoder):
    """Transformer Decoder of Deformable DETR."""

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            DeformableDetrTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')

    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                value: Tensor,
                key_padding_mask: Tensor,
                reference_points: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                reg_branches: Optional[nn.Module] = None,
                **kwargs) -> Tuple[Tensor]:
        """Forward function of Transformer decoder.

        Args:
            query (Tensor): The input queries, has shape (bs, num_queries,
                dim).
            query_pos (Tensor): The input positional query, has shape
                (bs, num_queries, dim). It will be added to `query` before
                forward function.
            value (Tensor): The input values, has shape (bs, num_value, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `cross_attn`
                input. ByteTensor, has shape (bs, num_value).
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h) when `as_two_stage` is `True`, otherwise has
                shape (bs, num_queries, 2) with the last dimension arranged
                as (cx, cy).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            reg_branches: (obj:`nn.ModuleList`, optional): Used for refining
                the regression results. Only would be passed when
                `with_box_refine` is `True`, otherwise would be `None`.

        Returns:
            tuple[Tensor]: Outputs of Deformable Transformer Decoder.

            - output (Tensor): Output embeddings of the last decoder, has
              shape (num_queries, bs, embed_dims) when `return_intermediate`
              is `False`. Otherwise, Intermediate output embeddings of all
              decoder layers, has shape (num_decoder_layers, num_queries, bs,
              embed_dims).
            - reference_points (Tensor): The reference of the last decoder
              layer, has shape (bs, num_queries, 4)  when `return_intermediate`
              is `False`. Otherwise, Intermediate references of all decoder
              layers, has shape (num_decoder_layers, bs, num_queries, 4). The
              coordinates are arranged as (cx, cy, w, h)
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        for layer_id, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * \
                    torch.cat([valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * \
                    valid_ratios[:, None]
            output = layer(
                output,
                query_pos=query_pos,
                value=value,
                key_padding_mask=key_padding_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input,
                **kwargs)

            if reg_branches is not None:
                tmp_reg_preds = reg_branches[layer_id](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp_reg_preds + inverse_sigmoid(
                        reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp_reg_preds
                    new_reference_points[..., :2] = tmp_reg_preds[
                        ..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points


class DeformableDetrTransformerEncoderLayer(DetrTransformerEncoderLayer):
    """Encoder layer of Deformable DETR."""

    def _init_layers(self) -> None:
        """Initialize self_attn, ffn, and norms."""
        self.self_attn = MultiScaleDeformableAttention(**self.self_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(2)
        ]
        self.norms = ModuleList(norms_list)

class DeformableDetrGuideTransformerEncoderLayer(DetrTransformerEncoderLayer):
    """Encoder layer of Deformable DETR."""
    def __init__(self, before_FFN = True, **kwargs):
        super(DeformableDetrGuideTransformerEncoderLayer, self).__init__(**kwargs)
        self.before_FFN = before_FFN

    def _init_layers(self) -> None:
        """Initialize self_attn, ffn, and norms."""
        self.self_attn = MultiScaleDeformableAttention(**self.self_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(2)
        ]
        self.norms = ModuleList(norms_list)

    def forward(self, query: Tensor, query_pos: Tensor,
                key_padding_mask: Tensor, **kwargs) -> Tensor:
        """Forward function of an encoder layer.

        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `query`.
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor. has shape (bs, num_queries).
        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """
        level_start_index = kwargs['level_start_index']
        reference_points = kwargs['reference_points']
        spatial_shapes = kwargs['spatial_shapes']
        reference_points_high = reference_points[:, level_start_index[2]:]
        spatial_shapes_high = spatial_shapes[2:]
        query_high = query[:, level_start_index[2]:]
        query_pos_high = query_pos[:, level_start_index[2]:]
        level_start_index_high = torch.cat((
            spatial_shapes_high.new_zeros((1, )),  # (num_level)
            spatial_shapes_high.prod(1).cumsum(0)[:-1]))

        P3 = query[:, :level_start_index[1]]
        P4 = query[:, level_start_index[1]:level_start_index[2]]
        if key_padding_mask is not None:
            key_padding_mask_high = key_padding_mask[:, level_start_index[2]:]
        else:
            key_padding_mask_high = None

        bs, _, dim = query.shape
        H3, W3 = spatial_shapes[0]
        H4, W4 = spatial_shapes[1]
        H5, W5 = spatial_shapes[2]
        if self.before_FFN:
            P5 = query_high[:, :level_start_index_high[1]]
            P4 = P4 + F.interpolate(
                P5.reshape(bs, H5, W5, dim).permute(0, -1, 1, 2),
                size=(H4, W4), mode='nearest').reshape(bs, dim,
                                                       H4 * W4).permute(0, 2, 1).contiguous()
            P3 = P3 + F.interpolate(
                P4.reshape(bs, H4, W4, dim).permute(0, -1, 1, 2),
                size=(H3, W3), mode='nearest').reshape(bs, dim,
                                                       H3 * W3).permute(0, 2, 1).contiguous()

        if not self.before_FFN:
            P5 = query_high[:, :level_start_index_high[1]]
            P4 = P4 + F.interpolate(
                P5.reshape(bs, H5, W5, dim).permute(0, -1, 1, 2),
                size=(H4, W4), mode='nearest').reshape(bs, dim,
                                                       H4 * W4).permute(0, 2, 1).contiguous()

        if not self.before_FFN:
            P3 = P3 + F.interpolate(
                P4.reshape(bs, H4, W4, dim).permute(0, -1, 1, 2),
                size=(H3, W3), mode='nearest').reshape(bs, dim,
                                                       H3 * W3).permute(0, 2, 1).contiguous()


        memory = torch.cat([P3, P4, query_high], dim=1)
        query = self.self_attn(
            query=query,
            key=None,
            value=memory,
            query_pos=query_pos_high,
            key_pos=query_pos_high,
            key_padding_mask=key_padding_mask,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points_high,
            level_start_index=level_start_index)

        query_high = self.norms[0](query_high)
        query_high = self.ffn(query_high)
        query_high = self.norms[1](query_high)

        P4 = self.norms[2](P4)
        P4 = self.ffn1(P4)
        P4 = self.norms[3](P4)

        P3 = self.norms[4](P3)
        P3 = self.ffn2(P3)
        P3 = self.norms[5](P3)

        query = torch.cat([P3, P4, query_high], dim=1)
        return query


class DeformableDetrDecoupleTransformerEncoderLayer(DetrTransformerEncoderLayer):
    """Encoder layer of Deformable DETR."""

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize self_attn, ffn, and norms."""
        num_level = self.self_attn_cfg['num_levels']
        assert num_level == 4, print('num_levels is not 4')
        num_level_list = [2, 3, 3, 2]
        self.level_range = [[0, 2], [0, 3], [1, 4], [2, 4]]
        self.self_attns = []
        self.embed_dims = self.self_attn_cfg['embed_dims']
        for num in num_level_list:
            self_attn_layer = []
            self.self_attn_cfg_ = self.self_attn_cfg.copy()
            self.self_attn_cfg_['num_levels'] = num
            self_attn_layer.append(MultiScaleDeformableAttention(**self.self_attn_cfg_))
            self_attn_layer.append(build_norm_layer(self.norm_cfg, self.embed_dims)[1])
            self_attn_layer.append(FFN(**self.ffn_cfg))
            self_attn_layer.append(build_norm_layer(self.norm_cfg, self.embed_dims)[1])
            self.self_attns.append(ModuleList(self_attn_layer))
        self.self_attns = ModuleList(self.self_attns)
        #
        # self.ffn = FFN(**self.ffn_cfg)
        # norms_list = [
        #     build_norm_layer(self.norm_cfg, self.embed_dims)[1]
        #     for _ in range(2)
        # ]
        # self.norms = ModuleList(norms_list)

    def forward(self, query: Tensor, query_pos: Tensor,
                key_padding_mask: Tensor,
                spatial_shapes,
                reference_points,
                level_start_index,
                ) -> Tensor:
        """Forward function of an encoder layer.

        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `query`.
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor. has shape (bs, num_queries).
        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """
        query_decouple_list = []
        query_pos_decouple_list = []
        reference_points_decouple_list = []
        key_padding_mask_decouple_list = []
        for i, level_start in enumerate(level_start_index):
            level_range = self.level_range[i]
            if i == level_start_index.shape[0] - 1:
                query_decouple_list.append(query[:, level_start:, :])
                query_pos_decouple_list.append(query_pos[:, level_start:, :])
                reference_points_decouple_list.append(reference_points[:, level_start:,
                                                      level_range[0]:level_range[1], :])
                if key_padding_mask is not None:
                    key_padding_mask_decouple_list.append(key_padding_mask[:, level_start:])


            else:
                level_end = level_start_index[i + 1]
                query_decouple_list.append(query[:, level_start:level_end, :])
                query_pos_decouple_list.append(query_pos[:, level_start:level_end, :])
                reference_points_decouple_list.append(reference_points[:, level_start:level_end,
                                                      level_range[0]:level_range[1], :])
                if key_padding_mask is not None:
                    key_padding_mask_decouple_list.append(key_padding_mask[:, level_start:level_end])


        query_decouple_update = []
        for ind, layers in enumerate(self.self_attns):
            # get value, spatial shapes and level start index
            level_range = self.level_range[ind]
            value_decouple = torch.cat(query_decouple_list[level_range[0]:level_range[1]], dim=1)
            spatial_shapes_decouple = spatial_shapes[level_range[0]:level_range[1],...]
            level_start_index_decouple = torch.cat((
                spatial_shapes_decouple.new_zeros((1,)),  # (num_level)
                spatial_shapes_decouple.prod(1).cumsum(0)[:-1]))

            # get query
            query_decouple = query_decouple_list[ind]
            query_pos_decouple = query_pos_decouple_list[ind]
            if key_padding_mask is not None:
                key_padding_mask_decouple = torch.cat(key_padding_mask_decouple_list[level_range[0]:level_range[1]], dim=1)
            else:
                key_padding_mask_decouple = None
            reference_points_decouple = reference_points_decouple_list[ind]
            for i, layer in enumerate(layers):
                if i == 0:
                    query_decouple_ = layer(
                        query=query_decouple,
                        key=None,
                        value=value_decouple,
                        query_pos=query_pos_decouple,
                        key_pos=None,
                        key_padding_mask=key_padding_mask_decouple,
                        spatial_shapes=spatial_shapes_decouple,
                        reference_points=reference_points_decouple,
                        level_start_index=level_start_index_decouple
                    )
                else:
                    query_decouple_ = layer(query_decouple_)
            query_decouple_update.append(query_decouple_)
        query = torch.cat(query_decouple_update, dim=1)
        # query = self.norms[0](query)
        # query = self.ffn(query)
        # query = self.norms[1](query)
        return query

class DeformableDetrTransformerDecoderLayer(DetrTransformerDecoderLayer):
    """Decoder layer of Deformable DETR."""

    def _init_layers(self) -> None:
        """Initialize self_attn, cross-attn, ffn, and norms."""
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.cross_attn = MultiScaleDeformableAttention(**self.cross_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(3)
        ]
        self.norms = ModuleList(norms_list)
