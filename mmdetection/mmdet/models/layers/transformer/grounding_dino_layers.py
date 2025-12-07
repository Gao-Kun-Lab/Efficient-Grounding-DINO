# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer, Linear
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.ops import MultiScaleDeformableAttention
from mmengine.model import ModuleList, BaseModule
from torch import Tensor
from mmcv.cnn.bricks import DropPath
from mmdet.models.utils.vlfuse_helper import SingleScaleBiAttentionBlock
from mmdet.utils import ConfigType, OptConfigType
from .deformable_detr_layers import (DeformableDetrTransformerDecoderLayer,
                                     DeformableDetrTransformerEncoder,
                                     DeformableDetrTransformerEncoderLayer,
                                     DeformableDetrDecoupleTransformerEncoderLayer,
                                     )
from .detr_layers import DetrTransformerEncoderLayer, DetrTransformerWeightEncoderLayer
from .dino_layers import DinoTransformerDecoder
from .utils import MLP, get_text_sine_pos_embed
from .sparse_layers import DeformableDetrSparseTransformerEncoderLayer
from typing import Optional
from .MHAttention import MultiheadAttention_weight
from .hybrid_encoder import CCFF
from .detr_layers import DetrTransformerEncoder
# from torchprofile import profile_macs
from fvcore.nn import FlopCountAnalysis
try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except Exception:
    checkpoint_wrapper = None


class GroundingDinoTransformerDecoderLayer(
        DeformableDetrTransformerDecoderLayer):

    def __init__(self,
                 cross_attn_text_cfg: OptConfigType = dict(
                     embed_dims=256,
                     num_heads=8,
                     dropout=0.0,
                     batch_first=True),
                 **kwargs) -> None:
        """Decoder layer of Deformable DETR."""
        self.cross_attn_text_cfg = cross_attn_text_cfg
        if 'batch_first' not in self.cross_attn_text_cfg:
            self.cross_attn_text_cfg['batch_first'] = True
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize self_attn, cross-attn, ffn, and norms."""
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.cross_attn_text = MultiheadAttention(**self.cross_attn_text_cfg)
        self.cross_attn = MultiScaleDeformableAttention(**self.cross_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(4)
        ]
        self.norms = ModuleList(norms_list)

    def forward(self,
                query: Tensor,
                key: Tensor = None,
                value: Tensor = None,
                query_pos: Tensor = None,
                key_pos: Tensor = None,
                self_attn_mask: Tensor = None,
                cross_attn_mask: Tensor = None,
                key_padding_mask: Tensor = None,
                memory_text: Tensor = None,
                text_attention_mask: Tensor = None,
                **kwargs) -> Tensor:
        """Implements decoder layer in Grounding DINO transformer.

        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            key (Tensor, optional): The input key, has shape (bs, num_keys,
                dim). If `None`, the `query` will be used. Defaults to `None`.
            value (Tensor, optional): The input value, has the same shape as
                `key`, as in `nn.MultiheadAttention.forward`. If `None`, the
                `key` will be used. Defaults to `None`.
            query_pos (Tensor, optional): The positional encoding for `query`,
                has the same shape as `query`. If not `None`, it will be added
                to `query` before forward function. Defaults to `None`.
            key_pos (Tensor, optional): The positional encoding for `key`, has
                the same shape as `key`. If not `None`, it will be added to
                `key` before forward function. If None, and `query_pos` has the
                same shape as `key`, then `query_pos` will be used for
                `key_pos`. Defaults to None.
            self_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            cross_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor, optional): The `key_padding_mask` of
                `self_attn` input. ByteTensor, has shape (bs, num_value).
                Defaults to None.
            memory_text (Tensor): Memory text. It has shape (bs, len_text,
                text_embed_dims).
            text_attention_mask (Tensor): Text token mask. It has shape (bs,
                len_text).

        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """
        # self attention
        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
            attn_mask=self_attn_mask,
            **kwargs)
        query = self.norms[0](query)
        # cross attention between query and text
        query = self.cross_attn_text(
            query=query,
            query_pos=query_pos,
            key=memory_text,
            value=memory_text,
            key_padding_mask=text_attention_mask)
        query = self.norms[1](query)
        # cross attention between query and image
        query = self.cross_attn(
            query=query,
            key=key,
            value=value,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=cross_attn_mask,
            key_padding_mask=key_padding_mask,
            **kwargs)
        query = self.norms[2](query)
        query = self.ffn(query)
        query = self.norms[3](query)

        return query


class GroundingDinoTransformerEncoder(DeformableDetrTransformerEncoder):

    def __init__(self, text_layer_cfg: ConfigType,
                 fusion_layer_cfg: ConfigType, **kwargs) -> None:
        self.text_layer_cfg = text_layer_cfg
        self.fusion_layer_cfg = fusion_layer_cfg
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList([
            DeformableDetrTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.text_layers = ModuleList([
            DetrTransformerEncoderLayer(**self.text_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.fusion_layers = ModuleList([
            SingleScaleBiAttentionBlock(**self.fusion_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.num_cp > 0:
            if checkpoint_wrapper is None:
                raise NotImplementedError(
                    'If you want to reduce GPU memory usage, \
                    please install fairscale by executing the \
                    following command: pip install fairscale.')
            for i in range(self.num_cp):
                self.layers[i] = checkpoint_wrapper(self.layers[i])
                self.fusion_layers[i] = checkpoint_wrapper(
                    self.fusion_layers[i])

    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                key_padding_mask: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                memory_text: Tensor = None,
                text_attention_mask: Tensor = None,
                pos_text: Tensor = None,
                text_self_attention_masks: Tensor = None,
                position_ids: Tensor = None):
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
            memory_text (Tensor, optional): Memory text. It has shape (bs,
                len_text, text_embed_dims).
            text_attention_mask (Tensor, optional): Text token mask. It has
                shape (bs,len_text).
            pos_text (Tensor, optional): The positional encoding for text.
                Defaults to None.
            text_self_attention_masks (Tensor, optional): Text self attention
                mask. Defaults to None.
            position_ids (Tensor, optional): Text position ids.
                Defaults to None.
        """
        output = query
        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=query.device)
        if self.text_layers:
            # generate pos_text
            bs, n_text, _ = memory_text.shape
            if pos_text is None and position_ids is None:
                pos_text = (
                    torch.arange(n_text,
                                 device=memory_text.device).float().unsqueeze(
                                     0).unsqueeze(-1).repeat(bs, 1, 1))
                pos_text = get_text_sine_pos_embed(
                    pos_text, num_pos_feats=256, exchange_xy=False)
            if position_ids is not None:
                pos_text = get_text_sine_pos_embed(
                    position_ids[..., None],
                    num_pos_feats=256,
                    exchange_xy=False)

        # main process
        for layer_id, layer in enumerate(self.layers):
            if self.fusion_layers:
                output, memory_text = self.fusion_layers[layer_id](
                    visual_feature=output,
                    lang_feature=memory_text,
                    attention_mask_v=key_padding_mask,
                    attention_mask_l=text_attention_mask,
                )
            if self.text_layers:
                text_num_heads = self.text_layers[
                    layer_id].self_attn_cfg.num_heads
                memory_text = self.text_layers[layer_id](
                    query=memory_text,
                    query_pos=(pos_text if pos_text is not None else None),
                    attn_mask=~text_self_attention_masks.repeat(
                        text_num_heads, 1, 1),  # note we use ~ for mask here
                    key_padding_mask=None,
                )
            output = layer(
                query=output,
                query_pos=query_pos,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=key_padding_mask)
        return output, memory_text

class GroundingDinoDecoupleInvertBlockTransformerEncoder(DeformableDetrTransformerEncoder):

    def __init__(self, text_layer_cfg: ConfigType,
                 fusion_layer_cfg: ConfigType,
                 num_block,
                 **kwargs) -> None:
        self.text_layer_cfg = text_layer_cfg
        self.fusion_layer_cfg = fusion_layer_cfg
        self.num_block = num_block
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.block_layer_num = int(self.num_layers / self.num_block)
        self.layers = []
        for i in range(self.num_layers):
            if (i + 1) % self.block_layer_num == 0:
                self.layers.append(DeformableDetrTransformerEncoderLayer(**self.layer_cfg))
            else:
                self.layers.append(DeformableDetrSparseTransformerEncoderLayer(**self.layer_cfg))
        self.layers = ModuleList(self.layers)
        # self.layers = ModuleList([
        #     DeformableDetrTransformerEncoderLayer(**self.layer_cfg)
        #     for _ in range(self.num_layers)
        # ])
        self.text_layers = ModuleList([
            DetrTransformerEncoderLayer(**self.text_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.fusion_layers = ModuleList([
            TransformerOnlyDecoupleFusionLayer(**self.fusion_layer_cfg)
            for _ in range(self.num_block)
        ])
        self.embed_dims = self.layers[0].embed_dims
        # if self.num_cp > 0:
        #     if checkpoint_wrapper is None:
        #         raise NotImplementedError(
        #             'If you want to reduce GPU memory usage, \
        #             please install fairscale by executing the \
        #             following command: pip install fairscale.')
        #     for i in range(self.num_cp):
        #         self.layers[i] = checkpoint_wrapper(self.layers[i])
        #     for i in range(self.num_block):
        #         self.fusion_layers[i] = checkpoint_wrapper(
        #             self.fusion_layers[i])

    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                key_padding_mask: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                memory_text: Tensor = None,
                text_attention_mask: Tensor = None,
                pos_text: Tensor = None,
                text_self_attention_masks: Tensor = None,
                position_ids: Tensor = None):
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
            memory_text (Tensor, optional): Memory text. It has shape (bs,
                len_text, text_embed_dims).
            text_attention_mask (Tensor, optional): Text token mask. It has
                shape (bs,len_text).
            pos_text (Tensor, optional): The positional encoding for text.
                Defaults to None.
            text_self_attention_masks (Tensor, optional): Text self attention
                mask. Defaults to None.
            position_ids (Tensor, optional): Text position ids.
                Defaults to None.
        """
        output = query
        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=query.device)
        if self.text_layers:
            # generate pos_text
            bs, n_text, _ = memory_text.shape
            if pos_text is None and position_ids is None:
                pos_text = (
                    torch.arange(n_text,
                                 device=memory_text.device).float().unsqueeze(
                                     0).unsqueeze(-1).repeat(bs, 1, 1))
                pos_text = get_text_sine_pos_embed(
                    pos_text, num_pos_feats=256, exchange_xy=False)
            if position_ids is not None:
                pos_text = get_text_sine_pos_embed(
                    position_ids[..., None],
                    num_pos_feats=256,
                    exchange_xy=False)

        # main process
        for layer_id, layer in enumerate(self.layers):
            if layer_id % self.block_layer_num == 0:
                if self.fusion_layers:
                    output, memory_text = self.fusion_layers[layer_id // self.block_layer_num](
                        output=output,
                        memory_text=memory_text,
                        query_pos=query_pos,
                        pos_text=(pos_text if pos_text is not None else None),
                        attn_mask=None,
                        text_attention_mask=text_attention_mask,
                        spatial_shapes=spatial_shapes,
                        key_padding_mask_image=key_padding_mask
                    )
            if self.text_layers:
                text_num_heads = self.text_layers[
                    layer_id].self_attn_cfg.num_heads
                memory_text = self.text_layers[layer_id](
                    query=memory_text,
                    query_pos=(pos_text if pos_text is not None else None),
                    attn_mask=~text_self_attention_masks.repeat(
                        text_num_heads, 1, 1),  # note we use ~ for mask here
                    key_padding_mask=None,
                )
            if (layer_id + 1) % self.block_layer_num == 0:
                output = layer(
                    query=output,
                    query_pos=query_pos,
                    reference_points=reference_points,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    key_padding_mask=key_padding_mask)
            elif (layer_id + 1) % self.block_layer_num == 1:
                output_lite = output[:, level_start_index[2]:, :]
                output_res = output[:, :level_start_index[2], :]
                reference_points_lite = reference_points[:, level_start_index[2]:, ...]
                query_pos_lite = query_pos[:, level_start_index[2]:, :]
                spatial_shapes_lite = spatial_shapes[2:]
                output_lite = layer(
                    query=output_lite,
                    query_pos=query_pos_lite,
                    memory=output,
                    reference_points=reference_points_lite,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    key_padding_mask=key_padding_mask)
                H3, W3 = spatial_shapes_lite[0]
                H1, W1 = spatial_shapes[0]
                H2, W2 = spatial_shapes[1]
                l3_ind = (H3 * W3).item()
                bs, _, dim = output_res.shape
                P3 = output_res[:, :H1 * W1, :]
                P4 = output_res[:, H1 * W1:, :]
                P4 = P4 + F.interpolate(output_lite[:, :l3_ind, :].reshape(bs, H3, W3, dim).permute(0, -1, 1, 2),
                                        size=(H2, W2), mode='nearest').reshape(bs, dim,
                                                                               P4.shape[1]).permute(0, 2,
                                                                                                    1).contiguous()
                P3 = P3 + F.interpolate(P4.reshape(bs, H2, W2, dim).permute(0, -1, 1, 2),
                                        size=(H1, W1), mode='nearest').reshape(bs, dim,
                                                                               P3.shape[1]).permute(0, 2,
                                                                                                    1).contiguous()
                output_res = torch.cat((P3, P4), dim=1)
                output = torch.cat((output_res, output_lite), dim=1)
            elif (layer_id + 1) % self.block_layer_num == 2:
                output_lite = output[:, level_start_index[1]:, :]
                output_res = output[:, :level_start_index[1], :]
                reference_points_lite = reference_points[:, level_start_index[1]:, ...]
                query_pos_lite = query_pos[:, level_start_index[1]:, :]
                spatial_shapes_lite = spatial_shapes[1:]
                output_lite = layer(
                    query=output_lite,
                    query_pos=query_pos_lite,
                    memory=output,
                    reference_points=reference_points_lite,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    key_padding_mask=key_padding_mask)
                H2, W2 = spatial_shapes_lite[0]
                H1, W1 = spatial_shapes[0]
                l2_ind = (H2 * W2).item()
                bs, _, dim = output_res.shape
                output_res = output_res + F.interpolate(output_lite[:,:l2_ind,:].reshape(bs, H2, W2, dim).permute(0,-1, 1, 2),
                                       size=(H1, W1), mode='nearest').reshape(bs,dim,
                                                                               output_res.shape[1]).permute(0,2,1).contiguous()
                output = torch.cat((output_res, output_lite), dim=1)
        return output, memory_text

class GroundingDinoAblationInvertBlockTransformerEncoder(DeformableDetrTransformerEncoder):

    def __init__(self, text_layer_cfg: ConfigType,
                 fusion_layer_cfg: ConfigType,
                 num_block,
                 fusion_mode='Decouple',
                 block_mode='Invert',
                 guide=True,
                 **kwargs) -> None:
        self.text_layer_cfg = text_layer_cfg
        self.fusion_layer_cfg = fusion_layer_cfg
        self.num_block = num_block
        self.fusion_mode = fusion_mode
        self.block_mode = block_mode
        self.guide = guide
        assert self.fusion_mode in ['Decouple', 'Enhance', 'LQVG'], \
            print("fusion mode is not in [Decouple, Enhance, LQVG]")
        assert self.block_mode in ['Invert', 'Full_Def', 'Lite']
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.block_layer_num = int(self.num_layers / self.num_block)
        if self.block_mode is 'Invert':
            self.layers = []
            for i in range(self.num_layers):
                if (i + 1) % self.block_layer_num == 0:
                    self.layers.append(DeformableDetrTransformerEncoderLayer(**self.layer_cfg))
                else:
                    self.layers.append(DeformableDetrSparseTransformerEncoderLayer(**self.layer_cfg))
            self.layers = ModuleList(self.layers)
        elif self.block_mode is 'Full_Def':
            self.layers = ModuleList([
                DeformableDetrTransformerEncoderLayer(**self.layer_cfg)
                for _ in range(self.num_layers)
            ])
        elif self.block_mode is 'Lite':
            self.layers = ModuleList([
                DeformableDetrSparseTransformerEncoderLayer(**self.layer_cfg)
                for _ in range(self.num_layers)
            ])
        self.text_layers = ModuleList([
            DetrTransformerEncoderLayer(**self.text_layer_cfg)
            for _ in range(self.num_layers)
        ])
        if self.fusion_mode is 'Decouple':
            self.fusion_layers = ModuleList([
                TransformerOnlyDecoupleFusionLayer(**self.fusion_layer_cfg)
                for _ in range(self.num_block)
            ])
        elif self.fusion_mode is 'Enhance':
            self.fusion_layers = ModuleList([
                SingleScaleBiAttentionBlock(**self.fusion_layer_cfg)
                for _ in range(self.num_block)
            ])
        elif self.fusion_mode is 'LQVG':
            self.fusion_layers = ModuleList([
                TransformerDecoupleLQVGFusionLayer(**self.fusion_layer_cfg)
                for _ in range(self.num_block)
            ])
        self.embed_dims = self.layers[0].embed_dims
        # if self.num_cp > 0:
        #     if checkpoint_wrapper is None:
        #         raise NotImplementedError(
        #             'If you want to reduce GPU memory usage, \
        #             please install fairscale by executing the \
        #             following command: pip install fairscale.')
        #     for i in range(self.num_cp):
        #         self.layers[i] = checkpoint_wrapper(self.layers[i])
        #     for i in range(self.num_block):
        #         self.fusion_layers[i] = checkpoint_wrapper(
        #             self.fusion_layers[i])

    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                key_padding_mask: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                memory_text: Tensor = None,
                text_attention_mask: Tensor = None,
                pos_text: Tensor = None,
                text_self_attention_masks: Tensor = None,
                position_ids: Tensor = None):
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
            memory_text (Tensor, optional): Memory text. It has shape (bs,
                len_text, text_embed_dims).
            text_attention_mask (Tensor, optional): Text token mask. It has
                shape (bs,len_text).
            pos_text (Tensor, optional): The positional encoding for text.
                Defaults to None.
            text_self_attention_masks (Tensor, optional): Text self attention
                mask. Defaults to None.
            position_ids (Tensor, optional): Text position ids.
                Defaults to None.
        """
        output = query
        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=query.device)
        if self.text_layers:
            # generate pos_text
            bs, n_text, _ = memory_text.shape
            if pos_text is None and position_ids is None:
                pos_text = (
                    torch.arange(n_text,
                                 device=memory_text.device).float().unsqueeze(
                                     0).unsqueeze(-1).repeat(bs, 1, 1))
                pos_text = get_text_sine_pos_embed(
                    pos_text, num_pos_feats=256, exchange_xy=False)
            if position_ids is not None:
                pos_text = get_text_sine_pos_embed(
                    position_ids[..., None],
                    num_pos_feats=256,
                    exchange_xy=False)

        # main process
        for layer_id, layer in enumerate(self.layers):
            if layer_id % self.block_layer_num == 0:
                if self.fusion_layers:
                    if self.fusion_mode in ['Decouple', 'LQVG']:
                        output, memory_text = self.fusion_layers[layer_id // self.block_layer_num](
                            output=output,
                            memory_text=memory_text,
                            query_pos=query_pos,
                            pos_text=(pos_text if pos_text is not None else None),
                            attn_mask=None,
                            text_attention_mask=text_attention_mask,
                            spatial_shapes=spatial_shapes,
                            key_padding_mask_image=key_padding_mask
                        )
                    elif self.fusion_mode is 'Enhance':
                        output, memory_text = self.fusion_layers[layer_id // self.block_layer_num](
                            visual_feature=output,
                            lang_feature=memory_text,
                            attention_mask_v=key_padding_mask,
                            attention_mask_l=text_attention_mask,
                        )
            if self.text_layers:
                text_num_heads = self.text_layers[
                    layer_id].self_attn_cfg.num_heads
                memory_text = self.text_layers[layer_id](
                    query=memory_text,
                    query_pos=(pos_text if pos_text is not None else None),
                    attn_mask=~text_self_attention_masks.repeat(
                        text_num_heads, 1, 1),  # note we use ~ for mask here
                    key_padding_mask=None,
                )
            if self.block_mode is 'Invert':
                if (layer_id + 1) % self.block_layer_num == 0:
                    output = layer(
                        query=output,
                        query_pos=query_pos,
                        reference_points=reference_points,
                        spatial_shapes=spatial_shapes,
                        level_start_index=level_start_index,
                        key_padding_mask=key_padding_mask)
                elif (layer_id + 1) % self.block_layer_num == 1:
                    output_lite = output[:, level_start_index[2]:, :]
                    output_res = output[:, :level_start_index[2], :]
                    reference_points_lite = reference_points[:, level_start_index[2]:, ...]
                    query_pos_lite = query_pos[:, level_start_index[2]:, :]
                    spatial_shapes_lite = spatial_shapes[2:]
                    output_lite = layer(
                        query=output_lite,
                        query_pos=query_pos_lite,
                        memory=output,
                        reference_points=reference_points_lite,
                        spatial_shapes=spatial_shapes,
                        level_start_index=level_start_index,
                        key_padding_mask=key_padding_mask)
                    if self.guide:
                        H3, W3 = spatial_shapes_lite[0]
                        H1, W1 = spatial_shapes[0]
                        H2, W2 = spatial_shapes[1]
                        l3_ind = (H3 * W3).item()
                        bs, _, dim = output_res.shape
                        P3 = output_res[:, :H1 * W1, :]
                        P4 = output_res[:, H1 * W1:, :]
                        P4 = P4 + F.interpolate(output_lite[:, :l3_ind, :].reshape(bs, H3, W3, dim).permute(0, -1, 1, 2),
                                                size=(H2, W2), mode='nearest').reshape(bs, dim,
                                                                                       P4.shape[1]).permute(0, 2,
                                                                                                            1).contiguous()
                        P3 = P3 + F.interpolate(P4.reshape(bs, H2, W2, dim).permute(0, -1, 1, 2),
                                                size=(H1, W1), mode='nearest').reshape(bs, dim,
                                                                                       P3.shape[1]).permute(0, 2,
                                                                                                            1).contiguous()
                        output_res = torch.cat((P3, P4), dim=1)
                    output = torch.cat((output_res, output_lite), dim=1)
                elif (layer_id + 1) % self.block_layer_num == 2:
                    output_lite = output[:, level_start_index[1]:, :]
                    output_res = output[:, :level_start_index[1], :]
                    reference_points_lite = reference_points[:, level_start_index[1]:, ...]
                    query_pos_lite = query_pos[:, level_start_index[1]:, :]
                    spatial_shapes_lite = spatial_shapes[1:]
                    output_lite = layer(
                        query=output_lite,
                        query_pos=query_pos_lite,
                        memory=output,
                        reference_points=reference_points_lite,
                        spatial_shapes=spatial_shapes,
                        level_start_index=level_start_index,
                        key_padding_mask=key_padding_mask)
                    if self.guide:
                        H2, W2 = spatial_shapes_lite[0]
                        H1, W1 = spatial_shapes[0]
                        l2_ind = (H2 * W2).item()
                        bs, _, dim = output_res.shape
                        output_res = output_res + F.interpolate(output_lite[:,:l2_ind,:].reshape(bs, H2, W2, dim).permute(0,-1, 1, 2),
                                               size=(H1, W1), mode='nearest').reshape(bs,dim,
                                                                                       output_res.shape[1]).permute(0,2,1).contiguous()
                    output = torch.cat((output_res, output_lite), dim=1)
            elif self.block_mode is 'Full_Def':
                output = layer(
                    query=output,
                    query_pos=query_pos,
                    reference_points=reference_points,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    key_padding_mask=key_padding_mask)
            elif self.block_mode is 'Lite':
                if (layer_id + 1) % self.block_layer_num == 0:
                    output_lite = output[:, :level_start_index[1], :]
                    output_res = output[:, level_start_index[1]:, :]
                    reference_points_lite = reference_points[:, :level_start_index[1], ...]
                    query_pos_lite = query_pos[:, :level_start_index[1], :]
                else:
                    output_lite = output[:, level_start_index[1]:, :]
                    output_res = output[:, :level_start_index[1], :]
                    reference_points_lite = reference_points[:, level_start_index[1]:, ...]
                    query_pos_lite = query_pos[:, level_start_index[1]:, :]
                output_lite = layer(
                    query=output_lite,
                    query_pos=query_pos_lite,
                    memory=output,
                    reference_points=reference_points_lite,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    key_padding_mask=key_padding_mask)
                if (layer_id + 1) % self.block_layer_num == 0:
                    output = torch.cat((output_lite, output_res), dim=1)
                else:
                    output = torch.cat((output_res, output_lite), dim=1)
        return output, memory_text

class GroundingDinoAblationTransformerEncoder(DeformableDetrTransformerEncoder):

    def __init__(self, text_layer_cfg: ConfigType,
                 fusion_layer_cfg: ConfigType,
                 num_block,
                 fusion_layer_index,
                 fusion_mode='Decouple',
                 block_mode='Invert',
                 **kwargs) -> None:
        self.text_layer_cfg = text_layer_cfg
        self.fusion_layer_cfg = fusion_layer_cfg
        self.num_block = num_block
        self.fusion_layer_index = fusion_layer_index
        self.fusion_layer_num = len(self.fusion_layer_index)
        self.fusion_mode = fusion_mode
        self.block_mode = block_mode
        assert self.fusion_mode in ['Decouple', 'Enhance', 'LQVG'], \
            print("fusion mode is not in [Decouple, Enhance, LQVG]")
        assert self.block_mode in ['Invert', 'Full_Def', 'Lite']
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.block_layer_num = int(self.num_layers / self.num_block)
        if self.block_mode is 'Invert':
            self.layers = []
            for i in range(self.num_layers):
                if (i + 1) % self.block_layer_num == 0:
                    self.layers.append(DeformableDetrTransformerEncoderLayer(**self.layer_cfg))
                else:
                    self.layers.append(DeformableDetrSparseTransformerEncoderLayer(**self.layer_cfg))
            self.layers = ModuleList(self.layers)
        elif self.block_mode is 'Full_Def':
            self.layers = ModuleList([
                DeformableDetrTransformerEncoderLayer(**self.layer_cfg)
                for _ in range(self.num_layers)
            ])
        elif self.block_mode is 'Lite':
            self.layers = ModuleList([
                DeformableDetrSparseTransformerEncoderLayer(**self.layer_cfg)
                for _ in range(self.num_layers)
            ])
        self.text_layers = ModuleList([
            DetrTransformerEncoderLayer(**self.text_layer_cfg)
            for _ in range(self.num_layers)
        ])
        if self.fusion_mode is 'Decouple':
            self.fusion_layers = ModuleList([
                TransformerOnlyDecoupleFusionLayer(**self.fusion_layer_cfg)
                for _ in range(self.fusion_layer_num)
            ])
        elif self.fusion_mode is 'Enhance':
            self.fusion_layers = ModuleList([
                SingleScaleBiAttentionBlock(**self.fusion_layer_cfg)
                for _ in range(self.fusion_layer_num)
            ])
        elif self.fusion_mode is 'LQVG':
            self.fusion_layers = ModuleList([
                TransformerDecoupleLQVGFusionLayer(**self.fusion_layer_cfg)
                for _ in range(self.fusion_layer_num)
            ])
        self.embed_dims = self.layers[0].embed_dims
        if self.num_cp > 0:
            if checkpoint_wrapper is None:
                raise NotImplementedError(
                    'If you want to reduce GPU memory usage, \
                    please install fairscale by executing the \
                    following command: pip install fairscale.')
            for i in range(self.num_cp):
                self.layers[i] = checkpoint_wrapper(self.layers[i])
            for i in range(self.num_block):
                self.fusion_layers[i] = checkpoint_wrapper(
                    self.fusion_layers[i])

    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                key_padding_mask: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                memory_text: Tensor = None,
                text_attention_mask: Tensor = None,
                pos_text: Tensor = None,
                text_self_attention_masks: Tensor = None,
                position_ids: Tensor = None):
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
            memory_text (Tensor, optional): Memory text. It has shape (bs,
                len_text, text_embed_dims).
            text_attention_mask (Tensor, optional): Text token mask. It has
                shape (bs,len_text).
            pos_text (Tensor, optional): The positional encoding for text.
                Defaults to None.
            text_self_attention_masks (Tensor, optional): Text self attention
                mask. Defaults to None.
            position_ids (Tensor, optional): Text position ids.
                Defaults to None.
        """
        output = query
        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=query.device)
        if self.text_layers:
            # generate pos_text
            bs, n_text, _ = memory_text.shape
            if pos_text is None and position_ids is None:
                pos_text = (
                    torch.arange(n_text,
                                 device=memory_text.device).float().unsqueeze(
                                     0).unsqueeze(-1).repeat(bs, 1, 1))
                pos_text = get_text_sine_pos_embed(
                    pos_text, num_pos_feats=256, exchange_xy=False)
            if position_ids is not None:
                pos_text = get_text_sine_pos_embed(
                    position_ids[..., None],
                    num_pos_feats=256,
                    exchange_xy=False)

        # main process
        for layer_id, layer in enumerate(self.layers):
            if layer_id in self.fusion_layer_index:
                if self.fusion_layers:
                    if self.fusion_mode in ['Decouple', 'LQVG']:
                        output, memory_text = self.fusion_layers[self.fusion_layer_index.index(layer_id)](
                            output=output,
                            memory_text=memory_text,
                            query_pos=query_pos,
                            pos_text=(pos_text if pos_text is not None else None),
                            attn_mask=None,
                            text_attention_mask=text_attention_mask,
                            spatial_shapes=spatial_shapes,
                            key_padding_mask_image=key_padding_mask
                        )
                    elif self.fusion_mode is 'Enhance':
                        output, memory_text = self.fusion_layers[self.fusion_layer_index.index(layer_id)](
                            visual_feature=output,
                            lang_feature=memory_text,
                            attention_mask_v=key_padding_mask,
                            attention_mask_l=text_attention_mask,
                        )
            if self.text_layers:
                text_num_heads = self.text_layers[
                    layer_id].self_attn_cfg.num_heads
                memory_text = self.text_layers[layer_id](
                    query=memory_text,
                    query_pos=(pos_text if pos_text is not None else None),
                    attn_mask=~text_self_attention_masks.repeat(
                        text_num_heads, 1, 1),  # note we use ~ for mask here
                    key_padding_mask=None,
                )
            if self.block_mode is 'Invert':
                if (layer_id + 1) % self.block_layer_num == 0:
                    output = layer(
                        query=output,
                        query_pos=query_pos,
                        reference_points=reference_points,
                        spatial_shapes=spatial_shapes,
                        level_start_index=level_start_index,
                        key_padding_mask=key_padding_mask)
                elif (layer_id + 1) % self.block_layer_num == 1:
                    output_lite = output[:, level_start_index[2]:, :]
                    output_res = output[:, :level_start_index[2], :]
                    reference_points_lite = reference_points[:, level_start_index[2]:, ...]
                    query_pos_lite = query_pos[:, level_start_index[2]:, :]
                    spatial_shapes_lite = spatial_shapes[2:]
                    output_lite = layer(
                        query=output_lite,
                        query_pos=query_pos_lite,
                        memory=output,
                        reference_points=reference_points_lite,
                        spatial_shapes=spatial_shapes,
                        level_start_index=level_start_index,
                        key_padding_mask=key_padding_mask)
                    H3, W3 = spatial_shapes_lite[0]
                    H1, W1 = spatial_shapes[0]
                    H2, W2 = spatial_shapes[1]
                    l3_ind = (H3 * W3).item()
                    bs, _, dim = output_res.shape
                    P3 = output_res[:, :H1 * W1, :]
                    P4 = output_res[:, H1 * W1:, :]
                    P4 = P4 + F.interpolate(output_lite[:, :l3_ind, :].reshape(bs, H3, W3, dim).permute(0, -1, 1, 2),
                                            size=(H2, W2), mode='nearest').reshape(bs, dim,
                                                                                   P4.shape[1]).permute(0, 2,
                                                                                                        1).contiguous()
                    P3 = P3 + F.interpolate(P4.reshape(bs, H2, W2, dim).permute(0, -1, 1, 2),
                                            size=(H1, W1), mode='nearest').reshape(bs, dim,
                                                                                   P3.shape[1]).permute(0, 2,
                                                                                                        1).contiguous()
                    output_res = torch.cat((P3, P4), dim=1)
                    output = torch.cat((output_res, output_lite), dim=1)
                elif (layer_id + 1) % self.block_layer_num == 2:
                    output_lite = output[:, level_start_index[1]:, :]
                    output_res = output[:, :level_start_index[1], :]
                    reference_points_lite = reference_points[:, level_start_index[1]:, ...]
                    query_pos_lite = query_pos[:, level_start_index[1]:, :]
                    spatial_shapes_lite = spatial_shapes[1:]
                    output_lite = layer(
                        query=output_lite,
                        query_pos=query_pos_lite,
                        memory=output,
                        reference_points=reference_points_lite,
                        spatial_shapes=spatial_shapes,
                        level_start_index=level_start_index,
                        key_padding_mask=key_padding_mask)
                    H2, W2 = spatial_shapes_lite[0]
                    H1, W1 = spatial_shapes[0]
                    l2_ind = (H2 * W2).item()
                    bs, _, dim = output_res.shape
                    output_res = output_res + F.interpolate(output_lite[:,:l2_ind,:].reshape(bs, H2, W2, dim).permute(0,-1, 1, 2),
                                           size=(H1, W1), mode='nearest').reshape(bs,dim,
                                                                                   output_res.shape[1]).permute(0,2,1).contiguous()
                    output = torch.cat((output_res, output_lite), dim=1)
            elif self.block_mode is 'Full_Def':
                output = layer(
                    query=output,
                    query_pos=query_pos,
                    reference_points=reference_points,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    key_padding_mask=key_padding_mask)
            elif self.block_mode is 'Lite':
                if (layer_id + 1) % self.block_layer_num == 0:
                    output_lite = output[:, :level_start_index[1], :]
                    output_res = output[:, level_start_index[1]:, :]
                    reference_points_lite = reference_points[:, :level_start_index[1], ...]
                    query_pos_lite = query_pos[:, :level_start_index[1], :]
                else:
                    output_lite = output[:, level_start_index[1]:, :]
                    output_res = output[:, :level_start_index[1], :]
                    reference_points_lite = reference_points[:, level_start_index[1]:, ...]
                    query_pos_lite = query_pos[:, level_start_index[1]:, :]
                output_lite = layer(
                    query=output_lite,
                    query_pos=query_pos_lite,
                    memory=output,
                    reference_points=reference_points_lite,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    key_padding_mask=key_padding_mask)
                if (layer_id + 1) % self.block_layer_num == 0:
                    output = torch.cat((output_lite, output_res), dim=1)
                else:
                    output = torch.cat((output_res, output_lite), dim=1)
        return output, memory_text

class GroundingDinoStage2TransformerEncoder(DeformableDetrTransformerEncoder):

    def __init__(self, text_layer_cfg: ConfigType,
                 fusion_layer_cfg: ConfigType,
                 low_layer_num, **kwargs) -> None:
        self.text_layer_cfg = text_layer_cfg
        self.fusion_layer_cfg = fusion_layer_cfg
        self.low_layer_num = low_layer_num
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = [DeformableDetrSparseTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.low_layer_num)]
        self.layers.insert(0, DeformableDetrTransformerEncoderLayer(**self.layer_cfg))
        self.layers.append(DeformableDetrTransformerEncoderLayer(**self.layer_cfg))
        self.layers = ModuleList(self.layers)
        self.text_layers = ModuleList([
            DetrTransformerEncoderLayer(**self.text_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.fusion_layers = ModuleList([
            SingleScaleBiAttentionBlock(**self.fusion_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.num_cp > 0:
            if checkpoint_wrapper is None:
                raise NotImplementedError(
                    'If you want to reduce GPU memory usage, \
                    please install fairscale by executing the \
                    following command: pip install fairscale.')
            for i in range(self.num_cp):
                self.layers[i] = checkpoint_wrapper(self.layers[i])
                self.fusion_layers[i] = checkpoint_wrapper(
                    self.fusion_layers[i])

    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                key_padding_mask: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                memory_text: Tensor = None,
                text_attention_mask: Tensor = None,
                pos_text: Tensor = None,
                text_self_attention_masks: Tensor = None,
                position_ids: Tensor = None):
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
            memory_text (Tensor, optional): Memory text. It has shape (bs,
                len_text, text_embed_dims).
            text_attention_mask (Tensor, optional): Text token mask. It has
                shape (bs,len_text).
            pos_text (Tensor, optional): The positional encoding for text.
                Defaults to None.
            text_self_attention_masks (Tensor, optional): Text self attention
                mask. Defaults to None.
            position_ids (Tensor, optional): Text position ids.
                Defaults to None.
        """
        output = query
        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=query.device)
        if self.text_layers:
            # generate pos_text
            bs, n_text, _ = memory_text.shape
            if pos_text is None and position_ids is None:
                pos_text = (
                    torch.arange(n_text,
                                 device=memory_text.device).float().unsqueeze(
                                     0).unsqueeze(-1).repeat(bs, 1, 1))
                pos_text = get_text_sine_pos_embed(
                    pos_text, num_pos_feats=256, exchange_xy=False)
            if position_ids is not None:
                pos_text = get_text_sine_pos_embed(
                    position_ids[..., None],
                    num_pos_feats=256,
                    exchange_xy=False)

        # main process
        for layer_id, layer in enumerate(self.layers):
            if self.fusion_layers:
                if layer_id in [0, 5]:
                    output, memory_text = self.fusion_layers[layer_id](
                        visual_feature=output,
                        lang_feature=memory_text,
                        attention_mask_v=key_padding_mask,
                        attention_mask_l=text_attention_mask,
                    )
                else:
                    if layer_id in [1, 4]:
                        output_lite = output[:, level_start_index[1]:, :]
                        output_res = output[:, :level_start_index[1], :]
                        reference_points_lite = reference_points[:, level_start_index[1]:, ...]
                        query_pos_lite = query_pos[:, level_start_index[1]:, :]
                        spatial_shapes_lite = spatial_shapes[1:]
                        if key_padding_mask is not None:
                            key_padding_mask_lite = key_padding_mask[:, level_start_index[1]:]
                        else:
                            key_padding_mask_lite = None
                    elif layer_id in [2, 3]:
                        output_lite = output[:, level_start_index[2]:, :]
                        output_res = output[:, :level_start_index[2], :]
                        reference_points_lite = reference_points[:, level_start_index[2]:, ...]
                        query_pos_lite = query_pos[:, level_start_index[2]:, :]
                        spatial_shapes_lite = spatial_shapes[2:]
                        if key_padding_mask is not None:
                            key_padding_mask_lite = key_padding_mask[:, level_start_index[2]:]
                        else:
                            key_padding_mask_lite = None
                    output_lite, memory_text = self.fusion_layers[layer_id](
                        visual_feature=output_lite,
                        lang_feature=memory_text,
                        attention_mask_v=key_padding_mask_lite,
                        attention_mask_l=text_attention_mask,
                    )

            if self.text_layers:
                text_num_heads = self.text_layers[
                    layer_id].self_attn_cfg.num_heads
                memory_text = self.text_layers[layer_id](
                    query=memory_text,
                    query_pos=(pos_text if pos_text is not None else None),
                    attn_mask=~text_self_attention_masks.repeat(
                        text_num_heads, 1, 1),  # note we use ~ for mask here
                    key_padding_mask=None,
                )
            if layer_id in [0, 5]:
                output = layer(
                    query=output,
                    query_pos=query_pos,
                    reference_points=reference_points,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    key_padding_mask=key_padding_mask)
            else:
                output = torch.cat((output_res, output_lite), dim=1)
                output_lite = layer(
                    query=output_lite,
                    query_pos=query_pos_lite,
                    memory=output,
                    reference_points=reference_points_lite,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    key_padding_mask=key_padding_mask)
                output = torch.cat((output_res, output_lite), dim=1)
        return output, memory_text

class GroundingDinoStage2DecoupleTransformerEncoder(DeformableDetrTransformerEncoder):

    def __init__(self, text_layer_cfg: ConfigType,
                 fusion_layer_cfg: ConfigType,
                 low_layer_num, **kwargs) -> None:
        self.text_layer_cfg = text_layer_cfg
        self.fusion_layer_cfg = fusion_layer_cfg
        self.low_layer_num = low_layer_num
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = [DeformableDetrSparseTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.low_layer_num)]
        self.layers.insert(0, DeformableDetrTransformerEncoderLayer(**self.layer_cfg))
        self.layers.append(DeformableDetrTransformerEncoderLayer(**self.layer_cfg))
        self.layers = ModuleList(self.layers)
        self.text_layers = ModuleList([
            DetrTransformerEncoderLayer(**self.text_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.fusion_layer_cfg['feat_attn_layer_num'] = [1, 1]
        self.fusion_layers2 = ModuleList([
            TransformerLiteDecoupleFusionLayer(**self.fusion_layer_cfg)
            for _ in range(2)
        ])
        self.fusion_layer_cfg['feat_attn_layer_num'] = [1, 1, 1]
        self.fusion_layers3 = ModuleList([
            TransformerLiteDecoupleFusionLayer(**self.fusion_layer_cfg)
            for _ in range(2)
        ])
        self.fusion_layer_cfg['feat_attn_layer_num'] = [1, 1, 1, 1]
        self.fusion_layers4 = ModuleList([
            TransformerLiteDecoupleFusionLayer(**self.fusion_layer_cfg)
            for _ in range(2)
        ])
        self.embed_dims = self.layers[0].embed_dims
        # if self.num_cp > 0:
        #     if checkpoint_wrapper is None:
        #         raise NotImplementedError(
        #             'If you want to reduce GPU memory usage, \
        #             please install fairscale by executing the \
        #             following command: pip install fairscale.')
        #     for i in range(self.num_cp):
        #         self.layers[i] = checkpoint_wrapper(self.layers[i])
        #     for i in range(2):
        #         self.fusion_layers2[i] = checkpoint_wrapper(
        #             self.fusion_layers2[i])
        #         self.fusion_layers3[i] = checkpoint_wrapper(
        #             self.fusion_layers3[i])
        #         self.fusion_layers4[i] = checkpoint_wrapper(
        #             self.fusion_layers4[i])

    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                key_padding_mask: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                memory_text: Tensor = None,
                text_attention_mask: Tensor = None,
                pos_text: Tensor = None,
                text_self_attention_masks: Tensor = None,
                position_ids: Tensor = None):
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
            memory_text (Tensor, optional): Memory text. It has shape (bs,
                len_text, text_embed_dims).
            text_attention_mask (Tensor, optional): Text token mask. It has
                shape (bs,len_text).
            pos_text (Tensor, optional): The positional encoding for text.
                Defaults to None.
            text_self_attention_masks (Tensor, optional): Text self attention
                mask. Defaults to None.
            position_ids (Tensor, optional): Text position ids.
                Defaults to None.
        """
        output = query
        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=query.device)
        if self.text_layers:
            # generate pos_text
            bs, n_text, _ = memory_text.shape
            if pos_text is None and position_ids is None:
                pos_text = (
                    torch.arange(n_text,
                                 device=memory_text.device).float().unsqueeze(
                                     0).unsqueeze(-1).repeat(bs, 1, 1))
                pos_text = get_text_sine_pos_embed(
                    pos_text, num_pos_feats=256, exchange_xy=False)
            if position_ids is not None:
                pos_text = get_text_sine_pos_embed(
                    position_ids[..., None],
                    num_pos_feats=256,
                    exchange_xy=False)

        # main process
        for layer_id, layer in enumerate(self.layers):
            if self.fusion_layers2:
                if layer_id in [0, 5]:
                    output = self.fusion_layers4[[0, 5].index(layer_id)](
                        output=output,
                        memory_text=memory_text,
                        query_pos=query_pos,
                        pos_text=(pos_text if pos_text is not None else None),
                        attn_mask=None,
                        text_attention_mask=text_attention_mask,
                        spatial_shapes=spatial_shapes
                    )
                else:
                    if layer_id in [1, 4]:
                        output_lite = output[:, level_start_index[1]:, :]
                        output_res = output[:, :level_start_index[1], :]
                        reference_points_lite = reference_points[:, level_start_index[1]:, ...]
                        query_pos_lite = query_pos[:, level_start_index[1]:, :]
                        spatial_shapes_lite = spatial_shapes[1:]
                        if key_padding_mask is not None:
                            key_padding_mask_lite = key_padding_mask[:, level_start_index[1]:]
                        else:
                            key_padding_mask_lite = None
                        output_lite = self.fusion_layers3[[1, 4].index(layer_id)](
                            output=output_lite,
                            memory_text=memory_text,
                            query_pos=query_pos_lite,
                            pos_text=(pos_text if pos_text is not None else None),
                            attn_mask=None,
                            text_attention_mask=text_attention_mask,
                            spatial_shapes=spatial_shapes_lite
                        )

                    elif layer_id in [2, 3]:
                        output_lite = output[:, level_start_index[2]:, :]
                        output_res = output[:, :level_start_index[2], :]
                        reference_points_lite = reference_points[:, level_start_index[2]:, ...]
                        query_pos_lite = query_pos[:, level_start_index[2]:, :]
                        spatial_shapes_lite = spatial_shapes[2:]
                        if key_padding_mask is not None:
                            key_padding_mask_lite = key_padding_mask[:, level_start_index[2]:]
                        else:
                            key_padding_mask_lite = None
                        output_lite = self.fusion_layers2[[2, 3].index(layer_id)](
                            output=output_lite,
                            memory_text=memory_text,
                            query_pos=query_pos_lite,
                            pos_text=(pos_text if pos_text is not None else None),
                            attn_mask=None,
                            text_attention_mask=text_attention_mask,
                            spatial_shapes=spatial_shapes_lite
                        )

            if self.text_layers:
                text_num_heads = self.text_layers[
                    layer_id].self_attn_cfg.num_heads
                memory_text = self.text_layers[layer_id](
                    query=memory_text,
                    query_pos=(pos_text if pos_text is not None else None),
                    attn_mask=~text_self_attention_masks.repeat(
                        text_num_heads, 1, 1),  # note we use ~ for mask here
                    key_padding_mask=None,
                )
            if layer_id in [0, 5]:
                output = layer(
                    query=output,
                    query_pos=query_pos,
                    reference_points=reference_points,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    key_padding_mask=key_padding_mask)
            else:
                output = torch.cat((output_res, output_lite), dim=1)
                output_lite = layer(
                    query=output_lite,
                    query_pos=query_pos_lite,
                    memory=output,
                    reference_points=reference_points_lite,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    key_padding_mask=key_padding_mask)
                output = torch.cat((output_res, output_lite), dim=1)
        return output, memory_text

class GroundingDinoAttnGuideTransformerEncoder(DeformableDetrTransformerEncoder):

    def __init__(self, text_layer_cfg: ConfigType,
                 fusion_layer_cfg: ConfigType, **kwargs) -> None:
        self.text_layer_cfg = text_layer_cfg
        self.fusion_layer_cfg = fusion_layer_cfg
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList([
            DetrTransformerWeightEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.text_layers = ModuleList([
            DetrTransformerEncoderLayer(**self.text_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.fusion_layers = ModuleList([
            SingleScaleBiAttentionBlock(**self.fusion_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.num_cp > 0:
            if checkpoint_wrapper is None:
                raise NotImplementedError(
                    'If you want to reduce GPU memory usage, \
                    please install fairscale by executing the \
                    following command: pip install fairscale.')
            for i in range(self.num_cp):
                self.layers[i] = checkpoint_wrapper(self.layers[i])
                self.fusion_layers[i] = checkpoint_wrapper(
                    self.fusion_layers[i])

    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                key_padding_mask: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                memory_text: Tensor = None,
                text_attention_mask: Tensor = None,
                pos_text: Tensor = None,
                text_self_attention_masks: Tensor = None,
                position_ids: Tensor = None):
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
            memory_text (Tensor, optional): Memory text. It has shape (bs,
                len_text, text_embed_dims).
            text_attention_mask (Tensor, optional): Text token mask. It has
                shape (bs,len_text).
            pos_text (Tensor, optional): The positional encoding for text.
                Defaults to None.
            text_self_attention_masks (Tensor, optional): Text self attention
                mask. Defaults to None.
            position_ids (Tensor, optional): Text position ids.
                Defaults to None.
        """
        output = query
        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=query.device)
        if self.text_layers:
            # generate pos_text
            bs, n_text, _ = memory_text.shape
            if pos_text is None and position_ids is None:
                pos_text = (
                    torch.arange(n_text,
                                 device=memory_text.device).float().unsqueeze(
                                     0).unsqueeze(-1).repeat(bs, 1, 1))
                pos_text = get_text_sine_pos_embed(
                    pos_text, num_pos_feats=256, exchange_xy=False)
            if position_ids is not None:
                pos_text = get_text_sine_pos_embed(
                    position_ids[..., None],
                    num_pos_feats=256,
                    exchange_xy=False)

        # main process
        for layer_id, layer in enumerate(self.layers):
            # use p5 to calculate fusion model
            P5 = output[:, level_start_index[-1]:]
            P4 = output[:, level_start_index[1]:level_start_index[-1]]
            P3 = output[:, :level_start_index[1]]
            query_pos_P5 = query_pos[:, level_start_index[-1]:]
            if key_padding_mask is not None:
                key_padding_mask_P5 = key_padding_mask[:, level_start_index[-1]:]
            else:
                key_padding_mask_P5 = None
            if self.fusion_layers:
                P5, memory_text = self.fusion_layers[layer_id](
                    visual_feature=P5,
                    lang_feature=memory_text,
                    attention_mask_v=key_padding_mask_P5,
                    attention_mask_l=text_attention_mask,
                )
            if self.text_layers:
                text_num_heads = self.text_layers[
                    layer_id].self_attn_cfg.num_heads
                memory_text = self.text_layers[layer_id](
                    query=memory_text,
                    query_pos=(pos_text if pos_text is not None else None),
                    attn_mask=~text_self_attention_masks.repeat(
                        text_num_heads, 1, 1),  # note we use ~ for mask here
                    key_padding_mask=None,
                )
            output = layer(
                query=P5,
                P4=P4,
                P3=P3,
                spatial_shapes=spatial_shapes,
                query_pos=query_pos_P5,
                key_padding_mask=key_padding_mask_P5)
        return output, memory_text

class GroundingDinoLiteTransformerEncoder(DeformableDetrTransformerEncoder):

    def __init__(self, text_layer_cfg: ConfigType,
                 fusion_layer_cfg: ConfigType, **kwargs) -> None:
        self.text_layer_cfg = text_layer_cfg
        self.fusion_layer_cfg = fusion_layer_cfg
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList([
            DeformableDetrSparseTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.text_layers = ModuleList([
            DetrTransformerEncoderLayer(**self.text_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.fusion_layers = ModuleList([
            SingleScaleBiAttentionBlock(**self.fusion_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.num_cp > 0:
            if checkpoint_wrapper is None:
                raise NotImplementedError(
                    'If you want to reduce GPU memory usage, \
                    please install fairscale by executing the \
                    following command: pip install fairscale.')
            for i in range(self.num_cp):
                self.layers[i] = checkpoint_wrapper(self.layers[i])
                self.fusion_layers[i] = checkpoint_wrapper(
                    self.fusion_layers[i])

    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                key_padding_mask: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                memory_text: Tensor = None,
                text_attention_mask: Tensor = None,
                pos_text: Tensor = None,
                text_self_attention_masks: Tensor = None,
                position_ids: Tensor = None):
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
            memory_text (Tensor, optional): Memory text. It has shape (bs,
                len_text, text_embed_dims).
            text_attention_mask (Tensor, optional): Text token mask. It has
                shape (bs,len_text).
            pos_text (Tensor, optional): The positional encoding for text.
                Defaults to None.
            text_self_attention_masks (Tensor, optional): Text self attention
                mask. Defaults to None.
            position_ids (Tensor, optional): Text position ids.
                Defaults to None.
        """
        output = query
        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=query.device)
        if self.text_layers:
            # generate pos_text
            bs, n_text, _ = memory_text.shape
            if pos_text is None and position_ids is None:
                pos_text = (
                    torch.arange(n_text,
                                 device=memory_text.device).float().unsqueeze(
                                     0).unsqueeze(-1).repeat(bs, 1, 1))
                pos_text = get_text_sine_pos_embed(
                    pos_text, num_pos_feats=256, exchange_xy=False)
            if position_ids is not None:
                pos_text = get_text_sine_pos_embed(
                    position_ids[..., None],
                    num_pos_feats=256,
                    exchange_xy=False)

        # main process
        for layer_id, layer in enumerate(self.layers):
            if layer_id in [2, 5]:
                output_lite = output[:, :level_start_index[1], :]
                output_res = output[:, level_start_index[1]:, :]
                reference_points_lite = reference_points[:, :level_start_index[1], ...]
                query_pos_lite = query_pos[:, :level_start_index[1], :]
                if key_padding_mask is not None:
                    key_padding_mask_lite = key_padding_mask[:, :level_start_index[1]]
                else:
                    key_padding_mask_lite = None
            else:
                output_lite = output[:, level_start_index[1]:, :]
                output_res = output[:, :level_start_index[1], :]
                reference_points_lite = reference_points[:, level_start_index[1]:, ...]
                query_pos_lite = query_pos[:, level_start_index[1]:, :]
                if key_padding_mask is not None:
                    key_padding_mask_lite = key_padding_mask[:, level_start_index[1]:]
                else:
                    key_padding_mask_lite = None

            if self.fusion_layers:
                output_lite, memory_text = self.fusion_layers[layer_id](
                    visual_feature=output_lite,
                    lang_feature=memory_text,
                    attention_mask_v=key_padding_mask_lite,
                    attention_mask_l=text_attention_mask,
                )
            if self.text_layers:
                text_num_heads = self.text_layers[
                    layer_id].self_attn_cfg.num_heads
                memory_text = self.text_layers[layer_id](
                    query=memory_text,
                    query_pos=(pos_text if pos_text is not None else None),
                    attn_mask=~text_self_attention_masks.repeat(
                        text_num_heads, 1, 1),  # note we use ~ for mask here
                    key_padding_mask=None,
                )
            output_lite = layer(
                query=output_lite,
                query_pos=query_pos_lite,
                memory=output,
                reference_points=reference_points_lite,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=key_padding_mask)
            if layer_id in [2, 5]:
                output = torch.cat((output_lite, output_res), dim=1)
            else:
                output = torch.cat((output_res, output_lite), dim=1)
        return output, memory_text

class GroundingDinoLiteDecoupleTransformerEncoder(DeformableDetrTransformerEncoder):

    def __init__(self, text_layer_cfg: ConfigType,
                 fusion_layer_cfg: ConfigType,
                 low_layer_num,
                 guide=False,
                 **kwargs) -> None:
        self.text_layer_cfg = text_layer_cfg
        self.fusion_layer_cfg = fusion_layer_cfg
        self.low_layer_num = low_layer_num
        self.guide = guide
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList([
            DeformableDetrSparseTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.text_layers = ModuleList([
            DetrTransformerEncoderLayer(**self.text_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.fusion_layers = [
            TransformerLiteDecoupleFusionLayer(**self.fusion_layer_cfg)
            for _ in range(self.num_layers - self.low_layer_num)
        ]
        for n in range(self.low_layer_num):
            self.fusion_layer_cfg['feat_attn_layer_num'] = [1]
            self.fusion_layers.append(TransformerLiteDecoupleFusionLayer(**self.fusion_layer_cfg))
        self.fusion_layers = ModuleList(self.fusion_layers)
        self.embed_dims = self.layers[0].embed_dims
        # if self.num_cp > 0:
        #     if checkpoint_wrapper is None:
        #         raise NotImplementedError(
        #             'If you want to reduce GPU memory usage, \
        #             please install fairscale by executing the \
        #             following command: pip install fairscale.')
        #     for i in range(self.num_cp):
        #         self.layers[i] = checkpoint_wrapper(self.layers[i])
        #         self.fusion_layers[i] = checkpoint_wrapper(
        #             self.fusion_layers[i])

    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                key_padding_mask: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                memory_text: Tensor = None,
                text_attention_mask: Tensor = None,
                pos_text: Tensor = None,
                text_self_attention_masks: Tensor = None,
                position_ids: Tensor = None):
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
            memory_text (Tensor, optional): Memory text. It has shape (bs,
                len_text, text_embed_dims).
            text_attention_mask (Tensor, optional): Text token mask. It has
                shape (bs,len_text).
            pos_text (Tensor, optional): The positional encoding for text.
                Defaults to None.
            text_self_attention_masks (Tensor, optional): Text self attention
                mask. Defaults to None.
            position_ids (Tensor, optional): Text position ids.
                Defaults to None.
        """
        output = query
        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=query.device)
        if self.text_layers:
            # generate pos_text
            bs, n_text, _ = memory_text.shape
            if pos_text is None and position_ids is None:
                pos_text = (
                    torch.arange(n_text,
                                 device=memory_text.device).float().unsqueeze(
                                     0).unsqueeze(-1).repeat(bs, 1, 1))
                pos_text = get_text_sine_pos_embed(
                    pos_text, num_pos_feats=256, exchange_xy=False)
            if position_ids is not None:
                pos_text = get_text_sine_pos_embed(
                    position_ids[..., None],
                    num_pos_feats=256,
                    exchange_xy=False)

        # main process
        for layer_id, layer in enumerate(self.layers):
            if layer_id == self.num_layers - self.low_layer_num:
                output_lite = output[:, :level_start_index[1], :]
                output_res = output[:, level_start_index[1]:, :]
                reference_points_lite = reference_points[:, :level_start_index[1], ...]
                query_pos_lite = query_pos[:, :level_start_index[1], :]
                spatial_shapes_lite = spatial_shapes[:1]
                # if key_padding_mask is not None:
                #     key_padding_mask_lite = key_padding_mask[:, :level_start_index[1]]
                # else:
                #     key_padding_mask_lite = None
            else:
                output_lite = output[:, level_start_index[1]:, :]
                output_res = output[:, :level_start_index[1], :]
                reference_points_lite = reference_points[:, level_start_index[1]:, ...]
                query_pos_lite = query_pos[:, level_start_index[1]:, :]
                spatial_shapes_lite = spatial_shapes[1:]
                # if key_padding_mask is not None:
                #     key_padding_mask_lite = key_padding_mask[:, level_start_index[1]:]
                # else:
                #     key_padding_mask_lite = None

            if self.fusion_layers:
                output_lite = self.fusion_layers[layer_id](
                    output=output_lite,
                    memory_text=memory_text,
                    query_pos=query_pos_lite,
                    pos_text=(pos_text if pos_text is not None else None),
                    attn_mask=None,
                    text_attention_mask=text_attention_mask,
                    spatial_shapes=spatial_shapes_lite
                )
            if self.text_layers:
                text_num_heads = self.text_layers[
                    layer_id].self_attn_cfg.num_heads
                memory_text = self.text_layers[layer_id](
                    query=memory_text,
                    query_pos=(pos_text if pos_text is not None else None),
                    attn_mask=~text_self_attention_masks.repeat(
                        text_num_heads, 1, 1),  # note we use ~ for mask here
                    key_padding_mask=None,
                )
            if layer_id == self.num_layers - self.low_layer_num:
                output = torch.cat((output_lite, output_res), dim=1)
            else:
                if self.guide:
                    H2, W2 = spatial_shapes_lite[0]
                    H1, W1 = spatial_shapes[0]
                    l2_ind = (H2 * W2).item()
                    bs, _, dim = output_res.shape
                    output_res = output_res + F.interpolate(output_lite[:,:l2_ind,:].reshape(bs, H2, W2, dim).permute(0,-1, 1, 2),
                                           size=(H1, W1), mode='nearest').reshape(bs,dim,
                                                                                   output_res.shape[1]).permute(0,2,1).contiguous()
                output = torch.cat((output_res, output_lite), dim=1)
            output_lite = layer(
                query=output_lite,
                query_pos=query_pos_lite,
                memory=output,
                reference_points=reference_points_lite,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=key_padding_mask)
            if layer_id == self.num_layers - self.low_layer_num:
                output = torch.cat((output_lite, output_res), dim=1)
            else:
                if self.guide:
                    H2, W2 = spatial_shapes_lite[0]
                    H1, W1 = spatial_shapes[0]
                    l2_ind = (H2 * W2).item()
                    bs, _, dim = output_res.shape
                    output_res = output_res + F.interpolate(output_lite[:,:l2_ind,:].reshape(bs, H2, W2, dim).permute(0,-1, 1, 2),
                                           size=(H1, W1), mode='nearest').reshape(bs,dim,
                                                                                   output_res.shape[1]).permute(0,2,1).contiguous()

                output = torch.cat((output_res, output_lite), dim=1)
        return output, memory_text


class GroundingDinoGuideTransformerEncoder(DeformableDetrTransformerEncoder):

    def __init__(self, text_layer_cfg: ConfigType,
                 fusion_layer_cfg: ConfigType,
                 low_layer_num,
                 **kwargs) -> None:
        self.text_layer_cfg = text_layer_cfg
        self.fusion_layer_cfg = fusion_layer_cfg
        self.low_layer_num = low_layer_num
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList([
            DeformableDetrSparseTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.text_layers = ModuleList([
            DetrTransformerEncoderLayer(**self.text_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.fusion_layers = [
            TransformerGuideFusionLayer(**self.fusion_layer_cfg)
            for _ in range(self.num_layers - self.low_layer_num)
        ]
        for n in range(self.low_layer_num):
            self.fusion_layer_cfg['feat_attn_layer_num'] = [1]
            self.fusion_layers.append(TransformerGuideFusionLayer(**self.fusion_layer_cfg))
        self.fusion_layers = ModuleList(self.fusion_layers)
        self.embed_dims = self.layers[0].embed_dims
        # if self.num_cp > 0:
        #     if checkpoint_wrapper is None:
        #         raise NotImplementedError(
        #             'If you want to reduce GPU memory usage, \
        #             please install fairscale by executing the \
        #             following command: pip install fairscale.')
        #     for i in range(self.num_cp):
        #         self.layers[i] = checkpoint_wrapper(self.layers[i])
        #         self.fusion_layers[i] = checkpoint_wrapper(
        #             self.fusion_layers[i])

    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                key_padding_mask: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                memory_text: Tensor = None,
                text_attention_mask: Tensor = None,
                pos_text: Tensor = None,
                text_self_attention_masks: Tensor = None,
                position_ids: Tensor = None):
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
            memory_text (Tensor, optional): Memory text. It has shape (bs,
                len_text, text_embed_dims).
            text_attention_mask (Tensor, optional): Text token mask. It has
                shape (bs,len_text).
            pos_text (Tensor, optional): The positional encoding for text.
                Defaults to None.
            text_self_attention_masks (Tensor, optional): Text self attention
                mask. Defaults to None.
            position_ids (Tensor, optional): Text position ids.
                Defaults to None.
        """
        output = query
        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=query.device)
        if self.text_layers:
            # generate pos_text
            bs, n_text, _ = memory_text.shape
            if pos_text is None and position_ids is None:
                pos_text = (
                    torch.arange(n_text,
                                 device=memory_text.device).float().unsqueeze(
                                     0).unsqueeze(-1).repeat(bs, 1, 1))
                pos_text = get_text_sine_pos_embed(
                    pos_text, num_pos_feats=256, exchange_xy=False)
            if position_ids is not None:
                pos_text = get_text_sine_pos_embed(
                    position_ids[..., None],
                    num_pos_feats=256,
                    exchange_xy=False)

        # main process
        for layer_id, layer in enumerate(self.layers):
            if self.fusion_layers:
                output = self.fusion_layers[layer_id](
                    output=output,
                    memory_text=memory_text,
                    query_pos=query_pos,
                    pos_text=(pos_text if pos_text is not None else None),
                    attn_mask=None,
                    text_attention_mask=text_attention_mask,
                    spatial_shapes=spatial_shapes
                )
            if self.text_layers:
                text_num_heads = self.text_layers[
                    layer_id].self_attn_cfg.num_heads
                memory_text = self.text_layers[layer_id](
                    query=memory_text,
                    query_pos=(pos_text if pos_text is not None else None),
                    attn_mask=~text_self_attention_masks.repeat(
                        text_num_heads, 1, 1),  # note we use ~ for mask here
                    key_padding_mask=None,
                )
            start = 0
            output_ = []
            query_pos_ = []
            reference_points_ = []
            feat_inds = [(H * W).item() for (H, W) in spatial_shapes]
            for feat_ind in feat_inds:
                output_ind = output[:, start:start + feat_ind, :]
                query_pos_ind = query_pos[:, start:start + feat_ind, :]
                reference_points_ind = reference_points[:, start:start + feat_ind, ...]
                start += feat_ind
                output_.append(output_ind)
                query_pos_.append(query_pos_ind)
                reference_points_.append(reference_points_ind)
            output_lite = torch.cat([output_[1], output_[-1]], dim=1)
            query_pos_lite = torch.cat([query_pos_[1], query_pos_[-1]], dim=1)
            reference_points_lite = torch.cat([reference_points_[1], reference_points_[-1]], dim=1)

            output_lite = layer(
                query=output_lite,
                query_pos=query_pos_lite,
                memory=output,
                reference_points=reference_points_lite,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=key_padding_mask)

            bs, _, dim = output_[0].shape
            h1, w1 = spatial_shapes[0]
            h2, w2 = spatial_shapes[1]
            h3, w3 = spatial_shapes[2]
            h4, w4 = spatial_shapes[-1]
            output_final = []
            l2_output = output_lite[:, :output_[1].shape[1], :]
            # l3_output = output_lite[:, output_[1].shape[1]:output_[1].shape[1] + output_[2].shape[1], :]
            l4_output = output_lite[:, output_[1].shape[1]:, :]
            # output_final.append(output_[0])
            output_final.append(output_[0] + F.interpolate(l2_output.reshape(bs, h2, w2, dim).permute(0,-1, 1, 2),
                                           size=(h1, w1), mode='nearest').reshape(bs,dim,
                                                                                   output_[0].shape[1]).permute(0,2,1).contiguous())
            output_final.append(l2_output)
            # output_final.append(l3_output)
            output_final.append(output_[2] + F.interpolate(l4_output.reshape(bs, h4, w4, dim).permute(0,-1, 1, 2),
                                           size=(h3, w3), mode='nearest').reshape(bs,dim,
                                                                                   output_[2].shape[1]).permute(0,2,1).contiguous())
            output_final.append(l4_output)
            output = torch.cat(output_final, dim=1)

        return output, memory_text

class GroundingDinoMidGuideTransformerEncoder(DeformableDetrTransformerEncoder):

    def __init__(self, text_layer_cfg: ConfigType,
                 fusion_layer_cfg: ConfigType,
                 low_layer_num,
                 **kwargs) -> None:
        self.text_layer_cfg = text_layer_cfg
        self.fusion_layer_cfg = fusion_layer_cfg
        self.low_layer_num = low_layer_num
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList([
            DeformableDetrTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.text_layers = ModuleList([
            DetrTransformerEncoderLayer(**self.text_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.fusion_layers = [
            TransformerGuideFusionLayer(**self.fusion_layer_cfg)
            for _ in range(self.num_layers - self.low_layer_num)
        ]
        for n in range(self.low_layer_num):
            self.fusion_layer_cfg['feat_attn_layer_num'] = [1]
            self.fusion_layers.append(TransformerGuideFusionLayer(**self.fusion_layer_cfg))
        self.fusion_layers = ModuleList(self.fusion_layers)
        self.embed_dims = self.layers[0].embed_dims
        if self.num_cp > 0:
            if checkpoint_wrapper is None:
                raise NotImplementedError(
                    'If you want to reduce GPU memory usage, \
                    please install fairscale by executing the \
                    following command: pip install fairscale.')
            for i in range(self.num_cp):
                self.layers[i] = checkpoint_wrapper(self.layers[i])
                self.fusion_layers[i] = checkpoint_wrapper(
                    self.fusion_layers[i])

    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                key_padding_mask: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                memory_text: Tensor = None,
                text_attention_mask: Tensor = None,
                pos_text: Tensor = None,
                text_self_attention_masks: Tensor = None,
                position_ids: Tensor = None):
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
            memory_text (Tensor, optional): Memory text. It has shape (bs,
                len_text, text_embed_dims).
            text_attention_mask (Tensor, optional): Text token mask. It has
                shape (bs,len_text).
            pos_text (Tensor, optional): The positional encoding for text.
                Defaults to None.
            text_self_attention_masks (Tensor, optional): Text self attention
                mask. Defaults to None.
            position_ids (Tensor, optional): Text position ids.
                Defaults to None.
        """
        output = query
        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=query.device)
        if self.text_layers:
            # generate pos_text
            bs, n_text, _ = memory_text.shape
            if pos_text is None and position_ids is None:
                pos_text = (
                    torch.arange(n_text,
                                 device=memory_text.device).float().unsqueeze(
                                     0).unsqueeze(-1).repeat(bs, 1, 1))
                pos_text = get_text_sine_pos_embed(
                    pos_text, num_pos_feats=256, exchange_xy=False)
            if position_ids is not None:
                pos_text = get_text_sine_pos_embed(
                    position_ids[..., None],
                    num_pos_feats=256,
                    exchange_xy=False)

        # main process
        for layer_id, layer in enumerate(self.layers):
            if self.fusion_layers:
                output = self.fusion_layers[layer_id](
                    output=output,
                    memory_text=memory_text,
                    query_pos=query_pos,
                    pos_text=(pos_text if pos_text is not None else None),
                    attn_mask=None,
                    text_attention_mask=text_attention_mask,
                    spatial_shapes=spatial_shapes
                )
            if self.text_layers:
                text_num_heads = self.text_layers[
                    layer_id].self_attn_cfg.num_heads
                memory_text = self.text_layers[layer_id](
                    query=memory_text,
                    query_pos=(pos_text if pos_text is not None else None),
                    attn_mask=~text_self_attention_masks.repeat(
                        text_num_heads, 1, 1),  # note we use ~ for mask here
                    key_padding_mask=None,
                )
            output = layer(
                query=output,
                query_pos=query_pos,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=key_padding_mask)

        return output, memory_text

class GroundingDinoDecoupleEncTransformerEncoder(DeformableDetrTransformerEncoder):

    def __init__(self, text_layer_cfg: ConfigType,
                 fusion_layer_cfg: ConfigType, **kwargs) -> None:
        self.text_layer_cfg = text_layer_cfg
        self.fusion_layer_cfg = fusion_layer_cfg
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList([
            DeformableDetrDecoupleTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.text_layers = ModuleList([
            DetrTransformerEncoderLayer(**self.text_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.fusion_layers = ModuleList([
            SingleScaleBiAttentionBlock(**self.fusion_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.num_cp > 0:
            if checkpoint_wrapper is None:
                raise NotImplementedError(
                    'If you want to reduce GPU memory usage, \
                    please install fairscale by executing the \
                    following command: pip install fairscale.')
            for i in range(self.num_cp):
                self.layers[i] = checkpoint_wrapper(self.layers[i])
                self.fusion_layers[i] = checkpoint_wrapper(
                    self.fusion_layers[i])

    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                key_padding_mask: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                memory_text: Tensor = None,
                text_attention_mask: Tensor = None,
                pos_text: Tensor = None,
                text_self_attention_masks: Tensor = None,
                position_ids: Tensor = None):
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
            memory_text (Tensor, optional): Memory text. It has shape (bs,
                len_text, text_embed_dims).
            text_attention_mask (Tensor, optional): Text token mask. It has
                shape (bs,len_text).
            pos_text (Tensor, optional): The positional encoding for text.
                Defaults to None.
            text_self_attention_masks (Tensor, optional): Text self attention
                mask. Defaults to None.
            position_ids (Tensor, optional): Text position ids.
                Defaults to None.
        """
        output = query
        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=query.device)
        if self.text_layers:
            # generate pos_text
            bs, n_text, _ = memory_text.shape
            if pos_text is None and position_ids is None:
                pos_text = (
                    torch.arange(n_text,
                                 device=memory_text.device).float().unsqueeze(
                                     0).unsqueeze(-1).repeat(bs, 1, 1))
                pos_text = get_text_sine_pos_embed(
                    pos_text, num_pos_feats=256, exchange_xy=False)
            if position_ids is not None:
                pos_text = get_text_sine_pos_embed(
                    position_ids[..., None],
                    num_pos_feats=256,
                    exchange_xy=False)

        # main process
        for layer_id, layer in enumerate(self.layers):
            if self.fusion_layers:
                output, memory_text = self.fusion_layers[layer_id](
                    visual_feature=output,
                    lang_feature=memory_text,
                    attention_mask_v=key_padding_mask,
                    attention_mask_l=text_attention_mask,
                )
            if self.text_layers:
                text_num_heads = self.text_layers[
                    layer_id].self_attn_cfg.num_heads
                memory_text = self.text_layers[layer_id](
                    query=memory_text,
                    query_pos=(pos_text if pos_text is not None else None),
                    attn_mask=~text_self_attention_masks.repeat(
                        text_num_heads, 1, 1),  # note we use ~ for mask here
                    key_padding_mask=None,
                )
            output = layer(
                query=output,
                query_pos=query_pos,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=key_padding_mask)
        return output, memory_text

class GroundingDinoCAMTransformerEncoder(DeformableDetrTransformerEncoder):

    def __init__(self, text_layer_cfg: ConfigType,
                 fusion_layer_cfg: ConfigType, **kwargs) -> None:
        self.text_layer_cfg = text_layer_cfg
        self.fusion_layer_cfg = fusion_layer_cfg
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList([
            DeformableDetrTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.text_layers = ModuleList([
            DetrTransformerEncoderLayer(**self.text_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.fusion_layer_cfg_layer1 = self.fusion_layer_cfg['layer1']
        self.fusion_layer_cfg_layer2 = self.fusion_layer_cfg['layer2']
        self.fusion_layer_cfg_layer3 = self.fusion_layer_cfg['layer3']
        self.fusion_layer_cfg_layer4 = self.fusion_layer_cfg['layer4']
        self.fusion_layers1 = ModuleList([
            MultiheadAttention(**self.fusion_layer_cfg_layer1)
            for _ in range(self.num_layers)
        ])
        self.fusion_layers2 = ModuleList([
            MultiheadAttention(**self.fusion_layer_cfg_layer2)
            for _ in range(self.num_layers)
        ])
        self.fusion_layers3 = ModuleList([
            MultiheadAttention(**self.fusion_layer_cfg_layer3)
            for _ in range(self.num_layers * 2)
        ])
        self.fusion_layers4 = ModuleList([
            MultiheadAttention(**self.fusion_layer_cfg_layer4)
            for _ in range(self.num_layers * 2)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.num_cp > 0:
            if checkpoint_wrapper is None:
                raise NotImplementedError(
                    'If you want to reduce GPU memory usage, \
                    please install fairscale by executing the \
                    following command: pip install fairscale.')
            for i in range(self.num_cp):
                self.layers[i] = checkpoint_wrapper(self.layers[i])
                self.fusion_layers1[i] = checkpoint_wrapper(
                    self.fusion_layers1[i])
                self.fusion_layers2[i] = checkpoint_wrapper(
                    self.fusion_layers2[i])
                self.fusion_layers3[i] = checkpoint_wrapper(
                    self.fusion_layers3[i])
                self.fusion_layers4[i] = checkpoint_wrapper(
                    self.fusion_layers4[i])

    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                key_padding_mask: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                memory_text: Tensor = None,
                text_attention_mask: Tensor = None,
                pos_text: Tensor = None,
                text_self_attention_masks: Tensor = None,
                position_ids: Tensor = None):
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
            memory_text (Tensor, optional): Memory text. It has shape (bs,
                len_text, text_embed_dims).
            text_attention_mask (Tensor, optional): Text token mask. It has
                shape (bs,len_text).
            pos_text (Tensor, optional): The positional encoding for text.
                Defaults to None.
            text_self_attention_masks (Tensor, optional): Text self attention
                mask. Defaults to None.
            position_ids (Tensor, optional): Text position ids.
                Defaults to None.
        """
        output = query
        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=query.device)
        if self.text_layers:
            # generate pos_text
            bs, n_text, _ = memory_text.shape
            if pos_text is None and position_ids is None:
                pos_text = (
                    torch.arange(n_text,
                                 device=memory_text.device).float().unsqueeze(
                                     0).unsqueeze(-1).repeat(bs, 1, 1))
                pos_text = get_text_sine_pos_embed(
                    pos_text, num_pos_feats=256, exchange_xy=False)
            if position_ids is not None:
                pos_text = get_text_sine_pos_embed(
                    position_ids[..., None],
                    num_pos_feats=256,
                    exchange_xy=False)

        # main process
        feat_inds = [(H * W).item() for (H, W) in spatial_shapes]
        feat_num = len(feat_inds)

        for layer_id, layer in enumerate(self.layers):
            if self.fusion_layers1:
                output_ = []
                start = 0
                for ind, feat_ind in enumerate(feat_inds):
                    output_ind = output[:, start:start + feat_ind, :]
                    if key_padding_mask is not None:
                        key_padding_mask_ind = key_padding_mask[:, start:start + feat_ind]
                    else:
                        key_padding_mask_ind = None
                    query_pos_ind = query_pos[:, start:start + feat_ind, :]
                    start += feat_ind
                    if ind == 0:
                        output_ind = self.fusion_layers1[layer_id](
                            query=output_ind,
                            key=memory_text,
                            value=memory_text,
                            query_pos=query_pos_ind,
                            key_pos=(pos_text if pos_text is not None else None),
                            attn_mask=None,
                            key_padding_mask=text_attention_mask,
                        )
                        output_.append(output_ind)
                    if ind == 1:
                        output_ind = self.fusion_layers2[layer_id](
                            query=output_ind,
                            key=memory_text,
                            value=memory_text,
                            query_pos=query_pos_ind,
                            key_pos=(pos_text if pos_text is not None else None),
                            attn_mask=None,
                            key_padding_mask=text_attention_mask,
                        )
                        output_.append(output_ind)
                    if ind == 2:
                        output_ind = self.fusion_layers3[layer_id * 2](
                            query=output_ind,
                            key=memory_text,
                            value=memory_text,
                            query_pos=query_pos_ind,
                            key_pos=(pos_text if pos_text is not None else None),
                            attn_mask=None,
                            key_padding_mask=text_attention_mask,
                        )
                        output_ind = self.fusion_layers3[layer_id * 2 + 1](
                            query=output_ind,
                            key=memory_text,
                            value=memory_text,
                            query_pos=query_pos_ind,
                            key_pos=(pos_text if pos_text is not None else None),
                            attn_mask=None,
                            key_padding_mask=text_attention_mask,
                        )
                        output_.append(output_ind)
                    if ind == 3:
                        output_ind = self.fusion_layers4[layer_id * 2](
                            query=output_ind,
                            key=memory_text,
                            value=memory_text,
                            query_pos=query_pos_ind,
                            key_pos=(pos_text if pos_text is not None else None),
                            attn_mask=None,
                            key_padding_mask=text_attention_mask,
                        )
                        output_ind = self.fusion_layers4[layer_id * 2 + 1](
                            query=output_ind,
                            key=memory_text,
                            value=memory_text,
                            query_pos=query_pos_ind,
                            key_pos=(pos_text if pos_text is not None else None),
                            attn_mask=None,
                            key_padding_mask=text_attention_mask,
                        )
                        output_.append(output_ind)
                output = torch.cat(output_, dim=1)
            if self.text_layers:
                text_num_heads = self.text_layers[
                    layer_id].self_attn_cfg.num_heads
                memory_text = self.text_layers[layer_id](
                    query=memory_text,
                    query_pos=(pos_text if pos_text is not None else None),
                    attn_mask=~text_self_attention_masks.repeat(
                        text_num_heads, 1, 1),  # note we use ~ for mask here
                    key_padding_mask=None,
                )
            output = layer(
                query=output,
                query_pos=query_pos,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=key_padding_mask)
        return output, memory_text


class GroundingDinoTransformerEncoder_15(DetrTransformerEncoder):

    def __init__(self, text_layer_cfg: ConfigType,
                 fusion_layer_cfg: ConfigType,
                 CS_fusion_layer_cfg: ConfigType,
                 **kwargs) -> None:
        self.text_layer_cfg = text_layer_cfg
        self.fusion_layer_cfg = fusion_layer_cfg
        self.CS_fusion_layer_cfg = CS_fusion_layer_cfg
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList([
            DetrTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.text_layers = ModuleList([
            DetrTransformerEncoderLayer(**self.text_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.fusion_layers = ModuleList([
            SingleScaleBiAttentionBlock(**self.fusion_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.cross_scale_fusion = CCFF(**self.CS_fusion_layer_cfg)

        self.embed_dims = self.layers[0].embed_dims
        # if self.num_cp > 0:
        #     if checkpoint_wrapper is None:
        #         raise NotImplementedError(
        #             'If you want to reduce GPU memory usage, \
        #             please install fairscale by executing the \
        #             following command: pip install fairscale.')
        #     for i in range(self.num_cp):
        #         self.layers[i] = checkpoint_wrapper(self.layers[i])
        #         self.fusion_layers[i] = checkpoint_wrapper(
        #             self.fusion_layers[i])


    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                key_padding_mask: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                memory_text: Tensor = None,
                text_attention_mask: Tensor = None,
                pos_text: Tensor = None,
                text_self_attention_masks: Tensor = None,
                position_ids: Tensor = None):
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
            memory_text (Tensor, optional): Memory text. It has shape (bs,
                len_text, text_embed_dims).
            text_attention_mask (Tensor, optional): Text token mask. It has
                shape (bs,len_text).
            pos_text (Tensor, optional): The positional encoding for text.
                Defaults to None.
            text_self_attention_masks (Tensor, optional): Text self attention
                mask. Defaults to None.
            position_ids (Tensor, optional): Text position ids.
                Defaults to None.
        """
        output = query
        # reference_points = self.get_encoder_reference_points(
        #     spatial_shapes, valid_ratios, device=query.device)
        if self.text_layers:
            # generate pos_text
            bs, n_text, _ = memory_text.shape
            if pos_text is None and position_ids is None:
                pos_text = (
                    torch.arange(n_text,
                                 device=memory_text.device).float().unsqueeze(
                                     0).unsqueeze(-1).repeat(bs, 1, 1))
                pos_text = get_text_sine_pos_embed(
                    pos_text, num_pos_feats=256, exchange_xy=False)
            if position_ids is not None:
                pos_text = get_text_sine_pos_embed(
                    position_ids[..., None],
                    num_pos_feats=256,
                    exchange_xy=False)

        # main process
        P5 = output[:, level_start_index[-1]:]
        P4 = output[:, level_start_index[1]:level_start_index[-1]]
        P3 = output[:, :level_start_index[1]]
        query_pos_P5 = query_pos[:, level_start_index[-1]:]
        if key_padding_mask is not None:
            key_padding_mask_P5 = key_padding_mask[:, level_start_index[-1]:]
        else:
            key_padding_mask_P5 = None
        for layer_id, layer in enumerate(self.layers):
            # use p5 to calculate fusion model
            if self.fusion_layers:
                P5, memory_text = self.fusion_layers[layer_id](
                    visual_feature=P5,
                    lang_feature=memory_text,
                    attention_mask_v=key_padding_mask_P5,
                    attention_mask_l=text_attention_mask,
                )
            if self.text_layers:
                text_num_heads = self.text_layers[
                    layer_id].self_attn_cfg.num_heads
                memory_text = self.text_layers[layer_id](
                    query=memory_text,
                    query_pos=(pos_text if pos_text is not None else None),
                    attn_mask=~text_self_attention_masks.repeat(
                        text_num_heads, 1, 1),  # note we use ~ for mask here
                    key_padding_mask=None,
                )
            P5 = layer(
                query=P5,
                query_pos=query_pos_P5,
                key_padding_mask=key_padding_mask_P5)
        P5 = P5.permute(0,2,1).view(bs, self.embed_dims ,spatial_shapes[2][0],spatial_shapes[2][1]).contiguous()
        P4 = P4.permute(0,2,1).view(bs, self.embed_dims ,spatial_shapes[1][0],spatial_shapes[1][1]).contiguous()
        P3 = P3.permute(0,2,1).view(bs, self.embed_dims ,spatial_shapes[0][0],spatial_shapes[0][1]).contiguous()

            # with torch.no_grad():
                # inputs = torch.randn(1, 3, 1024, 1024).to(self.device)
                # macs = profile_macs(self.cross_scale_fusion[layer_id], [P3, P4, P5])
                # flops = FlopCountAnalysis(self.cross_scale_fusion[layer_id], [P3, P4, P5])
            # print(f"torchprofile: {macs / 1e9} GFLOPs")
            # print(f"fvcore: {flops.total() / 1e9} GFLOPs")
        P3, P4, P5 = self.cross_scale_fusion([P3, P4, P5])
        P3 = P3.view(bs, self.embed_dims, -1).contiguous()
        P4 = P4.view(bs, self.embed_dims, -1).contiguous()
        P5 = P5.view(bs, self.embed_dims, -1).contiguous()
        output = torch.concat((P3, P4, P5), dim=-1).permute(0, 2, 1).contiguous()
        # output = torch.concat((P3, P4, P5), dim=1).contiguous()
        return output, memory_text

class GroundingDinoDecoupleTransformerEncoder(DeformableDetrTransformerEncoder):

    def __init__(self, text_layer_cfg: ConfigType,
                 fusion_layer_cfg: ConfigType, **kwargs) -> None:
        self.text_layer_cfg = text_layer_cfg
        self.fusion_layer_cfg = fusion_layer_cfg
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList([
            DeformableDetrTransformerEncoderLayer(**self.layer_cfg)
            # DeformableDetrDecoupleTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        if self.text_layer_cfg is not None:
            self.text_layers = ModuleList([
                DetrTransformerEncoderLayer(**self.text_layer_cfg)
                for _ in range(self.num_layers)
            ])
        else:
            self.text_layers = None
        self.fusion_layers = ModuleList([
            TransformerDecoupleFusionLayer(**self.fusion_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        # if self.num_cp > 0:
        #     if checkpoint_wrapper is None:
        #         raise NotImplementedError(
        #             'If you want to reduce GPU memory usage, \
        #             please install fairscale by executing the \
        #             following command: pip install fairscale.')
        #     for i in range(self.num_cp):
        #         self.layers[i] = checkpoint_wrapper(self.layers[i])
        #         self.fusion_layers[i] = checkpoint_wrapper(
        #             self.fusion_layers[i])

    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                key_padding_mask: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                memory_text: Tensor = None,
                text_attention_mask: Tensor = None,
                pos_text: Tensor = None,
                text_self_attention_masks: Tensor = None,
                position_ids: Tensor = None):
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
            memory_text (Tensor, optional): Memory text. It has shape (bs,
                len_text, text_embed_dims).
            text_attention_mask (Tensor, optional): Text token mask. It has
                shape (bs,len_text).
            pos_text (Tensor, optional): The positional encoding for text.
                Defaults to None.
            text_self_attention_masks (Tensor, optional): Text self attention
                mask. Defaults to None.
            position_ids (Tensor, optional): Text position ids.
                Defaults to None.
        """
        output = query
        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=query.device)
        if self.text_layers:
            # generate pos_text
            bs, n_text, _ = memory_text.shape
            if pos_text is None and position_ids is None:
                pos_text = (
                    torch.arange(n_text,
                                 device=memory_text.device).float().unsqueeze(
                                     0).unsqueeze(-1).repeat(bs, 1, 1))
                pos_text = get_text_sine_pos_embed(
                    pos_text, num_pos_feats=256, exchange_xy=False)
            if position_ids is not None:
                pos_text = get_text_sine_pos_embed(
                    position_ids[..., None],
                    num_pos_feats=256,
                    exchange_xy=False)

        # main process
        for layer_id, layer in enumerate(self.layers):
            if self.fusion_layers:
                output = self.fusion_layers[layer_id](
                    output=output,
                    memory_text=memory_text,
                    query_pos=query_pos,
                    pos_text=(pos_text if pos_text is not None else None),
                    attn_mask=None,
                    text_attention_mask=text_attention_mask,
                    spatial_shapes=spatial_shapes
                )
            if self.text_layers:
                text_num_heads = self.text_layers[
                    layer_id].self_attn_cfg.num_heads
                memory_text = self.text_layers[layer_id](
                    query=memory_text,
                    query_pos=(pos_text if pos_text is not None else None),
                    attn_mask=~text_self_attention_masks.repeat(
                        text_num_heads, 1, 1),  # note we use ~ for mask here
                    key_padding_mask=None,
                )
            output = layer(
                query=output,
                query_pos=query_pos,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=key_padding_mask)
        return output, memory_text

class GroundingDinoDecoupleGuideTransformerEncoder(DeformableDetrTransformerEncoder):

    def __init__(self, text_layer_cfg: ConfigType,
                 fusion_layer_cfg: ConfigType,
                 guide_layer_cfg: ConfigType,
                 num_guide_layers, **kwargs) -> None:
        self.text_layer_cfg = text_layer_cfg
        self.fusion_layer_cfg = fusion_layer_cfg
        self.guide_layer_cfg = guide_layer_cfg
        self.num_guide_layers = num_guide_layers
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList([
            DeformableDetrTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        if self.text_layer_cfg is not None:
            self.text_layers = ModuleList([
                DetrTransformerEncoderLayer(**self.text_layer_cfg)
                for _ in range(self.num_layers)
            ])
        else:
            self.text_layers = None
        self.fusion_layers = ModuleList([
            TransformerDecoupleFusionLayer(**self.fusion_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.guide_layers = ModuleList([GuideFusionLayer(**self.guide_layer_cfg)
                                        for _ in range(self.num_guide_layers)])
        self.embed_dims = self.layers[0].embed_dims
        if self.num_cp > 0:
            if checkpoint_wrapper is None:
                raise NotImplementedError(
                    'If you want to reduce GPU memory usage, \
                    please install fairscale by executing the \
                    following command: pip install fairscale.')
            for i in range(self.num_cp):
                self.layers[i] = checkpoint_wrapper(self.layers[i])
                self.fusion_layers[i] = checkpoint_wrapper(
                    self.fusion_layers[i])

    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                key_padding_mask: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                memory_text: Tensor = None,
                text_attention_mask: Tensor = None,
                pos_text: Tensor = None,
                text_self_attention_masks: Tensor = None,
                position_ids: Tensor = None):
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
            memory_text (Tensor, optional): Memory text. It has shape (bs,
                len_text, text_embed_dims).
            text_attention_mask (Tensor, optional): Text token mask. It has
                shape (bs,len_text).
            pos_text (Tensor, optional): The positional encoding for text.
                Defaults to None.
            text_self_attention_masks (Tensor, optional): Text self attention
                mask. Defaults to None.
            position_ids (Tensor, optional): Text position ids.
                Defaults to None.
        """
        output = query
        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=query.device)
        if self.text_layers:
            # generate pos_text
            bs, n_text, _ = memory_text.shape
            if pos_text is None and position_ids is None:
                pos_text = (
                    torch.arange(n_text,
                                 device=memory_text.device).float().unsqueeze(
                                     0).unsqueeze(-1).repeat(bs, 1, 1))
                pos_text = get_text_sine_pos_embed(
                    pos_text, num_pos_feats=256, exchange_xy=False)
            if position_ids is not None:
                pos_text = get_text_sine_pos_embed(
                    position_ids[..., None],
                    num_pos_feats=256,
                    exchange_xy=False)

        # main process
        high_feats = output[:, level_start_index[1]:]
        low_feats = output[:, :level_start_index[1]]
        spatial_shapes_high = spatial_shapes[1:]
        reference_points_high = reference_points[:, level_start_index[1]:, 1:]
        query_pos_high = query_pos[:, level_start_index[1]:]
        level_start_index_high = torch.cat((
            spatial_shapes_high.new_zeros((1, )),  # (num_level)
            spatial_shapes_high.prod(1).cumsum(0)[:-1]))
        if key_padding_mask is not None:
            key_padding_mask_high = key_padding_mask[:, level_start_index[1]:]
        else:
            key_padding_mask_high = None

        for layer_id, layer in enumerate(self.layers):
            if self.fusion_layers:
                high_feats = self.fusion_layers[layer_id](
                    output=high_feats,
                    memory_text=memory_text,
                    query_pos=query_pos_high,
                    pos_text=(pos_text if pos_text is not None else None),
                    attn_mask=None,
                    text_attention_mask=text_attention_mask,
                    spatial_shapes=spatial_shapes_high
                )
            if self.text_layers:
                text_num_heads = self.text_layers[
                    layer_id].self_attn_cfg.num_heads
                memory_text = self.text_layers[layer_id](
                    query=memory_text,
                    query_pos=(pos_text if pos_text is not None else None),
                    attn_mask=~text_self_attention_masks.repeat(
                        text_num_heads, 1, 1),  # note we use ~ for mask here
                    key_padding_mask=None,
                )
            high_feats = layer(
                query=high_feats,
                query_pos=query_pos_high,
                reference_points=reference_points_high,
                spatial_shapes=spatial_shapes_high,
                level_start_index=level_start_index_high,
                key_padding_mask=key_padding_mask_high)


        for layer in self.guide_layers:
            low_feats = layer(
                low_feats=low_feats,
                spatial_shapes_low=spatial_shapes[:1],
                high_feats=high_feats,
                spatial_shapes_high=spatial_shapes_high,
                level_start_index_high=level_start_index_high)

        output = torch.cat([low_feats, high_feats], dim=1)

        return output, memory_text

class GroundingDinoCascadeTransformerEncoder(DeformableDetrTransformerEncoder):

    def __init__(self, text_layer_cfg: ConfigType,
                 fusion_layer_cfg: ConfigType,
                 low_layer_num,
                 fusion_layer_num,
                 guide=False,
                 return_intermediate=False,
                 **kwargs) -> None:
        self.text_layer_cfg = text_layer_cfg
        self.fusion_layer_cfg = fusion_layer_cfg
        self.low_layer_num = low_layer_num
        self.fusion_layer_num = fusion_layer_num
        self.guide = guide
        self.return_intermediate = return_intermediate
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = [
            DeformableDetrSparseTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers - self.low_layer_num)
        ]
        for n in range(self.low_layer_num):
            self.layers.append(DeformableDetrTransformerEncoderLayer(**self.layer_cfg))
        self.layers = ModuleList(self.layers)
        self.text_layers = ModuleList([
            DetrTransformerEncoderLayer(**self.text_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.fusion_layers = [
            TransformerCascadeFusionLayer(**self.fusion_layer_cfg)
            for _ in range(self.fusion_layer_num)
        ]
        self.fusion_layers = ModuleList(self.fusion_layers)
        self.embed_dims = self.layers[0].embed_dims
        if self.num_cp > 0:
            if checkpoint_wrapper is None:
                raise NotImplementedError(
                    'If you want to reduce GPU memory usage, \
                    please install fairscale by executing the \
                    following command: pip install fairscale.')
            for i in range(self.num_cp):
                self.layers[i] = checkpoint_wrapper(self.layers[i])
            for i in range(self.fusion_layer_num):
                self.fusion_layers[i] = checkpoint_wrapper(
                    self.fusion_layers[i])

    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                key_padding_mask: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                memory_text: Tensor = None,
                text_attention_mask: Tensor = None,
                pos_text: Tensor = None,
                text_self_attention_masks: Tensor = None,
                position_ids: Tensor = None):
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
            memory_text (Tensor, optional): Memory text. It has shape (bs,
                len_text, text_embed_dims).
            text_attention_mask (Tensor, optional): Text token mask. It has
                shape (bs,len_text).
            pos_text (Tensor, optional): The positional encoding for text.
                Defaults to None.
            text_self_attention_masks (Tensor, optional): Text self attention
                mask. Defaults to None.
            position_ids (Tensor, optional): Text position ids.
                Defaults to None.
        """
        output = query
        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=query.device)
        if self.text_layers:
            # generate pos_text
            bs, n_text, _ = memory_text.shape
            if pos_text is None and position_ids is None:
                pos_text = (
                    torch.arange(n_text,
                                 device=memory_text.device).float().unsqueeze(
                                     0).unsqueeze(-1).repeat(bs, 1, 1))
                pos_text = get_text_sine_pos_embed(
                    pos_text, num_pos_feats=256, exchange_xy=False)
            if position_ids is not None:
                pos_text = get_text_sine_pos_embed(
                    position_ids[..., None],
                    num_pos_feats=256,
                    exchange_xy=False)

        # main process
        for i in range(self.fusion_layer_num):
            output = self.fusion_layers[i](
                output=output,
                memory_text=memory_text,
                query_pos=query_pos,
                pos_text=(pos_text if pos_text is not None else None),
                attn_mask=None,
                text_attention_mask=text_attention_mask,
                spatial_shapes=spatial_shapes
            )

        intermediate_memory = []
        intermediate_text = []
        for layer_id, layer in enumerate(self.layers):
            if self.text_layers:
                text_num_heads = self.text_layers[
                    layer_id].self_attn_cfg.num_heads
                memory_text = self.text_layers[layer_id](
                    query=memory_text,
                    query_pos=(pos_text if pos_text is not None else None),
                    attn_mask=~text_self_attention_masks.repeat(
                        text_num_heads, 1, 1),  # note we use ~ for mask here
                    key_padding_mask=None,
                )

            if layer_id >= self.num_layers - self.low_layer_num:
                output = layer(
                    query=output,
                    query_pos=query_pos,
                    reference_points=reference_points,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    key_padding_mask=key_padding_mask)
            else:
                output_lite = output[:, level_start_index[1]:, :]
                output_res = output[:, :level_start_index[1], :]
                reference_points_lite = reference_points[:, level_start_index[1]:, ...]
                query_pos_lite = query_pos[:, level_start_index[1]:, :]
                spatial_shapes_lite = spatial_shapes[1:]
                # if key_padding_mask is not None:
                #     key_padding_mask_lite = key_padding_mask[:, level_start_index[1]:]
                # else:
                #     key_padding_mask_lite = None
                output_lite = layer(
                    query=output_lite,
                    query_pos=query_pos_lite,
                    memory=output,
                    reference_points=reference_points_lite,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    key_padding_mask=key_padding_mask)

                if self.guide:
                    H2, W2 = spatial_shapes_lite[0]
                    H1, W1 = spatial_shapes[0]
                    l2_ind = (H2 * W2).item()
                    bs, _, dim = output_res.shape
                    output_res = output_res + F.interpolate(output_lite[:,:l2_ind,:].reshape(bs, H2, W2, dim).permute(0,-1, 1, 2),
                                           size=(H1, W1), mode='nearest').reshape(bs,dim,
                                                                                   output_res.shape[1]).permute(0,2,1).contiguous()
                output = torch.cat((output_res, output_lite), dim=1)

            if self.return_intermediate:
                intermediate_memory.append(output)
                intermediate_text.append(memory_text)
        if self.return_intermediate:
            return intermediate_memory, intermediate_text
        else:
            return output, memory_text

class GroundingDinoOneFusionTransformerEncoder(DeformableDetrTransformerEncoder):

    def __init__(self, text_layer_cfg: ConfigType,
                 fusion_layer_cfg: ConfigType,
                 fusion_layer_num,
                 **kwargs) -> None:
        self.text_layer_cfg = text_layer_cfg
        self.fusion_layer_cfg = fusion_layer_cfg
        self.fusion_layer_num = fusion_layer_num
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList([
            DeformableDetrTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.text_layers = ModuleList([
            DetrTransformerEncoderLayer(**self.text_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.fusion_layers = [
            SingleScaleBiAttentionBlock(**self.fusion_layer_cfg)
            for _ in range(self.fusion_layer_num)
        ]
        self.fusion_layers = ModuleList(self.fusion_layers)
        self.embed_dims = self.layers[0].embed_dims
        if self.num_cp > 0:
            if checkpoint_wrapper is None:
                raise NotImplementedError(
                    'If you want to reduce GPU memory usage, \
                    please install fairscale by executing the \
                    following command: pip install fairscale.')
            for i in range(self.num_cp):
                self.layers[i] = checkpoint_wrapper(self.layers[i])
            for i in range(self.fusion_layer_num):
                self.fusion_layers[i] = checkpoint_wrapper(
                    self.fusion_layers[i])

    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                key_padding_mask: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                memory_text: Tensor = None,
                text_attention_mask: Tensor = None,
                pos_text: Tensor = None,
                text_self_attention_masks: Tensor = None,
                position_ids: Tensor = None):
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
            memory_text (Tensor, optional): Memory text. It has shape (bs,
                len_text, text_embed_dims).
            text_attention_mask (Tensor, optional): Text token mask. It has
                shape (bs,len_text).
            pos_text (Tensor, optional): The positional encoding for text.
                Defaults to None.
            text_self_attention_masks (Tensor, optional): Text self attention
                mask. Defaults to None.
            position_ids (Tensor, optional): Text position ids.
                Defaults to None.
        """
        output = query
        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=query.device)
        if self.text_layers:
            # generate pos_text
            bs, n_text, _ = memory_text.shape
            if pos_text is None and position_ids is None:
                pos_text = (
                    torch.arange(n_text,
                                 device=memory_text.device).float().unsqueeze(
                                     0).unsqueeze(-1).repeat(bs, 1, 1))
                pos_text = get_text_sine_pos_embed(
                    pos_text, num_pos_feats=256, exchange_xy=False)
            if position_ids is not None:
                pos_text = get_text_sine_pos_embed(
                    position_ids[..., None],
                    num_pos_feats=256,
                    exchange_xy=False)

        # main process
        for i in range(self.fusion_layer_num):
            if self.fusion_layers:
                output, memory_text = self.fusion_layers[i](
                    visual_feature=output,
                    lang_feature=memory_text,
                    attention_mask_v=key_padding_mask,
                    attention_mask_l=text_attention_mask,
                )

        for layer_id, layer in enumerate(self.layers):
            if self.text_layers:
                text_num_heads = self.text_layers[
                    layer_id].self_attn_cfg.num_heads
                memory_text = self.text_layers[layer_id](
                    query=memory_text,
                    query_pos=(pos_text if pos_text is not None else None),
                    attn_mask=~text_self_attention_masks.repeat(
                        text_num_heads, 1, 1),  # note we use ~ for mask here
                    key_padding_mask=None,
                )
            output = layer(
                query=output,
                query_pos=query_pos,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=key_padding_mask)
        return output, memory_text

class GroundingDinoInvertDecoupleTransformerEncoder(DeformableDetrTransformerEncoder):

    def __init__(self, text_layer_cfg: ConfigType,
                 fusion_layer_cfg: ConfigType,
                 low_layer_num,
                 guide=False,
                 **kwargs) -> None:
        self.text_layer_cfg = text_layer_cfg
        self.fusion_layer_cfg = fusion_layer_cfg
        self.low_layer_num = low_layer_num
        self.guide = guide
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = [
            DeformableDetrSparseTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers - self.low_layer_num)
        ]
        for n in range(self.low_layer_num):
            self.layers.append(DeformableDetrTransformerEncoderLayer(**self.layer_cfg))
        self.layers = ModuleList(self.layers)
        self.text_layers = ModuleList([
            DetrTransformerEncoderLayer(**self.text_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.fusion_layers = [
            TransformerLiteDecoupleFusionLayer(**self.fusion_layer_cfg)
            for _ in range(self.num_layers - self.low_layer_num)
        ]
        for n in range(self.low_layer_num):
            self.fusion_layer_cfg['feat_attn_layer_num'] = [1, 1, 1, 1]
            self.fusion_layers.append(TransformerLiteDecoupleFusionLayer(**self.fusion_layer_cfg))
        self.fusion_layers = ModuleList(self.fusion_layers)
        self.embed_dims = self.layers[0].embed_dims
        # if self.num_cp > 0:
        #     if checkpoint_wrapper is None:
        #         raise NotImplementedError(
        #             'If you want to reduce GPU memory usage, \
        #             please install fairscale by executing the \
        #             following command: pip install fairscale.')
        #     for i in range(self.num_cp):
        #         self.layers[i] = checkpoint_wrapper(self.layers[i])
        #         self.fusion_layers[i] = checkpoint_wrapper(
        #             self.fusion_layers[i])

    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                key_padding_mask: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                memory_text: Tensor = None,
                text_attention_mask: Tensor = None,
                pos_text: Tensor = None,
                text_self_attention_masks: Tensor = None,
                position_ids: Tensor = None):
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
            memory_text (Tensor, optional): Memory text. It has shape (bs,
                len_text, text_embed_dims).
            text_attention_mask (Tensor, optional): Text token mask. It has
                shape (bs,len_text).
            pos_text (Tensor, optional): The positional encoding for text.
                Defaults to None.
            text_self_attention_masks (Tensor, optional): Text self attention
                mask. Defaults to None.
            position_ids (Tensor, optional): Text position ids.
                Defaults to None.
        """
        output = query
        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=query.device)
        if self.text_layers:
            # generate pos_text
            bs, n_text, _ = memory_text.shape
            if pos_text is None and position_ids is None:
                pos_text = (
                    torch.arange(n_text,
                                 device=memory_text.device).float().unsqueeze(
                                     0).unsqueeze(-1).repeat(bs, 1, 1))
                pos_text = get_text_sine_pos_embed(
                    pos_text, num_pos_feats=256, exchange_xy=False)
            if position_ids is not None:
                pos_text = get_text_sine_pos_embed(
                    position_ids[..., None],
                    num_pos_feats=256,
                    exchange_xy=False)

        # main process
        for layer_id, layer in enumerate(self.layers):
            if layer_id >= self.num_layers - self.low_layer_num:
                if self.fusion_layers:
                    output = self.fusion_layers[layer_id](
                        output=output,
                        memory_text=memory_text,
                        query_pos=query_pos,
                        pos_text=(pos_text if pos_text is not None else None),
                        attn_mask=None,
                        text_attention_mask=text_attention_mask,
                        spatial_shapes=spatial_shapes
                    )
            else:
                output_lite = output[:, level_start_index[1]:, :]
                output_res = output[:, :level_start_index[1], :]
                reference_points_lite = reference_points[:, level_start_index[1]:, ...]
                query_pos_lite = query_pos[:, level_start_index[1]:, :]
                spatial_shapes_lite = spatial_shapes[1:]
                # if key_padding_mask is not None:
                #     key_padding_mask_lite = key_padding_mask[:, level_start_index[1]:]
                # else:
                #     key_padding_mask_lite = None

                if self.fusion_layers:
                    output_lite = self.fusion_layers[layer_id](
                        output=output_lite,
                        memory_text=memory_text,
                        query_pos=query_pos_lite,
                        pos_text=(pos_text if pos_text is not None else None),
                        attn_mask=None,
                        text_attention_mask=text_attention_mask,
                        spatial_shapes=spatial_shapes_lite
                    )
            if self.text_layers:
                text_num_heads = self.text_layers[
                    layer_id].self_attn_cfg.num_heads
                memory_text = self.text_layers[layer_id](
                    query=memory_text,
                    query_pos=(pos_text if pos_text is not None else None),
                    attn_mask=~text_self_attention_masks.repeat(
                        text_num_heads, 1, 1),  # note we use ~ for mask here
                    key_padding_mask=None,
                )
            if layer_id >= self.num_layers - self.low_layer_num:
                output = layer(
                    query=output,
                    query_pos=query_pos,
                    reference_points=reference_points,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    key_padding_mask=key_padding_mask)
            else:
                if self.guide:
                    H2, W2 = spatial_shapes_lite[0]
                    H1, W1 = spatial_shapes[0]
                    l2_ind = (H2 * W2).item()
                    bs, _, dim = output_res.shape
                    output_res = output_res + F.interpolate(output_lite[:,:l2_ind,:].reshape(bs, H2, W2, dim).permute(0,-1, 1, 2),
                                           size=(H1, W1), mode='nearest').reshape(bs,dim,
                                                                                   output_res.shape[1]).permute(0,2,1).contiguous()
                output = torch.cat((output_res, output_lite), dim=1)
                output_lite = layer(
                    query=output_lite,
                    query_pos=query_pos_lite,
                    memory=output,
                    reference_points=reference_points_lite,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    key_padding_mask=key_padding_mask)

                if self.guide:
                    H2, W2 = spatial_shapes_lite[0]
                    H1, W1 = spatial_shapes[0]
                    l2_ind = (H2 * W2).item()
                    bs, _, dim = output_res.shape
                    output_res = output_res + F.interpolate(output_lite[:,:l2_ind,:].reshape(bs, H2, W2, dim).permute(0,-1, 1, 2),
                                           size=(H1, W1), mode='nearest').reshape(bs,dim,
                                                                                   output_res.shape[1]).permute(0,2,1).contiguous()
                output = torch.cat((output_res, output_lite), dim=1)
        return output, memory_text

class GroundingDinoInvertTransformerEncoder(DeformableDetrTransformerEncoder):

    def __init__(self, text_layer_cfg: ConfigType,
                 fusion_layer_cfg: ConfigType,
                 low_layer_num,
                 guide=False,
                 **kwargs) -> None:
        self.text_layer_cfg = text_layer_cfg
        self.fusion_layer_cfg = fusion_layer_cfg
        self.low_layer_num = low_layer_num
        self.guide = guide
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = [
            DeformableDetrSparseTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers - self.low_layer_num)
        ]
        for n in range(self.low_layer_num):
            self.layers.append(DeformableDetrTransformerEncoderLayer(**self.layer_cfg))
        self.layers = ModuleList(self.layers)
        self.text_layers = ModuleList([
            DetrTransformerEncoderLayer(**self.text_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.fusion_layers = ModuleList([
            SingleScaleBiAttentionBlock(**self.fusion_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.num_cp > 0:
            if checkpoint_wrapper is None:
                raise NotImplementedError(
                    'If you want to reduce GPU memory usage, \
                    please install fairscale by executing the \
                    following command: pip install fairscale.')
            for i in range(self.num_cp):
                self.layers[i] = checkpoint_wrapper(self.layers[i])
                self.fusion_layers[i] = checkpoint_wrapper(
                    self.fusion_layers[i])

    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                key_padding_mask: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                memory_text: Tensor = None,
                text_attention_mask: Tensor = None,
                pos_text: Tensor = None,
                text_self_attention_masks: Tensor = None,
                position_ids: Tensor = None):
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
            memory_text (Tensor, optional): Memory text. It has shape (bs,
                len_text, text_embed_dims).
            text_attention_mask (Tensor, optional): Text token mask. It has
                shape (bs,len_text).
            pos_text (Tensor, optional): The positional encoding for text.
                Defaults to None.
            text_self_attention_masks (Tensor, optional): Text self attention
                mask. Defaults to None.
            position_ids (Tensor, optional): Text position ids.
                Defaults to None.
        """
        output = query
        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=query.device)
        if self.text_layers:
            # generate pos_text
            bs, n_text, _ = memory_text.shape
            if pos_text is None and position_ids is None:
                pos_text = (
                    torch.arange(n_text,
                                 device=memory_text.device).float().unsqueeze(
                                     0).unsqueeze(-1).repeat(bs, 1, 1))
                pos_text = get_text_sine_pos_embed(
                    pos_text, num_pos_feats=256, exchange_xy=False)
            if position_ids is not None:
                pos_text = get_text_sine_pos_embed(
                    position_ids[..., None],
                    num_pos_feats=256,
                    exchange_xy=False)

        # main process
        for layer_id, layer in enumerate(self.layers):
            if self.fusion_layers:
                output, memory_text = self.fusion_layers[layer_id](
                    visual_feature=output,
                    lang_feature=memory_text,
                    attention_mask_v=key_padding_mask,
                    attention_mask_l=text_attention_mask,
                )
            # else:
            #     output_lite = output[:, level_start_index[1]:, :]
            #     output_res = output[:, :level_start_index[1], :]
            #     reference_points_lite = reference_points[:, level_start_index[1]:, ...]
            #     query_pos_lite = query_pos[:, level_start_index[1]:, :]
            #     spatial_shapes_lite = spatial_shapes[1:]
            #     if key_padding_mask is not None:
            #         key_padding_mask_lite = key_padding_mask[:, level_start_index[1]:]
            #     else:
            #         key_padding_mask_lite = None
            #
            #     if self.fusion_layers:
            #         output_lite, memory_text = self.fusion_layers[layer_id](
            #             visual_feature=output_lite,
            #             lang_feature=memory_text,
            #             attention_mask_v=key_padding_mask_lite,
            #             attention_mask_l=text_attention_mask,
            #         )
            if self.text_layers:
                text_num_heads = self.text_layers[
                    layer_id].self_attn_cfg.num_heads
                memory_text = self.text_layers[layer_id](
                    query=memory_text,
                    query_pos=(pos_text if pos_text is not None else None),
                    attn_mask=~text_self_attention_masks.repeat(
                        text_num_heads, 1, 1),  # note we use ~ for mask here
                    key_padding_mask=None,
                )
            if layer_id in [4, 5]:
                output = layer(
                    query=output,
                    query_pos=query_pos,
                    reference_points=reference_points,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    key_padding_mask=key_padding_mask)
            else:
                if layer_id in [0, 1]:
                    output_lite = output[:, level_start_index[2]:, :]
                    output_res = output[:, :level_start_index[2], :]
                    reference_points_lite = reference_points[:, level_start_index[2]:, ...]
                    query_pos_lite = query_pos[:, level_start_index[2]:, :]
                    spatial_shapes_lite = spatial_shapes[2:]
                    # if key_padding_mask is not None:
                    #     key_padding_mask_lite = key_padding_mask[:, level_start_index[1]:]
                    # else:
                    #     key_padding_mask_lite = None
                elif layer_id in [2, 3]:
                    output_lite = output[:, level_start_index[1]:, :]
                    output_res = output[:, :level_start_index[1], :]
                    reference_points_lite = reference_points[:, level_start_index[1]:, ...]
                    query_pos_lite = query_pos[:, level_start_index[1]:, :]
                    spatial_shapes_lite = spatial_shapes[1:]
                    # if key_padding_mask is not None:
                    #     key_padding_mask_lite = key_padding_mask[:, level_start_index[1]:]
                    # else:
                    #     key_padding_mask_lite = None

                output_lite = layer(
                    query=output_lite,
                    query_pos=query_pos_lite,
                    memory=output,
                    reference_points=reference_points_lite,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    key_padding_mask=key_padding_mask)

                if self.guide:
                    if layer_id in [2, 3]:
                        H2, W2 = spatial_shapes_lite[0]
                        H1, W1 = spatial_shapes[0]
                        l2_ind = (H2 * W2).item()
                        bs, _, dim = output_res.shape
                        output_res = output_res + F.interpolate(output_lite[:,:l2_ind,:].reshape(bs, H2, W2, dim).permute(0,-1, 1, 2),
                                               size=(H1, W1), mode='nearest').reshape(bs,dim,
                                                                                       output_res.shape[1]).permute(0,2,1).contiguous()
                    elif layer_id in [0, 1]:
                        H3, W3 = spatial_shapes_lite[0]
                        H1, W1 = spatial_shapes[0]
                        H2, W2 = spatial_shapes[1]
                        l3_ind = (H3 * W3).item()
                        bs, _, dim = output_res.shape
                        P3 = output_res[:, :H1 * W1, :]
                        P4 = output_res[:, H1 * W1:, :]
                        P4 = P4 + F.interpolate(output_lite[:,:l3_ind,:].reshape(bs, H3, W3, dim).permute(0,-1, 1, 2),
                                               size=(H2, W2), mode='nearest').reshape(bs,dim,
                                                                                       P4.shape[1]).permute(0,2,1).contiguous()
                        P3 = P3 + F.interpolate(P4.reshape(bs, H2, W2, dim).permute(0,-1, 1, 2),
                                               size=(H1, W1), mode='nearest').reshape(bs,dim,
                                                                                       P3.shape[1]).permute(0,2,1).contiguous()
                        output_res = torch.cat((P3, P4), dim=1)
                output = torch.cat((output_res, output_lite), dim=1)
        return output, memory_text

class GroundingDinoUFusionTransformerEncoder(DeformableDetrTransformerEncoder):

    def __init__(self, text_layer_cfg: ConfigType,
                 fusion_layer_cfg: ConfigType,
                 low_layer_num,
                 fusion_layer_ind,
                 guide=False,
                 # norm_sign=False,
                 # norm_cfg: OptConfigType = dict(type='LN'),
                 **kwargs) -> None:
        self.text_layer_cfg = text_layer_cfg
        self.fusion_layer_cfg = fusion_layer_cfg
        self.low_layer_num = low_layer_num
        self.fusion_layer_num = len(fusion_layer_ind)
        self.fusion_layer_ind = fusion_layer_ind
        self.guide = guide
        # self.norm_cfg = norm_cfg
        # self.norm_sign = norm_sign
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = [
            DeformableDetrSparseTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers - self.low_layer_num)
        ]
        for n in range(self.low_layer_num):
            self.layers.append(DeformableDetrTransformerEncoderLayer(**self.layer_cfg))
        self.layers = ModuleList(self.layers)
        self.text_layers = ModuleList([
            DetrTransformerEncoderLayer(**self.text_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.fusion_layers = [
            TransformerOnlyDecoupleFusionLayer(**self.fusion_layer_cfg)
            for _ in range(self.fusion_layer_num)
        ]
        self.fusion_layers = ModuleList(self.fusion_layers)
        self.embed_dims = self.layers[0].embed_dims
        # norms_list = [
        #     build_norm_layer(self.norm_cfg, self.embed_dims)[1]
        #     for _ in range(self.num_layers - 1)
        # ]
        # self.norms = ModuleList(norms_list)
        if self.num_cp > 0:
            if checkpoint_wrapper is None:
                raise NotImplementedError(
                    'If you want to reduce GPU memory usage, \
                    please install fairscale by executing the \
                    following command: pip install fairscale.')
            for i in range(self.num_cp):
                self.layers[i] = checkpoint_wrapper(self.layers[i])
            for i in range(self.fusion_layer_num):
                self.fusion_layers[i] = checkpoint_wrapper(
                    self.fusion_layers[i])

    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                key_padding_mask: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                memory_text: Tensor = None,
                text_attention_mask: Tensor = None,
                pos_text: Tensor = None,
                text_self_attention_masks: Tensor = None,
                position_ids: Tensor = None):
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
            memory_text (Tensor, optional): Memory text. It has shape (bs,
                len_text, text_embed_dims).
            text_attention_mask (Tensor, optional): Text token mask. It has
                shape (bs,len_text).
            pos_text (Tensor, optional): The positional encoding for text.
                Defaults to None.
            text_self_attention_masks (Tensor, optional): Text self attention
                mask. Defaults to None.
            position_ids (Tensor, optional): Text position ids.
                Defaults to None.
        """
        output = query
        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=query.device)
        if self.text_layers:
            # generate pos_text
            bs, n_text, _ = memory_text.shape
            if pos_text is None and position_ids is None:
                pos_text = (
                    torch.arange(n_text,
                                 device=memory_text.device).float().unsqueeze(
                                     0).unsqueeze(-1).repeat(bs, 1, 1))
                pos_text = get_text_sine_pos_embed(
                    pos_text, num_pos_feats=256, exchange_xy=False)
            if position_ids is not None:
                pos_text = get_text_sine_pos_embed(
                    position_ids[..., None],
                    num_pos_feats=256,
                    exchange_xy=False)

        # main process
        for layer_id, layer in enumerate(self.layers):
            if layer_id in self.fusion_layer_ind:
                if self.fusion_layers:
                    output, memory_text = self.fusion_layers[self.fusion_layer_ind.index(layer_id)](
                        output=output,
                        memory_text=memory_text,
                        query_pos=query_pos,
                        pos_text=(pos_text if pos_text is not None else None),
                        attn_mask=None,
                        text_attention_mask=text_attention_mask,
                        spatial_shapes=spatial_shapes,
                        key_padding_mask_image=key_padding_mask
                    )


            if self.text_layers:
                text_num_heads = self.text_layers[
                    layer_id].self_attn_cfg.num_heads
                memory_text = self.text_layers[layer_id](
                    query=memory_text,
                    query_pos=(pos_text if pos_text is not None else None),
                    attn_mask=~text_self_attention_masks.repeat(
                        text_num_heads, 1, 1),  # note we use ~ for mask here
                    key_padding_mask=None,
                )

            if layer_id in [4, 5]:
                output = layer(
                    query=output,
                    query_pos=query_pos,
                    reference_points=reference_points,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    key_padding_mask=key_padding_mask)
            else:
                if layer_id in [0, 1]:
                    output_lite = output[:, level_start_index[2]:, :]
                    output_res = output[:, :level_start_index[2], :]
                    reference_points_lite = reference_points[:, level_start_index[2]:, ...]
                    query_pos_lite = query_pos[:, level_start_index[2]:, :]
                    spatial_shapes_lite = spatial_shapes[2:]
                    # if key_padding_mask is not None:
                    #     key_padding_mask_lite = key_padding_mask[:, level_start_index[1]:]
                    # else:
                    #     key_padding_mask_lite = None
                elif layer_id in [2, 3]:
                    output_lite = output[:, level_start_index[1]:, :]
                    output_res = output[:, :level_start_index[1], :]
                    reference_points_lite = reference_points[:, level_start_index[1]:, ...]
                    query_pos_lite = query_pos[:, level_start_index[1]:, :]
                    spatial_shapes_lite = spatial_shapes[1:]
                    # if key_padding_mask is not None:
                    #     key_padding_mask_lite = key_padding_mask[:, level_start_index[1]:]
                    # else:
                    #     key_padding_mask_lite = None

                output_lite = layer(
                    query=output_lite,
                    query_pos=query_pos_lite,
                    memory=output,
                    reference_points=reference_points_lite,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    key_padding_mask=key_padding_mask)

                if self.guide:
                    if layer_id in [2, 3]:
                        H2, W2 = spatial_shapes_lite[0]
                        H1, W1 = spatial_shapes[0]
                        l2_ind = (H2 * W2).item()
                        bs, _, dim = output_res.shape
                        output_res = output_res + F.interpolate(output_lite[:,:l2_ind,:].reshape(bs, H2, W2, dim).permute(0,-1, 1, 2),
                                               size=(H1, W1), mode='nearest').reshape(bs,dim,
                                                                                       output_res.shape[1]).permute(0,2,1).contiguous()
                        # if self.norm_sign:
                        #     output_res = self.norms[layer_id](output_res)
                    elif layer_id in [0, 1]:
                        H3, W3 = spatial_shapes_lite[0]
                        H1, W1 = spatial_shapes[0]
                        H2, W2 = spatial_shapes[1]
                        l3_ind = (H3 * W3).item()
                        bs, _, dim = output_res.shape
                        P3 = output_res[:, :H1 * W1, :]
                        P4 = output_res[:, H1 * W1:, :]
                        P4 = P4 + F.interpolate(output_lite[:,:l3_ind,:].reshape(bs, H3, W3, dim).permute(0,-1, 1, 2),
                                               size=(H2, W2), mode='nearest').reshape(bs,dim,
                                                                                       P4.shape[1]).permute(0,2,1).contiguous()
                        P3 = P3 + F.interpolate(P4.reshape(bs, H2, W2, dim).permute(0,-1, 1, 2),
                                               size=(H1, W1), mode='nearest').reshape(bs,dim,
                                                                                       P3.shape[1]).permute(0,2,1).contiguous()
                        output_res = torch.cat((P3, P4), dim=1)
                        # if self.norm_sign:
                        #     output_res = self.norms[layer_id](output_res)
                output = torch.cat((output_res, output_lite), dim=1)
        return output, memory_text

class GroundingDinoUTransformerEncoder(DeformableDetrTransformerEncoder):

    def __init__(self, text_layer_cfg: ConfigType,
                 fusion_layer_cfg: ConfigType,
                 fusion_layer_ind,
                 guide=False,
                 **kwargs) -> None:
        self.text_layer_cfg = text_layer_cfg
        self.fusion_layer_cfg = fusion_layer_cfg
        self.fusion_layer_num = len(fusion_layer_ind)
        self.fusion_layer_ind = fusion_layer_ind
        self.guide = guide
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList([
            DeformableDetrTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.text_layers = ModuleList([
            DetrTransformerEncoderLayer(**self.text_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.fusion_layers = [
            SingleScaleBiAttentionBlock(**self.fusion_layer_cfg)
            for _ in range(self.fusion_layer_num)
        ]
        self.fusion_layers = ModuleList(self.fusion_layers)
        self.embed_dims = self.layers[0].embed_dims
        # if self.num_cp > 0:
        #     if checkpoint_wrapper is None:
        #         raise NotImplementedError(
        #             'If you want to reduce GPU memory usage, \
        #             please install fairscale by executing the \
        #             following command: pip install fairscale.')
        #     for i in range(self.num_cp):
        #         self.layers[i] = checkpoint_wrapper(self.layers[i])
        #     for i in range(self.fusion_layer_num):
        #         self.fusion_layers[i] = checkpoint_wrapper(
        #             self.fusion_layers[i])

    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                key_padding_mask: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                memory_text: Tensor = None,
                text_attention_mask: Tensor = None,
                pos_text: Tensor = None,
                text_self_attention_masks: Tensor = None,
                position_ids: Tensor = None):
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
            memory_text (Tensor, optional): Memory text. It has shape (bs,
                len_text, text_embed_dims).
            text_attention_mask (Tensor, optional): Text token mask. It has
                shape (bs,len_text).
            pos_text (Tensor, optional): The positional encoding for text.
                Defaults to None.
            text_self_attention_masks (Tensor, optional): Text self attention
                mask. Defaults to None.
            position_ids (Tensor, optional): Text position ids.
                Defaults to None.
        """
        output = query
        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=query.device)
        if self.text_layers:
            # generate pos_text
            bs, n_text, _ = memory_text.shape
            if pos_text is None and position_ids is None:
                pos_text = (
                    torch.arange(n_text,
                                 device=memory_text.device).float().unsqueeze(
                                     0).unsqueeze(-1).repeat(bs, 1, 1))
                pos_text = get_text_sine_pos_embed(
                    pos_text, num_pos_feats=256, exchange_xy=False)
            if position_ids is not None:
                pos_text = get_text_sine_pos_embed(
                    position_ids[..., None],
                    num_pos_feats=256,
                    exchange_xy=False)

        # main process
        for layer_id, layer in enumerate(self.layers):
            if layer_id in self.fusion_layer_ind:
                if self.fusion_layers:
                    output, memory_text = self.fusion_layers[self.fusion_layer_ind.index(layer_id)](
                        visual_feature=output,
                        lang_feature=memory_text,
                        attention_mask_v=key_padding_mask,
                        attention_mask_l=text_attention_mask,
                    )


            if self.text_layers:
                text_num_heads = self.text_layers[
                    layer_id].self_attn_cfg.num_heads
                memory_text = self.text_layers[layer_id](
                    query=memory_text,
                    query_pos=(pos_text if pos_text is not None else None),
                    attn_mask=~text_self_attention_masks.repeat(
                        text_num_heads, 1, 1),  # note we use ~ for mask here
                    key_padding_mask=None,
                )

            if layer_id in [0, 1, 2, 3, 4, 5]:
                output = layer(
                    query=output,
                    query_pos=query_pos,
                    reference_points=reference_points,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    key_padding_mask=key_padding_mask)
            else:
                if layer_id in [0, 1]:
                    output_lite = output[:, level_start_index[2]:, :]
                    output_res = output[:, :level_start_index[2], :]
                    reference_points_lite = reference_points[:, level_start_index[2]:, ...]
                    query_pos_lite = query_pos[:, level_start_index[2]:, :]
                    spatial_shapes_lite = spatial_shapes[2:]
                    # if key_padding_mask is not None:
                    #     key_padding_mask_lite = key_padding_mask[:, level_start_index[1]:]
                    # else:
                    #     key_padding_mask_lite = None
                elif layer_id in [2, 3, 4]:
                    output_lite = output[:, level_start_index[1]:, :]
                    output_res = output[:, :level_start_index[1], :]
                    reference_points_lite = reference_points[:, level_start_index[1]:, ...]
                    query_pos_lite = query_pos[:, level_start_index[1]:, :]
                    spatial_shapes_lite = spatial_shapes[1:]
                    # if key_padding_mask is not None:
                    #     key_padding_mask_lite = key_padding_mask[:, level_start_index[1]:]
                    # else:
                    #     key_padding_mask_lite = None

                output_lite = layer(
                    query=output_lite,
                    query_pos=query_pos_lite,
                    memory=output,
                    reference_points=reference_points_lite,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    key_padding_mask=key_padding_mask)

                if self.guide:
                    if layer_id in [2, 3, 4]:
                        H2, W2 = spatial_shapes_lite[0]
                        H1, W1 = spatial_shapes[0]
                        l2_ind = (H2 * W2).item()
                        bs, _, dim = output_res.shape
                        output_res = output_res + F.interpolate(output_lite[:,:l2_ind,:].reshape(bs, H2, W2, dim).permute(0,-1, 1, 2),
                                               size=(H1, W1), mode='nearest').reshape(bs,dim,
                                                                                       output_res.shape[1]).permute(0,2,1).contiguous()
                    elif layer_id in [0, 1]:
                        H3, W3 = spatial_shapes_lite[0]
                        H1, W1 = spatial_shapes[0]
                        H2, W2 = spatial_shapes[1]
                        l3_ind = (H3 * W3).item()
                        bs, _, dim = output_res.shape
                        P3 = output_res[:, :H1 * W1, :]
                        P4 = output_res[:, H1 * W1:, :]
                        P4 = P4 + F.interpolate(output_lite[:,:l3_ind,:].reshape(bs, H3, W3, dim).permute(0,-1, 1, 2),
                                               size=(H2, W2), mode='nearest').reshape(bs,dim,
                                                                                       P4.shape[1]).permute(0,2,1).contiguous()
                        P3 = P3 + F.interpolate(P4.reshape(bs, H2, W2, dim).permute(0,-1, 1, 2),
                                               size=(H1, W1), mode='nearest').reshape(bs,dim,
                                                                                       P3.shape[1]).permute(0,2,1).contiguous()
                        output_res = torch.cat((P3, P4), dim=1)
                output = torch.cat((output_res, output_lite), dim=1)
        return output, memory_text

class GroundingDinoUDecoupleTransformerEncoder(DeformableDetrTransformerEncoder):

    def __init__(self, text_layer_cfg: ConfigType,
                 fusion_layer_cfg: ConfigType,
                 fusion_layer_ind,
                 guide=False,
                 lite=False,
                 **kwargs) -> None:
        self.text_layer_cfg = text_layer_cfg
        self.fusion_layer_cfg = fusion_layer_cfg
        self.fusion_layer_num = len(fusion_layer_ind)
        self.fusion_layer_ind = fusion_layer_ind
        self.guide = guide
        self.lite = lite
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        if self.lite:
            self.layers = ModuleList([
                DeformableDetrSparseTransformerEncoderLayer(**self.layer_cfg)
                for _ in range(self.num_layers)
            ])
        else:
            self.layers = ModuleList([
                DeformableDetrTransformerEncoderLayer(**self.layer_cfg)
                for _ in range(self.num_layers)
            ])
        self.text_layers = ModuleList([
            DetrTransformerEncoderLayer(**self.text_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.fusion_layers = [
            TransformerOnlyDecoupleFusionLayer(**self.fusion_layer_cfg)
            for _ in range(self.fusion_layer_num)
        ]
        self.fusion_layers = ModuleList(self.fusion_layers)
        self.embed_dims = self.layers[0].embed_dims
        if self.num_cp > 0:
            if checkpoint_wrapper is None:
                raise NotImplementedError(
                    'If you want to reduce GPU memory usage, \
                    please install fairscale by executing the \
                    following command: pip install fairscale.')
            for i in range(self.num_cp):
                self.layers[i] = checkpoint_wrapper(self.layers[i])
            for i in range(self.fusion_layer_num):
                self.fusion_layers[i] = checkpoint_wrapper(
                    self.fusion_layers[i])

    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                key_padding_mask: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                memory_text: Tensor = None,
                text_attention_mask: Tensor = None,
                pos_text: Tensor = None,
                text_self_attention_masks: Tensor = None,
                position_ids: Tensor = None):
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
            memory_text (Tensor, optional): Memory text. It has shape (bs,
                len_text, text_embed_dims).
            text_attention_mask (Tensor, optional): Text token mask. It has
                shape (bs,len_text).
            pos_text (Tensor, optional): The positional encoding for text.
                Defaults to None.
            text_self_attention_masks (Tensor, optional): Text self attention
                mask. Defaults to None.
            position_ids (Tensor, optional): Text position ids.
                Defaults to None.
        """
        output = query
        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=query.device)
        if self.text_layers:
            # generate pos_text
            bs, n_text, _ = memory_text.shape
            if pos_text is None and position_ids is None:
                pos_text = (
                    torch.arange(n_text,
                                 device=memory_text.device).float().unsqueeze(
                                     0).unsqueeze(-1).repeat(bs, 1, 1))
                pos_text = get_text_sine_pos_embed(
                    pos_text, num_pos_feats=256, exchange_xy=False)
            if position_ids is not None:
                pos_text = get_text_sine_pos_embed(
                    position_ids[..., None],
                    num_pos_feats=256,
                    exchange_xy=False)

        # main process
        for layer_id, layer in enumerate(self.layers):
            if layer_id in self.fusion_layer_ind:
                if self.fusion_layers:
                    output, memory_text = self.fusion_layers[self.fusion_layer_ind.index(layer_id)](
                        output=output,
                        memory_text=memory_text,
                        query_pos=query_pos,
                        pos_text=(pos_text if pos_text is not None else None),
                        attn_mask=None,
                        text_attention_mask=text_attention_mask,
                        spatial_shapes=spatial_shapes,
                        key_padding_mask_image=key_padding_mask
                    )

            if self.text_layers:
                text_num_heads = self.text_layers[
                    layer_id].self_attn_cfg.num_heads
                memory_text = self.text_layers[layer_id](
                    query=memory_text,
                    query_pos=(pos_text if pos_text is not None else None),
                    attn_mask=~text_self_attention_masks.repeat(
                        text_num_heads, 1, 1),  # note we use ~ for mask here
                    key_padding_mask=None,
                )
            if self.lite:
                if layer_id in [2, 5]:
                    output_lite = output[:, :level_start_index[1], :]
                    output_res = output[:, level_start_index[1]:, :]
                    reference_points_lite = reference_points[:, :level_start_index[1], ...]
                    query_pos_lite = query_pos[:, :level_start_index[1], :]
                    if key_padding_mask is not None:
                        key_padding_mask_lite = key_padding_mask[:, :level_start_index[1]]
                    else:
                        key_padding_mask_lite = None
                elif layer_id in [0, 1, 3, 4]:
                    output_lite = output[:, level_start_index[1]:, :]
                    output_res = output[:, :level_start_index[1], :]
                    reference_points_lite = reference_points[:, level_start_index[1]:, ...]
                    query_pos_lite = query_pos[:, level_start_index[1]:, :]
                    spatial_shapes_lite = spatial_shapes[1:]
                    # if key_padding_mask is not None:
                    #     key_padding_mask_lite = key_padding_mask[:, level_start_index[1]:]
                    # else:
                    #     key_padding_mask_lite = None
                output_lite = layer(
                    query=output_lite,
                    query_pos=query_pos_lite,
                    memory=output,
                    reference_points=reference_points_lite,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    key_padding_mask=key_padding_mask)
                if layer_id in [2, 5]:
                    output = torch.cat((output_lite, output_res), dim=1)
                else:
                    output = torch.cat((output_res, output_lite), dim=1)

            else:
                if layer_id in [0, 1, 2, 3, 4, 5]:
                    output = layer(
                        query=output,
                        query_pos=query_pos,
                        reference_points=reference_points,
                        spatial_shapes=spatial_shapes,
                        level_start_index=level_start_index,
                        key_padding_mask=key_padding_mask)
                elif layer_id in [0, 1, 3, 4]:
                    if layer_id in [0, 1]:
                        output_lite = output[:, level_start_index[2]:, :]
                        output_res = output[:, :level_start_index[2], :]
                        reference_points_lite = reference_points[:, level_start_index[2]:, ...]
                        query_pos_lite = query_pos[:, level_start_index[2]:, :]
                        spatial_shapes_lite = spatial_shapes[2:]
                        # if key_padding_mask is not None:
                        #     key_padding_mask_lite = key_padding_mask[:, level_start_index[1]:]
                        # else:
                        #     key_padding_mask_lite = None
                    elif layer_id in [2, 3, 4]:
                        output_lite = output[:, level_start_index[1]:, :]
                        output_res = output[:, :level_start_index[1], :]
                        reference_points_lite = reference_points[:, level_start_index[1]:, ...]
                        query_pos_lite = query_pos[:, level_start_index[1]:, :]
                        spatial_shapes_lite = spatial_shapes[1:]
                        # if key_padding_mask is not None:
                        #     key_padding_mask_lite = key_padding_mask[:, level_start_index[1]:]
                        # else:
                        #     key_padding_mask_lite = None

                    output_lite = layer(
                        query=output_lite,
                        query_pos=query_pos_lite,
                        memory=output,
                        reference_points=reference_points_lite,
                        spatial_shapes=spatial_shapes,
                        level_start_index=level_start_index,
                        key_padding_mask=key_padding_mask)

                    if self.guide:
                        if layer_id in [2, 3, 4]:
                            H2, W2 = spatial_shapes_lite[0]
                            H1, W1 = spatial_shapes[0]
                            l2_ind = (H2 * W2).item()
                            bs, _, dim = output_res.shape
                            output_res = output_res + F.interpolate(output_lite[:,:l2_ind,:].reshape(bs, H2, W2, dim).permute(0,-1, 1, 2),
                                                   size=(H1, W1), mode='nearest').reshape(bs,dim,
                                                                                           output_res.shape[1]).permute(0,2,1).contiguous()
                        elif layer_id in [0, 1]:
                            H3, W3 = spatial_shapes_lite[0]
                            H1, W1 = spatial_shapes[0]
                            H2, W2 = spatial_shapes[1]
                            l3_ind = (H3 * W3).item()
                            bs, _, dim = output_res.shape
                            P3 = output_res[:, :H1 * W1, :]
                            P4 = output_res[:, H1 * W1:, :]
                            P4 = P4 + F.interpolate(output_lite[:,:l3_ind,:].reshape(bs, H3, W3, dim).permute(0,-1, 1, 2),
                                                   size=(H2, W2), mode='nearest').reshape(bs,dim,
                                                                                           P4.shape[1]).permute(0,2,1).contiguous()
                            P3 = P3 + F.interpolate(P4.reshape(bs, H2, W2, dim).permute(0,-1, 1, 2),
                                                   size=(H1, W1), mode='nearest').reshape(bs,dim,
                                                                                           P3.shape[1]).permute(0,2,1).contiguous()
                            output_res = torch.cat((P3, P4), dim=1)
                    output = torch.cat((output_res, output_lite), dim=1)
        return output, memory_text

class GuideFusionLayer(BaseModule):
    def __init__(self,
                 ffn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True)),
                 norm_cfg: OptConfigType = dict(type='LN'),
                 init_cfg: OptConfigType = None,
                 ) -> None:

        super().__init__(init_cfg=init_cfg)

        self.ffn_cfg = ffn_cfg
        self.norm_cfg = norm_cfg
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize FFN, and normalization."""
        self.embed_dims = self.ffn_cfg['embed_dims']
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(2)
        ]
        self.norms = ModuleList(norms_list)
        # self.drop_path = DropPath(
        #     self.drop_path) if self.drop_path > 0. else nn.Identity()
        # self.gamma_4 = nn.Parameter(
        #     self.init_values * torch.ones(self.embed_dims), requires_grad=True)
        # self.gamma_5 = nn.Parameter(
        #     self.init_values * torch.ones(self.embed_dims), requires_grad=True)


    def forward(self,
                low_feats: Tensor,
                spatial_shapes_low,
                high_feats,
                spatial_shapes_high,
                level_start_index_high) -> Tensor:
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
        bs, _, dim = low_feats.shape
        H3, W3 = spatial_shapes_low[0]
        H4, W4 = spatial_shapes_high[0]
        H5, W5 = spatial_shapes_high[1]

        P4 = high_feats[:, :level_start_index_high[1]]
        P5 = high_feats[:, level_start_index_high[1]:level_start_index_high[2]]

        P4_upsample = F.interpolate(
            P4.reshape(bs, H4, W4, dim).permute(0, -1, 1, 2),
            size=(H3, W3), mode='nearest').reshape(bs, dim,
                                                   H3 * W3).permute(0, 2, 1).contiguous()
        P5_upsample = F.interpolate(
            P5.reshape(bs, H5, W5, dim).permute(0, -1, 1, 2),
            size=(H3, W3), mode='nearest').reshape(bs, dim,
                                                   H3 * W3).permute(0, 2, 1).contiguous()
        low_feats = low_feats + P4_upsample + P5_upsample

        low_feats = self.norms[0](low_feats)
        low_feats = self.ffn(low_feats)
        low_feats = self.norms[1](low_feats)

        return low_feats

class GroundingDinoDecoupleTextTransformerEncoder(DeformableDetrTransformerEncoder):

    def __init__(self, text_layer_cfg: ConfigType,
                 fusion_layer_cfg: ConfigType, **kwargs) -> None:
        self.text_layer_cfg = text_layer_cfg
        self.fusion_layer_cfg = fusion_layer_cfg
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList([
            DeformableDetrTransformerEncoderLayer(**self.layer_cfg)
            # DeformableDetrDecoupleTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        if self.text_layer_cfg is not None:
            self.text_layers = ModuleList([
                DetrTransformerEncoderLayer(**self.text_layer_cfg)
                for _ in range(self.num_layers)
            ])
        else:
            self.text_layers = None
        self.fusion_layers = ModuleList([
            TransformerDecoupleFusionLayer(**self.fusion_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.num_cp > 0:
            if checkpoint_wrapper is None:
                raise NotImplementedError(
                    'If you want to reduce GPU memory usage, \
                    please install fairscale by executing the \
                    following command: pip install fairscale.')
            for i in range(self.num_cp):
                self.layers[i] = checkpoint_wrapper(self.layers[i])
                self.fusion_layers[i] = checkpoint_wrapper(
                    self.fusion_layers[i])

    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                key_padding_mask: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                memory_text: Tensor = None,
                text_attention_mask: Tensor = None,
                pos_text: Tensor = None,
                text_self_attention_masks: Tensor = None,
                position_ids: Tensor = None):
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
            memory_text (Tensor, optional): Memory text. It has shape (bs,
                len_text, text_embed_dims).
            text_attention_mask (Tensor, optional): Text token mask. It has
                shape (bs,len_text).
            pos_text (Tensor, optional): The positional encoding for text.
                Defaults to None.
            text_self_attention_masks (Tensor, optional): Text self attention
                mask. Defaults to None.
            position_ids (Tensor, optional): Text position ids.
                Defaults to None.
        """
        output = query
        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=query.device)
        if self.text_layers:
            # generate pos_text
            bs, n_text, _ = memory_text.shape
            if pos_text is None and position_ids is None:
                pos_text = (
                    torch.arange(n_text,
                                 device=memory_text.device).float().unsqueeze(
                                     0).unsqueeze(-1).repeat(bs, 1, 1))
                pos_text = get_text_sine_pos_embed(
                    pos_text, num_pos_feats=256, exchange_xy=False)
            if position_ids is not None:
                pos_text = get_text_sine_pos_embed(
                    position_ids[..., None],
                    num_pos_feats=256,
                    exchange_xy=False)

        # main process
        for layer_id, layer in enumerate(self.layers):
            if self.fusion_layers:
                output = self.fusion_layers[layer_id](
                    output=output,
                    memory_text=memory_text,
                    query_pos=query_pos,
                    pos_text=(pos_text if pos_text is not None else None),
                    attn_mask=None,
                    text_attention_mask=text_attention_mask,
                    spatial_shapes=spatial_shapes
                )
            if self.text_layers:
                text_num_heads = self.text_layers[
                    layer_id].self_attn_cfg.num_heads
                memory_text = self.text_layers[layer_id](
                    query=memory_text,
                    query_pos=(pos_text if pos_text is not None else None),
                    attn_mask=~text_self_attention_masks.repeat(
                        text_num_heads, 1, 1),  # note we use ~ for mask here
                    key_padding_mask=None,
                )
            output = layer(
                query=output,
                query_pos=query_pos,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=key_padding_mask)
        return output, memory_text

class GroundingDinoVisTransformerEncoder(DeformableDetrTransformerEncoder):

    def __init__(self, text_layer_cfg: ConfigType,
                 fusion_layer_cfg: ConfigType, **kwargs) -> None:
        self.text_layer_cfg = text_layer_cfg
        self.fusion_layer_cfg = fusion_layer_cfg
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList([
            DeformableDetrTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.text_layers = ModuleList([
            DetrTransformerEncoderLayer(**self.text_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.fusion_layers = ModuleList([
            MultiheadAttention(**self.fusion_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.num_cp > 0:
            if checkpoint_wrapper is None:
                raise NotImplementedError(
                    'If you want to reduce GPU memory usage, \
                    please install fairscale by executing the \
                    following command: pip install fairscale.')
            for i in range(self.num_cp):
                self.layers[i] = checkpoint_wrapper(self.layers[i])
                self.fusion_layers[i] = checkpoint_wrapper(
                    self.fusion_layers[i])

    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                key_padding_mask: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                memory_text: Tensor = None,
                text_attention_mask: Tensor = None,
                pos_text: Tensor = None,
                text_self_attention_masks: Tensor = None,
                position_ids: Tensor = None):
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
            memory_text (Tensor, optional): Memory text. It has shape (bs,
                len_text, text_embed_dims).
            text_attention_mask (Tensor, optional): Text token mask. It has
                shape (bs,len_text).
            pos_text (Tensor, optional): The positional encoding for text.
                Defaults to None.
            text_self_attention_masks (Tensor, optional): Text self attention
                mask. Defaults to None.
            position_ids (Tensor, optional): Text position ids.
                Defaults to None.
        """
        output = query
        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=query.device)
        if self.text_layers:
            # generate pos_text
            bs, n_text, _ = memory_text.shape
            if pos_text is None and position_ids is None:
                pos_text = (
                    torch.arange(n_text,
                                 device=memory_text.device).float().unsqueeze(
                                     0).unsqueeze(-1).repeat(bs, 1, 1))
                pos_text = get_text_sine_pos_embed(
                    pos_text, num_pos_feats=256, exchange_xy=False)
            if position_ids is not None:
                pos_text = get_text_sine_pos_embed(
                    position_ids[..., None],
                    num_pos_feats=256,
                    exchange_xy=False)

        # main process
        for layer_id, layer in enumerate(self.layers):
            if self.fusion_layers:
                output = self.fusion_layers[layer_id](
                    query=output,
                    key=memory_text,
                    value=memory_text,
                    query_pos=query_pos,
                    key_pos=(pos_text if pos_text is not None else None),
                    attn_mask=None,
                    key_padding_mask=text_attention_mask,
                )
            if self.text_layers:
                text_num_heads = self.text_layers[
                    layer_id].self_attn_cfg.num_heads
                memory_text = self.text_layers[layer_id](
                    query=memory_text,
                    query_pos=(pos_text if pos_text is not None else None),
                    attn_mask=~text_self_attention_masks.repeat(
                        text_num_heads, 1, 1),  # note we use ~ for mask here
                    key_padding_mask=None,
                )
            output = layer(
                query=output,
                query_pos=query_pos,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=key_padding_mask)
        return output, memory_text

class GroundingDinoCAMFPNTransformerEncoder(DeformableDetrTransformerEncoder):

    def __init__(self, text_layer_cfg: ConfigType,
                 fusion_layer_cfg: ConfigType,
                 upsample_cfg: ConfigType = dict(mode='nearest'),
                 **kwargs) -> None:
        self.text_layer_cfg = text_layer_cfg
        self.fusion_layer_cfg = fusion_layer_cfg
        self.upsample_cfg = upsample_cfg.copy()
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList([
            DeformableDetrTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.text_layers = ModuleList([
            DetrTransformerEncoderLayer(**self.text_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.fusion_layers = ModuleList([
            MultiheadAttention(**self.fusion_layer_cfg)
            for _ in range(self.num_layers * 4)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.num_cp > 0:
            if checkpoint_wrapper is None:
                raise NotImplementedError(
                    'If you want to reduce GPU memory usage, \
                    please install fairscale by executing the \
                    following command: pip install fairscale.')
            for i in range(self.num_cp):
                self.layers[i] = checkpoint_wrapper(self.layers[i])
                self.fusion_layers[i] = checkpoint_wrapper(
                    self.fusion_layers[i])

    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                key_padding_mask: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                memory_text: Tensor = None,
                text_attention_mask: Tensor = None,
                pos_text: Tensor = None,
                text_self_attention_masks: Tensor = None,
                position_ids: Tensor = None):
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
            memory_text (Tensor, optional): Memory text. It has shape (bs,
                len_text, text_embed_dims).
            text_attention_mask (Tensor, optional): Text token mask. It has
                shape (bs,len_text).
            pos_text (Tensor, optional): The positional encoding for text.
                Defaults to None.
            text_self_attention_masks (Tensor, optional): Text self attention
                mask. Defaults to None.
            position_ids (Tensor, optional): Text position ids.
                Defaults to None.
        """
        output = query
        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=query.device)
        if self.text_layers:
            # generate pos_text
            bs, n_text, _ = memory_text.shape
            if pos_text is None and position_ids is None:
                pos_text = (
                    torch.arange(n_text,
                                 device=memory_text.device).float().unsqueeze(
                                     0).unsqueeze(-1).repeat(bs, 1, 1))
                pos_text = get_text_sine_pos_embed(
                    pos_text, num_pos_feats=256, exchange_xy=False)
            if position_ids is not None:
                pos_text = get_text_sine_pos_embed(
                    position_ids[..., None],
                    num_pos_feats=256,
                    exchange_xy=False)

        # main process
        feat_inds = [(H * W).item() for (H, W) in spatial_shapes]
        feat_num = len(feat_inds)

        for layer_id, layer in enumerate(self.layers):
            if self.fusion_layers:
                output_ = []
                output_inds = []
                query_pos_inds = []
                start = 0
                for ind, feat_ind in enumerate(feat_inds):
                    output_ind = output[:, start:start + feat_ind, :]
                    if key_padding_mask is not None:
                        key_padding_mask_ind = key_padding_mask[:, start:start + feat_ind]
                    else:
                        key_padding_mask_ind = None
                    query_pos_ind = query_pos[:, start:start + feat_ind, :]
                    start += feat_ind
                    output_inds.append(output_ind)
                    query_pos_inds.append(query_pos_ind)

                for ind in range(len(output_inds), 0, -1):
                    ind -= 1
                    output_ind = output_inds[ind]
                    query_pos_ind = query_pos_inds[ind]
                    if ind < len(output_inds) - 1:
                        bs, _, dim = output_ind.shape
                        high_feat = output_[0].permute(0,2,1).reshape(bs, dim, spatial_shapes[ind + 1][0], spatial_shapes[ind + 1][1])
                        low_feat = output_ind.permute(0,2,1).reshape(bs, dim, spatial_shapes[ind][0], spatial_shapes[ind][1])
                        fusion_feat = low_feat + F.interpolate(
                            high_feat, size=low_feat.shape[2:], **self.upsample_cfg)

                        output_ind = fusion_feat.reshape(bs, dim, -1).permute(0,2,1)
                    output_ind = self.fusion_layers[layer_id * feat_num + ind](
                        query=output_ind,
                        key=memory_text,
                        value=memory_text,
                        query_pos=query_pos_ind,
                        key_pos=(pos_text if pos_text is not None else None),
                        attn_mask=None,
                        key_padding_mask=text_attention_mask,
                    )
                    # if ind < len(output_inds) - 1:
                    output_.insert(0, output_ind)
                    # else:
                    #     output_.append(output_ind)
                output = torch.cat(output_, dim=1)
            if self.text_layers:
                text_num_heads = self.text_layers[
                    layer_id].self_attn_cfg.num_heads
                memory_text = self.text_layers[layer_id](
                    query=memory_text,
                    query_pos=(pos_text if pos_text is not None else None),
                    attn_mask=~text_self_attention_masks.repeat(
                        text_num_heads, 1, 1),  # note we use ~ for mask here
                    key_padding_mask=None,
                )
            output = layer(
                query=output,
                query_pos=query_pos,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=key_padding_mask)
        return output, memory_text

class GroundingDinoCoarseToFineTransformerEncoder(DeformableDetrTransformerEncoder):

    def __init__(self, text_layer_cfg: ConfigType,
                 fusion_layer_cfg: ConfigType,
                 coarse_layer_num,
                 fine_ratio,
                 **kwargs) -> None:
        self.text_layer_cfg = text_layer_cfg
        self.fusion_layer_cfg = fusion_layer_cfg
        self.coarse_layer_num = coarse_layer_num
        self.fine_ratio = fine_ratio

        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.coarse_layers = ModuleList([
            DeformableDetrTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.coarse_layer_num)
        ])
        # self.fine_layers = ModuleList([
        #     DeformableDetrSparseTransformerEncoderLayer(**self.layer_cfg)
        #     for _ in range(self.num_layers - self.coarse_layer_num)
        # ])
        self.fine_layers = ModuleList([
            DeformableDetrTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers - self.coarse_layer_num)
        ])
        self.text_layers = ModuleList([
            DetrTransformerEncoderLayer(**self.text_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.fusion_layers = ModuleList([
            SingleScaleBiAttentionBlock(**self.fusion_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.coarse_layers[0].embed_dims
        if self.num_cp > 0:
            if checkpoint_wrapper is None:
                raise NotImplementedError(
                    'If you want to reduce GPU memory usage, \
                    please install fairscale by executing the \
                    following command: pip install fairscale.')
            for i in range(self.num_cp):
                self.fusion_layers[i] = checkpoint_wrapper(
                    self.fusion_layers[i])
            for i in range(self.coarse_layer_num):
                self.coarse_layers[i] = checkpoint_wrapper(self.coarse_layers[i])
            for i in range(self.num_layers - self.coarse_layer_num):
                self.fine_layers[i] = checkpoint_wrapper(self.fine_layers[i])

    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                key_padding_mask: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                memory_text: Tensor = None,
                text_attention_mask: Tensor = None,
                pos_text: Tensor = None,
                text_self_attention_masks: Tensor = None,
                position_ids: Tensor = None,
                sparse_cls_branch=None):
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
            memory_text (Tensor, optional): Memory text. It has shape (bs,
                len_text, text_embed_dims).
            text_attention_mask (Tensor, optional): Text token mask. It has
                shape (bs,len_text).
            pos_text (Tensor, optional): The positional encoding for text.
                Defaults to None.
            text_self_attention_masks (Tensor, optional): Text self attention
                mask. Defaults to None.
            position_ids (Tensor, optional): Text position ids.
                Defaults to None.
        """
        output = query
        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=query.device)
        if self.text_layers:
            # generate pos_text
            bs, n_text, _ = memory_text.shape
            if pos_text is None and position_ids is None:
                pos_text = (
                    torch.arange(n_text,
                                 device=memory_text.device).float().unsqueeze(
                                     0).unsqueeze(-1).repeat(bs, 1, 1))
                pos_text = get_text_sine_pos_embed(
                    pos_text, num_pos_feats=256, exchange_xy=False)
            if position_ids is not None:
                pos_text = get_text_sine_pos_embed(
                    position_ids[..., None],
                    num_pos_feats=256,
                    exchange_xy=False)

        binary_class_list = []
        # main process
        for layer_id, layer in enumerate(self.coarse_layers):
            if self.fusion_layers:
                output, memory_text = self.fusion_layers[layer_id](
                    visual_feature=output,
                    lang_feature=memory_text,
                    attention_mask_v=key_padding_mask,
                    attention_mask_l=text_attention_mask,
                )
            if self.text_layers:
                text_num_heads = self.text_layers[
                    layer_id].self_attn_cfg.num_heads
                memory_text = self.text_layers[layer_id](
                    query=memory_text,
                    query_pos=(pos_text if pos_text is not None else None),
                    attn_mask=~text_self_attention_masks.repeat(
                        text_num_heads, 1, 1),  # note we use ~ for mask here
                    key_padding_mask=None,
                )
            output = layer(
                query=output,
                query_pos=query_pos,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=key_padding_mask)
            binary_class = sparse_cls_branch[layer_id](output)
            binary_class_list.append(binary_class)

        _, query_number, c = output.shape
        fine_queries = int((query_number + 1) * self.fine_ratio)
        topk_indices = torch.topk(
            binary_class_list[-1][..., 0], fine_queries, dim=1)[1]
        # fine_query = torch.gather(
        #     output, 1,
        #     topk_indices.unsqueeze(-1).repeat(1, 1, c))
        # fine_query_pos = torch.gather(
        #     query_pos, 1,
        #     topk_indices.unsqueeze(-1).repeat(1, 1, c))
        # fine_reference_points = torch.gather(
        #     reference_points, 1,
        #     topk_indices.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, reference_points.shape[-2], reference_points.shape[-1]))
        if key_padding_mask is not None:
            fine_key_padding_mask = torch.gather(
            key_padding_mask, 1, topk_indices)
        else:
            fine_key_padding_mask = None
        for layer_id, layer in enumerate(self.fine_layers):
            fine_query = torch.gather(output, 1, topk_indices.unsqueeze(-1).repeat(1, 1, c))
            if self.fusion_layers:
                fine_query, memory_text = self.fusion_layers[layer_id + self.coarse_layer_num](
                    visual_feature=fine_query,
                    lang_feature=memory_text,
                    attention_mask_v=fine_key_padding_mask,
                    attention_mask_l=text_attention_mask,
                )
            if self.text_layers:
                text_num_heads = self.text_layers[
                    layer_id + self.coarse_layer_num].self_attn_cfg.num_heads
                memory_text = self.text_layers[layer_id](
                    query=memory_text,
                    query_pos=(pos_text if pos_text is not None else None),
                    attn_mask=~text_self_attention_masks.repeat(
                        text_num_heads, 1, 1),  # note we use ~ for mask here
                    key_padding_mask=None,
                )

            output = output.scatter(1, topk_indices.unsqueeze(-1).repeat(1, 1, output.shape[-1]), fine_query)

            output = layer(
                query=output,
                query_pos=query_pos,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=key_padding_mask)

        #     fine_query = layer(
        #         query=fine_query,
        #         query_pos=fine_query_pos,
        #         memory=output,
        #         reference_points=fine_reference_points,
        #         spatial_shapes=spatial_shapes,
        #         level_start_index=level_start_index,
        #         key_padding_mask=key_padding_mask)
        #     #
        # output = output.scatter(1, topk_indices.unsqueeze(-1).repeat(1, 1, output.shape[-1]), fine_query)
        return output, memory_text, binary_class_list

class GroundingDinoDecoupleLinearTransformerEncoder(DeformableDetrTransformerEncoder):

    def __init__(self, text_layer_cfg: ConfigType,
                 fusion_layer_cfg: ConfigType, **kwargs) -> None:
        self.text_layer_cfg = text_layer_cfg
        self.fusion_layer_cfg = fusion_layer_cfg
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList([
            DeformableDetrTransformerEncoderLayer(**self.layer_cfg)
            # DeformableDetrDecoupleTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.text_layers = ModuleList([
            DetrTransformerEncoderLayer(**self.text_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.fusion_layers = ModuleList([
            TransformerDecoupleFusionLayer(**self.fusion_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.num_cp > 0:
            if checkpoint_wrapper is None:
                raise NotImplementedError(
                    'If you want to reduce GPU memory usage, \
                    please install fairscale by executing the \
                    following command: pip install fairscale.')
            for i in range(self.num_cp):
                self.layers[i] = checkpoint_wrapper(self.layers[i])
                self.fusion_layers[i] = checkpoint_wrapper(
                    self.fusion_layers[i])

    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                key_padding_mask: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                memory_text: Tensor = None,
                text_attention_mask: Tensor = None,
                pos_text: Tensor = None,
                text_self_attention_masks: Tensor = None,
                position_ids: Tensor = None):
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
            memory_text (Tensor, optional): Memory text. It has shape (bs,
                len_text, text_embed_dims).
            text_attention_mask (Tensor, optional): Text token mask. It has
                shape (bs,len_text).
            pos_text (Tensor, optional): The positional encoding for text.
                Defaults to None.
            text_self_attention_masks (Tensor, optional): Text self attention
                mask. Defaults to None.
            position_ids (Tensor, optional): Text position ids.
                Defaults to None.
        """
        output = query
        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=query.device)
        if self.text_layers:
            # generate pos_text
            bs, n_text, _ = memory_text.shape
            if pos_text is None and position_ids is None:
                pos_text = (
                    torch.arange(n_text,
                                 device=memory_text.device).float().unsqueeze(
                                     0).unsqueeze(-1).repeat(bs, 1, 1))
                pos_text = get_text_sine_pos_embed(
                    pos_text, num_pos_feats=256, exchange_xy=False)
            if position_ids is not None:
                pos_text = get_text_sine_pos_embed(
                    position_ids[..., None],
                    num_pos_feats=256,
                    exchange_xy=False)

        # main process
        if self.text_layers:
            for layer_id, text_layer in enumerate(self.text_layers):

                output, memory_text = self.fusion_layers[layer_id](
                    output=output,
                    memory_text=memory_text,
                    query_pos=query_pos,
                    pos_text=(pos_text if pos_text is not None else None),
                    attn_mask=None,
                    text_attention_mask=text_attention_mask,
                    spatial_shapes=spatial_shapes,
                    key_padding_mask_image=key_padding_mask
                )

                text_num_heads = text_layer.self_attn_cfg.num_heads
                memory_text = text_layer(
                    query=memory_text,
                    query_pos=(pos_text if pos_text is not None else None),
                    attn_mask=~text_self_attention_masks.repeat(
                        text_num_heads, 1, 1),  # note we use ~ for mask here
                    key_padding_mask=None,
                )

                output = self.layers[layer_id](
                    query=output,
                    query_pos=query_pos,
                    reference_points=reference_points,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    key_padding_mask=key_padding_mask)
        return output, memory_text


class GroundingDinoDecoupleMSTransformerEncoder(DeformableDetrTransformerEncoder):

    def __init__(self, text_layer_cfg: ConfigType,
                 fusion_layer_cfg: ConfigType, **kwargs) -> None:
        self.text_layer_cfg = text_layer_cfg
        self.fusion_layer_cfg = fusion_layer_cfg
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList([
            DeformableDetrTransformerEncoderLayer(**self.layer_cfg)
            # DeformableDetrDecoupleTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.text_layers = ModuleList([
            DetrTransformerEncoderLayer(**self.text_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.fusion_layers = ModuleList([
            TransformerDecoupleMSFusionLayer(**self.fusion_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.num_cp > 0:
            if checkpoint_wrapper is None:
                raise NotImplementedError(
                    'If you want to reduce GPU memory usage, \
                    please install fairscale by executing the \
                    following command: pip install fairscale.')
            for i in range(self.num_cp):
                self.layers[i] = checkpoint_wrapper(self.layers[i])
                self.fusion_layers[i] = checkpoint_wrapper(
                    self.fusion_layers[i])

    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                key_padding_mask: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                memory_text: Tensor = None,
                text_attention_mask: Tensor = None,
                pos_text: Tensor = None,
                text_self_attention_masks: Tensor = None,
                position_ids: Tensor = None):
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
            memory_text (Tensor, optional): Memory text. It has shape (bs,
                len_text, text_embed_dims).
            text_attention_mask (Tensor, optional): Text token mask. It has
                shape (bs,len_text).
            pos_text (Tensor, optional): The positional encoding for text.
                Defaults to None.
            text_self_attention_masks (Tensor, optional): Text self attention
                mask. Defaults to None.
            position_ids (Tensor, optional): Text position ids.
                Defaults to None.
        """
        output = query
        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=query.device)
        if self.text_layers:
            # generate pos_text
            bs, n_text, _ = memory_text.shape
            if pos_text is None and position_ids is None:
                pos_text = (
                    torch.arange(n_text,
                                 device=memory_text.device).float().unsqueeze(
                                     0).unsqueeze(-1).repeat(bs, 1, 1))
                pos_text = get_text_sine_pos_embed(
                    pos_text, num_pos_feats=256, exchange_xy=False)
            if position_ids is not None:
                pos_text = get_text_sine_pos_embed(
                    position_ids[..., None],
                    num_pos_feats=256,
                    exchange_xy=False)

        # main process
        for layer_id, layer in enumerate(self.layers):
            if self.fusion_layers:
                output, memory_text = self.fusion_layers[layer_id](
                    output=output,
                    memory_text=memory_text,
                    query_pos=query_pos,
                    pos_text=(pos_text if pos_text is not None else None),
                    attn_mask=None,
                    text_attention_mask=text_attention_mask,
                    spatial_shapes=spatial_shapes,
                    key_padding_mask_image=key_padding_mask
                )
            if self.text_layers:
                text_num_heads = self.text_layers[
                    layer_id].self_attn_cfg.num_heads
                memory_text = self.text_layers[layer_id](
                    query=memory_text,
                    query_pos=(pos_text if pos_text is not None else None),
                    attn_mask=~text_self_attention_masks.repeat(
                        text_num_heads, 1, 1),  # note we use ~ for mask here
                    key_padding_mask=None,
                )
            output = layer(
                query=output,
                query_pos=query_pos,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=key_padding_mask)
        return output, memory_text

class GroundingDinoTransformerDecoder(DinoTransformerDecoder):

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            GroundingDinoTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')
        self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                  self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)

class GroundingDinoDecoupleTextSingleTransformerEncoder(DeformableDetrTransformerEncoder):

    def __init__(self, text_layer_cfg: ConfigType,
                 fusion_layer_cfg: ConfigType, **kwargs) -> None:
        self.text_layer_cfg = text_layer_cfg
        self.fusion_layer_cfg = fusion_layer_cfg
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList([
            DeformableDetrTransformerEncoderLayer(**self.layer_cfg)
            # DeformableDetrDecoupleTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.text_layers = ModuleList([
            DetrTransformerEncoderLayer(**self.text_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.fusion_layers = ModuleList([
            TransformerDecoupleTextFusionLayer(**self.fusion_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.num_cp > 0:
            if checkpoint_wrapper is None:
                raise NotImplementedError(
                    'If you want to reduce GPU memory usage, \
                    please install fairscale by executing the \
                    following command: pip install fairscale.')
            for i in range(self.num_cp):
                self.layers[i] = checkpoint_wrapper(self.layers[i])
                self.fusion_layers[i] = checkpoint_wrapper(
                    self.fusion_layers[i])

    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                key_padding_mask: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                memory_text: Tensor = None,
                text_attention_mask: Tensor = None,
                pos_text: Tensor = None,
                text_self_attention_masks: Tensor = None,
                position_ids: Tensor = None):
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
            memory_text (Tensor, optional): Memory text. It has shape (bs,
                len_text, text_embed_dims).
            text_attention_mask (Tensor, optional): Text token mask. It has
                shape (bs,len_text).
            pos_text (Tensor, optional): The positional encoding for text.
                Defaults to None.
            text_self_attention_masks (Tensor, optional): Text self attention
                mask. Defaults to None.
            position_ids (Tensor, optional): Text position ids.
                Defaults to None.
        """
        output = query
        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=query.device)
        if self.text_layers:
            # generate pos_text
            bs, n_text, _ = memory_text.shape
            if pos_text is None and position_ids is None:
                pos_text = (
                    torch.arange(n_text,
                                 device=memory_text.device).float().unsqueeze(
                                     0).unsqueeze(-1).repeat(bs, 1, 1))
                pos_text = get_text_sine_pos_embed(
                    pos_text, num_pos_feats=256, exchange_xy=False)
            if position_ids is not None:
                pos_text = get_text_sine_pos_embed(
                    position_ids[..., None],
                    num_pos_feats=256,
                    exchange_xy=False)

        # main process
        for layer_id, layer in enumerate(self.layers):
            if self.fusion_layers:
                output, memory_text = self.fusion_layers[layer_id](
                    output=output,
                    memory_text=memory_text,
                    query_pos=query_pos,
                    pos_text=(pos_text if pos_text is not None else None),
                    attn_mask=None,
                    text_attention_mask=text_attention_mask,
                    spatial_shapes=spatial_shapes,
                    key_padding_mask_image=key_padding_mask
                )
            if self.text_layers:
                text_num_heads = self.text_layers[
                    layer_id].self_attn_cfg.num_heads
                memory_text = self.text_layers[layer_id](
                    query=memory_text,
                    query_pos=(pos_text if pos_text is not None else None),
                    attn_mask=~text_self_attention_masks.repeat(
                        text_num_heads, 1, 1),  # note we use ~ for mask here
                    key_padding_mask=None,
                )
            output = layer(
                query=output,
                query_pos=query_pos,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=key_padding_mask)
        return output, memory_text

class TransformerDecoupleFusionLayer(BaseModule):
    """Implements decoder layer in DETR transformer.

    Args:
        self_attn_cfg (:obj:`ConfigDict` or dict, optional): Config for self
            attention.
        cross_attn_cfg (:obj:`ConfigDict` or dict, optional): Config for cross
            attention.
        ffn_cfg (:obj:`ConfigDict` or dict, optional): Config for FFN.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config for
            normalization layers. All the layers will share the same
            config. Defaults to `LN`.
        init_cfg (:obj:`ConfigDict` or dict, optional): Config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 cross_attn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     num_heads=8,
                     dropout=0.0,
                     batch_first=True),
                 feat_attn_layer_num=[1, 1, 2, 2],
                 ffn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True),
                 ),
                 norm_cfg: OptConfigType = dict(type='LN'),
                 init_cfg: OptConfigType = None) -> None:

        super().__init__(init_cfg=init_cfg)

        self.cross_attn_cfg = cross_attn_cfg

        if 'batch_first' not in self.cross_attn_cfg:
            self.cross_attn_cfg['batch_first'] = True
        else:
            assert self.cross_attn_cfg['batch_first'] is True, 'First \
            dimension of all DETRs in mmdet is `batch`, \
            please set `batch_first` flag.'

        self.ffn_cfg = ffn_cfg
        self.norm_cfg = norm_cfg
        self.feat_attn_layer_num = feat_attn_layer_num
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize self-attention, FFN, and normalization."""
        self.embed_dims = self.cross_attn_cfg.embed_dims
        self.fusion_layers = []
        for num in self.feat_attn_layer_num:
            fusion_layer = []
            for i in range(num):
                fusion_layer.append(MultiheadAttention(**self.cross_attn_cfg))

                # fusion_layer.append(build_norm_layer(self.norm_cfg, self.embed_dims)[1])
                # fusion_layer.append(FFN(**self.ffn_cfg))
                # fusion_layer.append(build_norm_layer(self.norm_cfg, self.embed_dims)[1])

            self.fusion_layers.append(ModuleList(fusion_layer))
        self.fusion_layers = ModuleList(self.fusion_layers)
        # self.fusion_layers_text = MultiheadAttention(**self.cross_attn_cfg)

        # self.ffn = FFN(**self.ffn_cfg)
        # norms_list = [
        #     build_norm_layer(self.norm_cfg, self.embed_dims)[1]
        #     for _ in range(2)
        # ]
        # self.norms = ModuleList(norms_list)

    def forward(self,
                output: Tensor,
                memory_text: Tensor = None,
                query_pos: Tensor = None,
                pos_text: Tensor = None,
                attn_mask: Tensor = None,
                text_attention_mask: Tensor = None,
                spatial_shapes=None,
                key_padding_mask_image=None) -> Tensor:
        """
        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            key (Tensor, optional): The input key, has shape (bs, num_keys,
                dim). If `None`, the `query` will be used. Defaults to `None`.
            value (Tensor, optional): The input value, has the same shape as
                `key`, as in `nn.MultiheadAttention.forward`. If `None`, the
                `key` will be used. Defaults to `None`.
            query_pos (Tensor, optional): The positional encoding for `query`,
                has the same shape as `query`. If not `None`, it will be added
                to `query` before forward function. Defaults to `None`.
            key_pos (Tensor, optional): The positional encoding for `key`, has
                the same shape as `key`. If not `None`, it will be added to
                `key` before forward function. If None, and `query_pos` has the
                same shape as `key`, then `query_pos` will be used for
                `key_pos`. Defaults to None.
            self_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            cross_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor, optional): The `key_padding_mask` of
                `self_attn` input. ByteTensor, has shape (bs, num_value).
                Defaults to None.

        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """
        feat_inds = [(H * W).item() for (H, W) in spatial_shapes]
        output_ = []
        start = 0
        for ind, feat_ind in enumerate(feat_inds):
            output_ind = output[:, start:start + feat_ind, :]
            query_pos_ind = query_pos[:, start:start + feat_ind, :]
            start += feat_ind
            for i, layer in enumerate(self.fusion_layers[ind]):
                # if i % 4 == 0:
                output_ind_ = layer(
                    query=output_ind,
                    key=memory_text,
                    value=memory_text,
                    query_pos=query_pos_ind,
                    key_pos=(pos_text if pos_text is not None else None),
                    attn_mask=attn_mask,
                    key_padding_mask=text_attention_mask,
                )
                # else:
                #     output_ind = layer(output_ind)
            output_.append(output_ind_)
        # memory_text = self.fusion_layers_text(
        #     query=memory_text,
        #     key=output,
        #     value=output,
        #     query_pos=(pos_text if pos_text is not None else None),
        #     key_pos=query_pos,
        #     attn_mask=attn_mask,
        #     key_padding_mask=key_padding_mask_image, )
        output = torch.cat(output_, dim=1)
        # output_text = self.fc(torch.cat(output_text, dim=-1))
        # output = self.norms[0](output)
        # output = self.ffn(output)
        # output = self.norms[1](output)

        return output

class TransformerLiteDecoupleFusionLayer(BaseModule):
    """Implements decoder layer in DETR transformer.

    Args:
        self_attn_cfg (:obj:`ConfigDict` or dict, optional): Config for self
            attention.
        cross_attn_cfg (:obj:`ConfigDict` or dict, optional): Config for cross
            attention.
        ffn_cfg (:obj:`ConfigDict` or dict, optional): Config for FFN.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config for
            normalization layers. All the layers will share the same
            config. Defaults to `LN`.
        init_cfg (:obj:`ConfigDict` or dict, optional): Config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 cross_attn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     num_heads=8,
                     dropout=0.0,
                     batch_first=True),
                 feat_attn_layer_num=[1, 1, 2, 2],
                 ffn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True),
                 ),
                 norm_cfg: OptConfigType = dict(type='LN'),
                 init_cfg: OptConfigType = None,
                 ) -> None:

        super().__init__(init_cfg=init_cfg)

        self.cross_attn_cfg = cross_attn_cfg

        if 'batch_first' not in self.cross_attn_cfg:
            self.cross_attn_cfg['batch_first'] = True
        else:
            assert self.cross_attn_cfg['batch_first'] is True, 'First \
            dimension of all DETRs in mmdet is `batch`, \
            please set `batch_first` flag.'

        self.ffn_cfg = ffn_cfg
        self.norm_cfg = norm_cfg
        self.feat_attn_layer_num = feat_attn_layer_num
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize self-attention, FFN, and normalization."""
        self.embed_dims = self.cross_attn_cfg.embed_dims
        self.fusion_layers = []
        for num in self.feat_attn_layer_num:
            fusion_layer = []
            for i in range(num):
                fusion_layer.append(MultiheadAttention(**self.cross_attn_cfg))

                # fusion_layer.append(build_norm_layer(self.norm_cfg, self.embed_dims)[1])
                # fusion_layer.append(FFN(**self.ffn_cfg))
                # fusion_layer.append(build_norm_layer(self.norm_cfg, self.embed_dims)[1])

            self.fusion_layers.append(ModuleList(fusion_layer))
        self.fusion_layers = ModuleList(self.fusion_layers)
        # self.fusion_layers_text = MultiheadAttention(**self.cross_attn_cfg)

        # self.ffn = FFN(**self.ffn_cfg)
        # norms_list = [
        #     build_norm_layer(self.norm_cfg, self.embed_dims)[1]
        #     for _ in range(2)
        # ]
        # self.norms = ModuleList(norms_list)

    def forward(self,
                output: Tensor,
                memory_text: Tensor = None,
                query_pos: Tensor = None,
                pos_text: Tensor = None,
                attn_mask: Tensor = None,
                text_attention_mask: Tensor = None,
                spatial_shapes=None,
                key_padding_mask_image=None) -> Tensor:
        """
        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            key (Tensor, optional): The input key, has shape (bs, num_keys,
                dim). If `None`, the `query` will be used. Defaults to `None`.
            value (Tensor, optional): The input value, has the same shape as
                `key`, as in `nn.MultiheadAttention.forward`. If `None`, the
                `key` will be used. Defaults to `None`.
            query_pos (Tensor, optional): The positional encoding for `query`,
                has the same shape as `query`. If not `None`, it will be added
                to `query` before forward function. Defaults to `None`.
            key_pos (Tensor, optional): The positional encoding for `key`, has
                the same shape as `key`. If not `None`, it will be added to
                `key` before forward function. If None, and `query_pos` has the
                same shape as `key`, then `query_pos` will be used for
                `key_pos`. Defaults to None.
            self_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            cross_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor, optional): The `key_padding_mask` of
                `self_attn` input. ByteTensor, has shape (bs, num_value).
                Defaults to None.

        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """
        feat_inds = [(H * W).item() for (H, W) in spatial_shapes]
        output_ = []
        weight_ = []
        start = 0
        for ind, feat_ind in enumerate(feat_inds):
            output_ind = output[:, start:start + feat_ind, :]
            query_pos_ind = query_pos[:, start:start + feat_ind, :]
            start += feat_ind
            for i, layer in enumerate(self.fusion_layers[ind]):
                # if i % 4 == 0:
                output_ind_ = layer(
                    query=output_ind,
                    key=memory_text,
                    value=memory_text,
                    query_pos=query_pos_ind,
                    key_pos=(pos_text if pos_text is not None else None),
                    attn_mask=attn_mask,
                    key_padding_mask=text_attention_mask,
                )
                # else:
                #     output_ind = layer(output_ind)
            output_.append(output_ind_)

        output = torch.cat(output_, dim=1)

        return output

class TransformerOnlyDecoupleFusionLayer(BaseModule):
    """Implements decoder layer in DETR transformer.

    Args:
        self_attn_cfg (:obj:`ConfigDict` or dict, optional): Config for self
            attention.
        cross_attn_cfg (:obj:`ConfigDict` or dict, optional): Config for cross
            attention.
        ffn_cfg (:obj:`ConfigDict` or dict, optional): Config for FFN.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config for
            normalization layers. All the layers will share the same
            config. Defaults to `LN`.
        init_cfg (:obj:`ConfigDict` or dict, optional): Config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 cross_attn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     num_heads=8,
                     dropout=0.0,
                     batch_first=True),
                 feat_attn_layer_num=[1, 1, 1, 1],
                 init_cfg: OptConfigType = None,
                 ) -> None:

        super().__init__(init_cfg=init_cfg)

        self.cross_attn_cfg = cross_attn_cfg

        if 'batch_first' not in self.cross_attn_cfg:
            self.cross_attn_cfg['batch_first'] = True
        else:
            assert self.cross_attn_cfg['batch_first'] is True, 'First \
            dimension of all DETRs in mmdet is `batch`, \
            please set `batch_first` flag.'
        self.feat_attn_layer_num = feat_attn_layer_num
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize self-attention, FFN, and normalization."""
        self.embed_dims = self.cross_attn_cfg.embed_dims
        self.fusion_layers = []
        for num in self.feat_attn_layer_num:
            fusion_layer = []
            for i in range(num):
                fusion_layer.append(MultiheadAttention(**self.cross_attn_cfg))
            self.fusion_layers.append(ModuleList(fusion_layer))
        self.fusion_layers = ModuleList(self.fusion_layers)
        self.fusion_text_layers = ModuleList([MultiheadAttention(**self.cross_attn_cfg)])

    def forward(self,
                output: Tensor,
                memory_text: Tensor = None,
                query_pos: Tensor = None,
                pos_text: Tensor = None,
                attn_mask: Tensor = None,
                text_attention_mask: Tensor = None,
                spatial_shapes=None,
                key_padding_mask_image=None) -> Tensor:
        """
        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            key (Tensor, optional): The input key, has shape (bs, num_keys,
                dim). If `None`, the `query` will be used. Defaults to `None`.
            value (Tensor, optional): The input value, has the same shape as
                `key`, as in `nn.MultiheadAttention.forward`. If `None`, the
                `key` will be used. Defaults to `None`.
            query_pos (Tensor, optional): The positional encoding for `query`,
                has the same shape as `query`. If not `None`, it will be added
                to `query` before forward function. Defaults to `None`.
            key_pos (Tensor, optional): The positional encoding for `key`, has
                the same shape as `key`. If not `None`, it will be added to
                `key` before forward function. If None, and `query_pos` has the
                same shape as `key`, then `query_pos` will be used for
                `key_pos`. Defaults to None.
            self_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            cross_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor, optional): The `key_padding_mask` of
                `self_attn` input. ByteTensor, has shape (bs, num_value).
                Defaults to None.

        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """
        feat_inds = [(H * W).item() for (H, W) in spatial_shapes]
        output_ = []
        weight_ = []
        start = 0
        for ind, feat_ind in enumerate(feat_inds):
            output_ind = output[:, start:start + feat_ind, :]
            query_pos_ind = query_pos[:, start:start + feat_ind, :]
            start += feat_ind
            for i, layer in enumerate(self.fusion_layers[ind]):
                # if i % 4 == 0:
                output_ind_ = layer(
                    query=output_ind,
                    key=memory_text,
                    value=memory_text,
                    query_pos=query_pos_ind,
                    key_pos=(pos_text if pos_text is not None else None),
                    attn_mask=attn_mask,
                    key_padding_mask=text_attention_mask,
                )
                # else:
                #     output_ind = layer(output_ind)
            output_.append(output_ind_)

        for text_layer in self.fusion_text_layers:
            memory_text = text_layer(
                    query=memory_text,
                    key=output,
                    value=output,
                    query_pos=(pos_text if pos_text is not None else None),
                    key_pos=query_pos,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask_image,
                )
        output = torch.cat(output_, dim=1)

        return output, memory_text

class TransformerCascadeFusionLayer(BaseModule):
    """Implements decoder layer in DETR transformer.

    Args:
        self_attn_cfg (:obj:`ConfigDict` or dict, optional): Config for self
            attention.
        cross_attn_cfg (:obj:`ConfigDict` or dict, optional): Config for cross
            attention.
        ffn_cfg (:obj:`ConfigDict` or dict, optional): Config for FFN.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config for
            normalization layers. All the layers will share the same
            config. Defaults to `LN`.
        init_cfg (:obj:`ConfigDict` or dict, optional): Config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 cross_attn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     num_heads=8,
                     dropout=0.0,
                     batch_first=True),
                 feat_attn_layer_num=[1, 1, 2, 2],
                 ffn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True),
                 ),
                 norm_cfg: OptConfigType = dict(type='LN'),
                 init_cfg: OptConfigType = None,
                 ) -> None:

        super().__init__(init_cfg=init_cfg)

        self.cross_attn_cfg = cross_attn_cfg

        if 'batch_first' not in self.cross_attn_cfg:
            self.cross_attn_cfg['batch_first'] = True
        else:
            assert self.cross_attn_cfg['batch_first'] is True, 'First \
            dimension of all DETRs in mmdet is `batch`, \
            please set `batch_first` flag.'

        self.ffn_cfg = ffn_cfg
        self.norm_cfg = norm_cfg
        self.feat_attn_layer_num = feat_attn_layer_num
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize self-attention, FFN, and normalization."""
        self.embed_dims = self.cross_attn_cfg.embed_dims
        self.fusion_layers = []
        for num in self.feat_attn_layer_num:
            fusion_layer = []
            for i in range(num):
                fusion_layer.append(MultiheadAttention(**self.cross_attn_cfg))

                # fusion_layer.append(build_norm_layer(self.norm_cfg, self.embed_dims)[1])
                # fusion_layer.append(FFN(**self.ffn_cfg))
                # fusion_layer.append(build_norm_layer(self.norm_cfg, self.embed_dims)[1])

            self.fusion_layers.append(ModuleList(fusion_layer))
        self.fusion_layers = ModuleList(self.fusion_layers)

    def forward(self,
                output: Tensor,
                memory_text: Tensor = None,
                query_pos: Tensor = None,
                pos_text: Tensor = None,
                attn_mask: Tensor = None,
                text_attention_mask: Tensor = None,
                spatial_shapes=None,
                key_padding_mask_image=None) -> Tensor:
        """
        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            key (Tensor, optional): The input key, has shape (bs, num_keys,
                dim). If `None`, the `query` will be used. Defaults to `None`.
            value (Tensor, optional): The input value, has the same shape as
                `key`, as in `nn.MultiheadAttention.forward`. If `None`, the
                `key` will be used. Defaults to `None`.
            query_pos (Tensor, optional): The positional encoding for `query`,
                has the same shape as `query`. If not `None`, it will be added
                to `query` before forward function. Defaults to `None`.
            key_pos (Tensor, optional): The positional encoding for `key`, has
                the same shape as `key`. If not `None`, it will be added to
                `key` before forward function. If None, and `query_pos` has the
                same shape as `key`, then `query_pos` will be used for
                `key_pos`. Defaults to None.
            self_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            cross_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor, optional): The `key_padding_mask` of
                `self_attn` input. ByteTensor, has shape (bs, num_value).
                Defaults to None.

        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """
        feat_inds = [(H * W).item() for (H, W) in spatial_shapes]
        output_ = []
        weight_ = []
        start = 0
        for ind, feat_ind in enumerate(feat_inds):
            output_ind = output[:, start:start + feat_ind, :]
            query_pos_ind = query_pos[:, start:start + feat_ind, :]
            start += feat_ind
            for i, layer in enumerate(self.fusion_layers[ind]):
                # if i % 4 == 0:
                if layer._get_name() == 'MultiheadAttention':
                    output_ind = layer(
                        query=output_ind,
                        key=memory_text,
                        value=memory_text,
                        query_pos=query_pos_ind,
                        key_pos=(pos_text if pos_text is not None else None),
                        attn_mask=attn_mask,
                        key_padding_mask=text_attention_mask,
                    )
                else:
                    output_ind = layer(output_ind)
            output_.append(output_ind)

        output = torch.cat(output_, dim=1)

        return output

class TransformerGuideFusionLayer(BaseModule):
    """Implements decoder layer in DETR transformer.

    Args:
        self_attn_cfg (:obj:`ConfigDict` or dict, optional): Config for self
            attention.
        cross_attn_cfg (:obj:`ConfigDict` or dict, optional): Config for cross
            attention.
        ffn_cfg (:obj:`ConfigDict` or dict, optional): Config for FFN.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config for
            normalization layers. All the layers will share the same
            config. Defaults to `LN`.
        init_cfg (:obj:`ConfigDict` or dict, optional): Config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 cross_attn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     num_heads=8,
                     dropout=0.0,
                     batch_first=True),
                 feat_attn_layer_num=[1, 1, 2, 2],
                 ffn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True),
                 ),
                 norm_cfg: OptConfigType = dict(type='LN'),
                 init_cfg: OptConfigType = None,
                 guide_layers = [0,2]) -> None:

        super().__init__(init_cfg=init_cfg)

        self.cross_attn_cfg = cross_attn_cfg

        if 'batch_first' not in self.cross_attn_cfg:
            self.cross_attn_cfg['batch_first'] = True
        else:
            assert self.cross_attn_cfg['batch_first'] is True, 'First \
            dimension of all DETRs in mmdet is `batch`, \
            please set `batch_first` flag.'

        self.guide_layers = guide_layers
        self.ffn_cfg = ffn_cfg
        self.norm_cfg = norm_cfg
        self.feat_attn_layer_num = feat_attn_layer_num
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize self-attention, FFN, and normalization."""
        self.embed_dims = self.cross_attn_cfg.embed_dims
        self.fusion_layers = []
        for num in self.feat_attn_layer_num:
            fusion_layer = []
            if num > 0:
                for i in range(num):
                    fusion_layer.append(MultiheadAttention_weight(**self.cross_attn_cfg))

                    # fusion_layer.append(build_norm_layer(self.norm_cfg, self.embed_dims)[1])
                    # fusion_layer.append(FFN(**self.ffn_cfg))
                    # fusion_layer.append(build_norm_layer(self.norm_cfg, self.embed_dims)[1])

                self.fusion_layers.append(ModuleList(fusion_layer))
            else:
                self.fusion_layers.append(nn.Identity())
        self.fusion_layers = ModuleList(self.fusion_layers)
        # self.fusion_layers_text = MultiheadAttention(**self.cross_attn_cfg)

        # self.ffn = FFN(**self.ffn_cfg)
        # norms_list = [
        #     build_norm_layer(self.norm_cfg, self.embed_dims)[1]
        #     for _ in range(2)
        # ]
        # self.norms = ModuleList(norms_list)

    def forward(self,
                output: Tensor,
                memory_text: Tensor = None,
                query_pos: Tensor = None,
                pos_text: Tensor = None,
                attn_mask: Tensor = None,
                text_attention_mask: Tensor = None,
                spatial_shapes=None,
                key_padding_mask_image=None) -> Tensor:
        """
        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            key (Tensor, optional): The input key, has shape (bs, num_keys,
                dim). If `None`, the `query` will be used. Defaults to `None`.
            value (Tensor, optional): The input value, has the same shape as
                `key`, as in `nn.MultiheadAttention.forward`. If `None`, the
                `key` will be used. Defaults to `None`.
            query_pos (Tensor, optional): The positional encoding for `query`,
                has the same shape as `query`. If not `None`, it will be added
                to `query` before forward function. Defaults to `None`.
            key_pos (Tensor, optional): The positional encoding for `key`, has
                the same shape as `key`. If not `None`, it will be added to
                `key` before forward function. If None, and `query_pos` has the
                same shape as `key`, then `query_pos` will be used for
                `key_pos`. Defaults to None.
            self_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            cross_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor, optional): The `key_padding_mask` of
                `self_attn` input. ByteTensor, has shape (bs, num_value).
                Defaults to None.

        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """
        feat_inds = [(H * W).item() for (H, W) in spatial_shapes]
        output_ = []
        weight_ = []
        start = 0
        for ind, feat_ind in enumerate(feat_inds):
            output_ind = output[:, start:start + feat_ind, :]
            query_pos_ind = query_pos[:, start:start + feat_ind, :]
            start += feat_ind
            if ind in self.guide_layers:
                bs, _ ,dim = output_ind.shape
                h, w = spatial_shapes[ind]
                h_h, w_h = spatial_shapes[ind - 1]
                assert h * w == output_ind.shape[1], "input h * w != feature shape!!!"
                for i, layer in enumerate(self.fusion_layers[ind]):
                    # if i % 4 == 0:
                    output_ind, weights_ind_ = layer(
                        query=output_ind,
                        key=memory_text,
                        value=memory_text,
                        query_pos=query_pos_ind,
                        key_pos=(pos_text if pos_text is not None else None),
                        attn_mask=attn_mask,
                        key_padding_mask=text_attention_mask,
                    )
                output_.append(output_ind)

            else:
                output_ind_ = self.fusion_layers[ind](output_ind)
                output_.append(output_ind_)

        bs, _, dim = output.shape
        h_3, w_3 = spatial_shapes[0]
        h_4, w_4 = spatial_shapes[1]
        h_5, w_5 = spatial_shapes[2]
        if self.guide_layers[0] == 1:
            output_[0] = output_[0] + \
                         F.interpolate(output_[1].reshape(bs, h_4, w_4, dim).permute(0,-1, 1, 2),
                                       size=(h_3, w_3), mode='nearest').reshape(bs,dim,
                                                                               feat_inds[0]).permute(0,2,1).contiguous()
            output_[2] = output_[2] + \
                         F.interpolate(output_[1].reshape(bs, h_4, w_4, dim).permute(0,-1, 1, 2),
                                       size=(h_5, w_5), mode='nearest').reshape(bs,dim,
                                                                               feat_inds[2]).permute(0,2,1).contiguous()
        elif self.guide_layers[0] == 2:
            output_[0] = output_[0] + \
                         F.interpolate(output_[2].reshape(bs, h_5, w_5, dim).permute(0,-1, 1, 2),
                                       size=(h_3, w_3), mode='nearest').reshape(bs,dim,
                                                                               feat_inds[0]).permute(0,2,1).contiguous()
            output_[1] = output_[1] + \
                         F.interpolate(output_[2].reshape(bs, h_5, w_5, dim).permute(0,-1, 1, 2),
                                       size=(h_4, w_4), mode='nearest').reshape(bs,dim,
                                                                               feat_inds[1]).permute(0,2,1).contiguous()

        output = torch.cat(output_, dim=1)

        return output


class TransformerMidGuideFusionLayer(BaseModule):
    """Implements decoder layer in DETR transformer.

    Args:
        self_attn_cfg (:obj:`ConfigDict` or dict, optional): Config for self
            attention.
        cross_attn_cfg (:obj:`ConfigDict` or dict, optional): Config for cross
            attention.
        ffn_cfg (:obj:`ConfigDict` or dict, optional): Config for FFN.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config for
            normalization layers. All the layers will share the same
            config. Defaults to `LN`.
        init_cfg (:obj:`ConfigDict` or dict, optional): Config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 cross_attn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     num_heads=8,
                     dropout=0.0,
                     batch_first=True),
                 feat_attn_layer_num=[1, 1, 2, 2],
                 ffn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True),
                 ),
                 norm_cfg: OptConfigType = dict(type='LN'),
                 init_cfg: OptConfigType = None,
                 guide_layers = [0,2]) -> None:

        super().__init__(init_cfg=init_cfg)

        self.cross_attn_cfg = cross_attn_cfg

        if 'batch_first' not in self.cross_attn_cfg:
            self.cross_attn_cfg['batch_first'] = True
        else:
            assert self.cross_attn_cfg['batch_first'] is True, 'First \
            dimension of all DETRs in mmdet is `batch`, \
            please set `batch_first` flag.'

        self.guide_layers = guide_layers
        self.ffn_cfg = ffn_cfg
        self.norm_cfg = norm_cfg
        self.feat_attn_layer_num = feat_attn_layer_num
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize self-attention, FFN, and normalization."""
        self.embed_dims = self.cross_attn_cfg.embed_dims
        self.fusion_layers = []
        for num in self.feat_attn_layer_num:
            fusion_layer = []
            if num > 0:
                for i in range(num):
                    fusion_layer.append(MultiheadAttention_weight(**self.cross_attn_cfg))

                    # fusion_layer.append(build_norm_layer(self.norm_cfg, self.embed_dims)[1])
                    # fusion_layer.append(FFN(**self.ffn_cfg))
                    # fusion_layer.append(build_norm_layer(self.norm_cfg, self.embed_dims)[1])

                self.fusion_layers.append(ModuleList(fusion_layer))
            else:
                self.fusion_layers.append(nn.Identity())
        self.fusion_layers = ModuleList(self.fusion_layers)
        # self.fusion_layers_text = MultiheadAttention(**self.cross_attn_cfg)

        # self.ffn = FFN(**self.ffn_cfg)
        # norms_list = [
        #     build_norm_layer(self.norm_cfg, self.embed_dims)[1]
        #     for _ in range(2)
        # ]
        # self.norms = ModuleList(norms_list)

    def forward(self,
                output: Tensor,
                memory_text: Tensor = None,
                query_pos: Tensor = None,
                pos_text: Tensor = None,
                attn_mask: Tensor = None,
                text_attention_mask: Tensor = None,
                spatial_shapes=None,
                key_padding_mask_image=None) -> Tensor:
        """
        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            key (Tensor, optional): The input key, has shape (bs, num_keys,
                dim). If `None`, the `query` will be used. Defaults to `None`.
            value (Tensor, optional): The input value, has the same shape as
                `key`, as in `nn.MultiheadAttention.forward`. If `None`, the
                `key` will be used. Defaults to `None`.
            query_pos (Tensor, optional): The positional encoding for `query`,
                has the same shape as `query`. If not `None`, it will be added
                to `query` before forward function. Defaults to `None`.
            key_pos (Tensor, optional): The positional encoding for `key`, has
                the same shape as `key`. If not `None`, it will be added to
                `key` before forward function. If None, and `query_pos` has the
                same shape as `key`, then `query_pos` will be used for
                `key_pos`. Defaults to None.
            self_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            cross_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor, optional): The `key_padding_mask` of
                `self_attn` input. ByteTensor, has shape (bs, num_value).
                Defaults to None.

        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """
        feat_inds = [(H * W).item() for (H, W) in spatial_shapes]
        output_ = []
        weight_ = []
        start = 0
        for ind, feat_ind in enumerate(feat_inds):
            output_ind = output[:, start:start + feat_ind, :]
            query_pos_ind = query_pos[:, start:start + feat_ind, :]
            start += feat_ind
            if ind in self.guide_layers:
                bs, _ ,dim = output_ind.shape
                h, w = spatial_shapes[ind]
                h_h, w_h = spatial_shapes[ind - 1]
                assert h * w == output_ind.shape[1], "input h * w != feature shape!!!"
                for i, layer in enumerate(self.fusion_layers[ind]):
                    # if i % 4 == 0:
                    output_ind_, weights_ind_ = layer(
                        query=output_ind,
                        key=memory_text,
                        value=memory_text,
                        query_pos=query_pos_ind,
                        key_pos=(pos_text if pos_text is not None else None),
                        attn_mask=attn_mask,
                        key_padding_mask=text_attention_mask,
                    )
                output_.append(output_ind_)

                output_[ind - 1] = output_[ind - 1] + \
                             F.interpolate(output_ind_.reshape(bs, h, w, dim).permute(0,-1, 1, 2),
                                           size=(h_h, w_h), mode='nearest').reshape(bs,dim,
                                                                                   feat_inds[ind - 1]).permute(0,2,1).contiguous()

            else:
                output_ind_ = self.fusion_layers[ind](output_ind)
                output_.append(output_ind_)

        # bs, _, dim = output.shape
        # h, w = spatial_shapes[1]
        # h_h, w_h = spatial_shapes[0]
        # output_[0] = output_[0] + \
        #              F.interpolate(output_[1].reshape(bs, h, w, dim).permute(0,-1, 1, 2),
        #                            size=(h_h, w_h), mode='nearest').reshape(bs,dim,
        #                                                                    feat_inds[0]).permute(0,2,1).contiguous()

        output = torch.cat(output_, dim=1)

        return output

class TransformerDecoupleTextFusionLayer(BaseModule):
    """Implements decoder layer in DETR transformer.

    Args:
        self_attn_cfg (:obj:`ConfigDict` or dict, optional): Config for self
            attention.
        cross_attn_cfg (:obj:`ConfigDict` or dict, optional): Config for cross
            attention.
        ffn_cfg (:obj:`ConfigDict` or dict, optional): Config for FFN.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config for
            normalization layers. All the layers will share the same
            config. Defaults to `LN`.
        init_cfg (:obj:`ConfigDict` or dict, optional): Config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 cross_attn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     num_heads=8,
                     dropout=0.0,
                     batch_first=True),
                 feat_attn_layer_num=[1, 1, 1, 1],
                 ffn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True),
                 ),
                 norm_cfg: OptConfigType = dict(type='LN'),
                 init_cfg: OptConfigType = None) -> None:

        super().__init__(init_cfg=init_cfg)

        self.cross_attn_cfg = cross_attn_cfg

        if 'batch_first' not in self.cross_attn_cfg:
            self.cross_attn_cfg['batch_first'] = True
        else:
            assert self.cross_attn_cfg['batch_first'] is True, 'First \
            dimension of all DETRs in mmdet is `batch`, \
            please set `batch_first` flag.'

        self.ffn_cfg = ffn_cfg
        self.norm_cfg = norm_cfg
        self.feat_attn_layer_num = feat_attn_layer_num
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize self-attention, FFN, and normalization."""
        self.embed_dims = self.cross_attn_cfg.embed_dims
        self.fusion_layers = []
        for num in self.feat_attn_layer_num:
            fusion_layer = []
            for i in range(num):
                fusion_layer.append(MultiheadAttention(**self.cross_attn_cfg))
            self.fusion_layers.append(ModuleList(fusion_layer))
        self.fusion_layers = ModuleList(self.fusion_layers)
        self.fusion_layers_text = MultiheadAttention(**self.cross_attn_cfg)

    def forward(self,
                output: Tensor,
                memory_text: Tensor = None,
                query_pos: Tensor = None,
                pos_text: Tensor = None,
                attn_mask: Tensor = None,
                text_attention_mask: Tensor = None,
                spatial_shapes=None,
                key_padding_mask_image=None) -> Tensor:
        """
        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            key (Tensor, optional): The input key, has shape (bs, num_keys,
                dim). If `None`, the `query` will be used. Defaults to `None`.
            value (Tensor, optional): The input value, has the same shape as
                `key`, as in `nn.MultiheadAttention.forward`. If `None`, the
                `key` will be used. Defaults to `None`.
            query_pos (Tensor, optional): The positional encoding for `query`,
                has the same shape as `query`. If not `None`, it will be added
                to `query` before forward function. Defaults to `None`.
            key_pos (Tensor, optional): The positional encoding for `key`, has
                the same shape as `key`. If not `None`, it will be added to
                `key` before forward function. If None, and `query_pos` has the
                same shape as `key`, then `query_pos` will be used for
                `key_pos`. Defaults to None.
            self_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            cross_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor, optional): The `key_padding_mask` of
                `self_attn` input. ByteTensor, has shape (bs, num_value).
                Defaults to None.

        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """
        feat_inds = [(H * W).item() for (H, W) in spatial_shapes]
        output_ = []
        start = 0
        for ind, feat_ind in enumerate(feat_inds):
            output_ind = output[:, start:start + feat_ind, :]
            query_pos_ind = query_pos[:, start:start + feat_ind, :]
            start += feat_ind
            for i, layer in enumerate(self.fusion_layers[ind]):
                output_ind_ = layer(
                    query=output_ind,
                    key=memory_text,
                    value=memory_text,
                    query_pos=query_pos_ind,
                    key_pos=(pos_text if pos_text is not None else None),
                    attn_mask=attn_mask,
                    key_padding_mask=text_attention_mask,
                )
            output_.append(output_ind_)
        memory_text = self.fusion_layers_text(
            query=memory_text,
            key=output,
            value=output,
            query_pos=(pos_text if pos_text is not None else None),
            key_pos=query_pos,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask_image, )
        output = torch.cat(output_, dim=1)

        return output, memory_text


class TransformerDecoupleMSFusionLayer(BaseModule):
    """Implements decoder layer in DETR transformer.

    Args:
        self_attn_cfg (:obj:`ConfigDict` or dict, optional): Config for self
            attention.
        cross_attn_cfg (:obj:`ConfigDict` or dict, optional): Config for cross
            attention.
        ffn_cfg (:obj:`ConfigDict` or dict, optional): Config for FFN.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config for
            normalization layers. All the layers will share the same
            config. Defaults to `LN`.
        init_cfg (:obj:`ConfigDict` or dict, optional): Config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 cross_attn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     num_heads=8,
                     dropout=0.0,
                     batch_first=True),
                 feat_attn_layer_num=[1, 1, 1, 1],
                 ffn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True),
                 ),
                 norm_cfg: OptConfigType = dict(type='LN'),
                 init_cfg: OptConfigType = None) -> None:

        super().__init__(init_cfg=init_cfg)

        self.cross_attn_cfg = cross_attn_cfg

        if 'batch_first' not in self.cross_attn_cfg:
            self.cross_attn_cfg['batch_first'] = True
        else:
            assert self.cross_attn_cfg['batch_first'] is True, 'First \
            dimension of all DETRs in mmdet is `batch`, \
            please set `batch_first` flag.'

        self.ffn_cfg = ffn_cfg
        self.norm_cfg = norm_cfg
        self.feat_attn_layer_num = feat_attn_layer_num
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize self-attention, FFN, and normalization."""
        self.embed_dims = self.cross_attn_cfg.embed_dims
        self.fusion_layers_text = []
        self.fusion_layers = []
        for num in self.feat_attn_layer_num:
            fusion_layer = []
            fusion_layer_text = []
            for i in range(num):
                fusion_layer.append(MultiheadAttention(**self.cross_attn_cfg))
                fusion_layer_text.append(MultiheadAttention(**self.cross_attn_cfg))
                # fusion_layer.append(build_norm_layer(self.norm_cfg, self.embed_dims)[1])
                # fusion_layer.append(FFN(**self.ffn_cfg))
                # fusion_layer.append(build_norm_layer(self.norm_cfg, self.embed_dims)[1])

            self.fusion_layers.append(ModuleList(fusion_layer))
            self.fusion_layers_text.append(ModuleList(fusion_layer_text))
        self.fusion_layers = ModuleList(self.fusion_layers)
        self.fusion_layers_text = ModuleList(self.fusion_layers_text)
        self.fc = Linear(self.embed_dims * len(self.feat_attn_layer_num), self.embed_dims)
        # self.ffn = FFN(**self.ffn_cfg)
        # norms_list = [
        #     build_norm_layer(self.norm_cfg, self.embed_dims)[1]
        #     for _ in range(2)
        # ]
        # self.norms = ModuleList(norms_list)

    def forward(self,
                output: Tensor,
                memory_text: Tensor = None,
                query_pos: Tensor = None,
                pos_text: Tensor = None,
                attn_mask: Tensor = None,
                text_attention_mask: Tensor = None,
                spatial_shapes=None,
                key_padding_mask_image=None) -> Tensor:
        """
        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            key (Tensor, optional): The input key, has shape (bs, num_keys,
                dim). If `None`, the `query` will be used. Defaults to `None`.
            value (Tensor, optional): The input value, has the same shape as
                `key`, as in `nn.MultiheadAttention.forward`. If `None`, the
                `key` will be used. Defaults to `None`.
            query_pos (Tensor, optional): The positional encoding for `query`,
                has the same shape as `query`. If not `None`, it will be added
                to `query` before forward function. Defaults to `None`.
            key_pos (Tensor, optional): The positional encoding for `key`, has
                the same shape as `key`. If not `None`, it will be added to
                `key` before forward function. If None, and `query_pos` has the
                same shape as `key`, then `query_pos` will be used for
                `key_pos`. Defaults to None.
            self_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            cross_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor, optional): The `key_padding_mask` of
                `self_attn` input. ByteTensor, has shape (bs, num_value).
                Defaults to None.

        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """
        feat_inds = [(H * W).item() for (H, W) in spatial_shapes]
        output_ = []
        output_text = []
        start = 0
        for ind, feat_ind in enumerate(feat_inds):
            output_ind = output[:, start:start + feat_ind, :]
            query_pos_ind = query_pos[:, start:start + feat_ind, :]
            if key_padding_mask_image is not None:
                key_padding_mask_image_single = key_padding_mask_image[:, start:start + feat_ind]
            else:
                key_padding_mask_image_single = None
            start += feat_ind
            for i, layer in enumerate(self.fusion_layers[ind]):
                # if i % 4 == 0:
                output_ind_ = layer(
                    query=output_ind,
                    key=memory_text,
                    value=memory_text,
                    query_pos=query_pos_ind,
                    key_pos=(pos_text if pos_text is not None else None),
                    attn_mask=attn_mask,
                    key_padding_mask=text_attention_mask,
                )
                # else:
                #     output_ind = layer(output_ind)
            for j, layer_text in enumerate(self.fusion_layers_text[ind]):
                memory_text_ = layer_text(
                    query=memory_text,
                    key=output_ind,
                    value=output_ind,
                    query_pos=(pos_text if pos_text is not None else None),
                    key_pos=query_pos_ind,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask_image_single,)
            output_.append(output_ind_)
            output_text.append(memory_text_)
        output = torch.cat(output_, dim=1)
        output_text = self.fc(torch.cat(output_text, dim=-1))
        # output = self.norms[0](output)
        # output = self.ffn(output)
        # output = self.norms[1](output)

        return output, output_text

class GroundingDinoLQVGTransformerEncoder(DeformableDetrTransformerEncoder):

    def __init__(self, text_layer_cfg: ConfigType,
                 fusion_layer_cfg: ConfigType,
                 fusion_layer_ind=[0,1,2,3,4,5],
                 **kwargs) -> None:
        self.text_layer_cfg = text_layer_cfg
        self.fusion_layer_cfg = fusion_layer_cfg
        self.fusion_layer_ind = fusion_layer_ind
        self.fusion_layer_num = len(fusion_layer_ind)
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList([
            DeformableDetrTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.text_layers = ModuleList([
            DetrTransformerEncoderLayer(**self.text_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.fusion_layers = ModuleList([
            TransformerDecoupleLQVGFusionLayer(**self.fusion_layer_cfg)
            for _ in range(self.fusion_layer_num)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.num_cp > 0:
            if checkpoint_wrapper is None:
                raise NotImplementedError(
                    'If you want to reduce GPU memory usage, \
                    please install fairscale by executing the \
                    following command: pip install fairscale.')
            for i in range(self.num_cp):
                self.layers[i] = checkpoint_wrapper(self.layers[i])
            for i in range(self.fusion_layer_num):
                self.fusion_layers[i] = checkpoint_wrapper(
                    self.fusion_layers[i])

    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                key_padding_mask: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                memory_text: Tensor = None,
                text_attention_mask: Tensor = None,
                pos_text: Tensor = None,
                text_self_attention_masks: Tensor = None,
                position_ids: Tensor = None):
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
            memory_text (Tensor, optional): Memory text. It has shape (bs,
                len_text, text_embed_dims).
            text_attention_mask (Tensor, optional): Text token mask. It has
                shape (bs,len_text).
            pos_text (Tensor, optional): The positional encoding for text.
                Defaults to None.
            text_self_attention_masks (Tensor, optional): Text self attention
                mask. Defaults to None.
            position_ids (Tensor, optional): Text position ids.
                Defaults to None.
        """
        output = query
        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=query.device)
        if self.text_layers:
            # generate pos_text
            bs, n_text, _ = memory_text.shape
            if pos_text is None and position_ids is None:
                pos_text = (
                    torch.arange(n_text,
                                 device=memory_text.device).float().unsqueeze(
                                     0).unsqueeze(-1).repeat(bs, 1, 1))
                pos_text = get_text_sine_pos_embed(
                    pos_text, num_pos_feats=256, exchange_xy=False)
            if position_ids is not None:
                pos_text = get_text_sine_pos_embed(
                    position_ids[..., None],
                    num_pos_feats=256,
                    exchange_xy=False)

        # main process
        for layer_id, layer in enumerate(self.layers):
            if layer_id in self.fusion_layer_ind:
                if self.fusion_layers:
                    output, memory_text = self.fusion_layers[self.fusion_layer_ind.index(layer_id)](
                        output=output,
                        memory_text=memory_text,
                        query_pos=query_pos,
                        pos_text=(pos_text if pos_text is not None else None),
                        attn_mask=None,
                        text_attention_mask=text_attention_mask,
                        spatial_shapes=spatial_shapes,
                        key_padding_mask_image=key_padding_mask
                    )
            if self.text_layers:
                text_num_heads = self.text_layers[
                    layer_id].self_attn_cfg.num_heads
                memory_text = self.text_layers[layer_id](
                    query=memory_text,
                    query_pos=(pos_text if pos_text is not None else None),
                    attn_mask=~text_self_attention_masks.repeat(
                        text_num_heads, 1, 1),  # note we use ~ for mask here
                    key_padding_mask=None,
                )
            output = layer(
                query=output,
                query_pos=query_pos,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=key_padding_mask)
        return output, memory_text

class TransformerDecoupleLQVGFusionLayer(BaseModule):
    """Implements decoder layer in DETR transformer.

    Args:
        self_attn_cfg (:obj:`ConfigDict` or dict, optional): Config for self
            attention.
        cross_attn_cfg (:obj:`ConfigDict` or dict, optional): Config for cross
            attention.
        ffn_cfg (:obj:`ConfigDict` or dict, optional): Config for FFN.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config for
            normalization layers. All the layers will share the same
            config. Defaults to `LN`.
        init_cfg (:obj:`ConfigDict` or dict, optional): Config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 cross_attn_cfg: OptConfigType = dict(
                     embed_dims=256,
                     num_heads=8,
                     dropout=0.0,
                     ),
                 feat_attn_layer_num=[1, 1, 1, 1],
                 init_cfg: OptConfigType = None) -> None:

        super().__init__(init_cfg=init_cfg)

        self.cross_attn_cfg = cross_attn_cfg
        self.feat_attn_layer_num = feat_attn_layer_num
        if 'batch_first' not in self.cross_attn_cfg:
            self.cross_attn_cfg['batch_first'] = True
        else:
            assert self.cross_attn_cfg['batch_first'] is True, 'First \
            dimension of all DETRs in mmdet is `batch`, \
            please set `batch_first` flag.'
        self._init_layers()

    def _init_layers(self) -> None:
        """Initialize self-attention, FFN, and normalization."""
        self.embed_dims = self.cross_attn_cfg.embed_dim
        self.fusion_layers_text = []
        self.fusion_layers = []
        for num in self.feat_attn_layer_num:
            fusion_layer = []
            fusion_layer_text = []
            for i in range(num):
                fusion_layer.append(nn.MultiheadAttention(**self.cross_attn_cfg))
                fusion_layer_text.append(nn.MultiheadAttention(**self.cross_attn_cfg))

            self.fusion_layers.append(ModuleList(fusion_layer))
            self.fusion_layers_text.append(ModuleList(fusion_layer_text))
        self.fusion_layers = ModuleList(self.fusion_layers)
        self.fusion_layers_text = ModuleList(self.fusion_layers_text)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self,
                output: Tensor,
                memory_text: Tensor = None,
                query_pos: Tensor = None,
                pos_text: Tensor = None,
                attn_mask: Tensor = None,
                text_attention_mask: Tensor = None,
                spatial_shapes=None,
                key_padding_mask_image=None) -> Tensor:
        """
        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            key (Tensor, optional): The input key, has shape (bs, num_keys,
                dim). If `None`, the `query` will be used. Defaults to `None`.
            value (Tensor, optional): The input value, has the same shape as
                `key`, as in `nn.MultiheadAttention.forward`. If `None`, the
                `key` will be used. Defaults to `None`.
            query_pos (Tensor, optional): The positional encoding for `query`,
                has the same shape as `query`. If not `None`, it will be added
                to `query` before forward function. Defaults to `None`.
            key_pos (Tensor, optional): The positional encoding for `key`, has
                the same shape as `key`. If not `None`, it will be added to
                `key` before forward function. If None, and `query_pos` has the
                same shape as `key`, then `query_pos` will be used for
                `key_pos`. Defaults to None.
            self_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            cross_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor, optional): The `key_padding_mask` of
                `self_attn` input. ByteTensor, has shape (bs, num_value).
                Defaults to None.

        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """
        feat_inds = [(H * W).item() for (H, W) in spatial_shapes]
        output_ = []
        output_text = []
        start = 0
        text_initial_features = memory_text
        for ind, feat_ind in enumerate(feat_inds):
            output_ind = output[:, start:start + feat_ind, :]
            query_pos_ind = query_pos[:, start:start + feat_ind, :]
            if key_padding_mask_image is not None:
                key_padding_mask_image_single = key_padding_mask_image[:, start:start + feat_ind]
            else:
                key_padding_mask_image_single = None
            start += feat_ind
            for i, layer in enumerate(self.fusion_layers[ind]):
                output_ind_ = layer(
                    query=self.with_pos_embed(output_ind, query_pos_ind),
                    key=self.with_pos_embed(text_initial_features, pos_text),
                    value=text_initial_features,
                    attn_mask=attn_mask,
                    key_padding_mask=text_attention_mask,
                )[0]
            output_.append(output_ind_ * output_ind)
            for j, layer_text in enumerate(self.fusion_layers_text[ind]):
                memory_text_ = layer_text(
                    query=self.with_pos_embed(memory_text, pos_text),
                    key=self.with_pos_embed(output_ind, query_pos_ind),
                    value=output_ind,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask_image_single,)[0]

            memory_text = memory_text_ * memory_text
        output = torch.cat(output_, dim=1)
        # output = self.norms[0](output)
        # output = self.ffn(output)
        # output = self.norms[1](output)

        return output, memory_text
