# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.ops import MultiScaleDeformableAttention
from mmengine.model import ModuleList
from torch import Tensor

from mmdet.models.utils.vlfuse_helper import SingleScaleBiAttentionBlock, SingleScaleBiAttentionBlock_debug
from mmdet.utils import ConfigType, OptConfigType
from .deformable_detr_layers import (DeformableDetrTransformerDecoderLayer,
                                     DeformableDetrTransformerEncoder,
                                     DeformableDetrTransformerEncoderLayer)
from .sparse_layers import DeformableDetrSparseTransformerEncoderLayer
from .detr_layers import DetrTransformerEncoderLayer
from .dino_layers import DinoTransformerDecoder
from .utils import MLP, get_text_sine_pos_embed, inverse_sigmoid, coordinate_to_encoding
from .grounding_dino_layers import GroundingDinoTransformerEncoder, GroundingDinoTransformerDecoderLayer
from typing import Optional, Tuple, Union
from ..transformer.transformer_mmcv import MultiheadAttention_attn_weights
import copy
from mmdet.models.utils import draw_feature_map

try:
    from fairscale.nn.checkpoint import checkpoint_wrapper
except Exception:
    checkpoint_wrapper = None


class GroundingDinoSparseTransformerDecoderLayer(
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

class GroundingDinoLSparseTransformerEncoder_epx2(GroundingDinoTransformerEncoder):

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
                batch_data_samples = None):
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

        # sparse the language feature, find out the label feature to cross attn
        # token_positive_map = batch_data_samples[0].token_positive_map
        # label_inds = []
        # for label in token_positive_map:
        #     label_inds += token_positive_map[label]
        # label_inds = torch.tensor(label_inds, device=output.device)
        # text_attention_mask_sparse = text_attention_mask[:, label_inds]
        # main process
        for layer_id, layer in enumerate(self.layers):
            # memory_text_sparse = memory_text[:, label_inds, :]
            if layer_id == len(self.layers) - 1:
                level_start = level_start_index[0]
                level_end = level_start_index[1]
                query_inds = torch.arange(level_start, level_end).cuda(device=query.device)
            else:
                level_start = level_start_index[1]
                level_end = query.shape[1]
                query_inds = torch.arange(level_start, level_end).cuda(device=query.device)

            output_sparse = torch.gather(output, 1,
                                         query_inds[:, None].unsqueeze(0).repeat(output.shape[0], 1 ,output.shape[-1]))
            if key_padding_mask is not None:
                key_padding_mask_sparse = key_padding_mask[:, query_inds]
            else:
                key_padding_mask_sparse = None

            if self.fusion_layers:
                output_sparse, memory_text = self.fusion_layers[layer_id](
                    visual_feature=output_sparse,
                    lang_feature=memory_text,
                    attention_mask_v=key_padding_mask_sparse,
                    attention_mask_l=text_attention_mask,
                )
            output = output.scatter(1, query_inds[:, None].unsqueeze(0).repeat(output.shape[0], 1 ,output.shape[-1]),
                                    output_sparse)

            # memory_text = memory_text.scatter(1, label_inds[:, None].unsqueeze(0).repeat(memory_text.shape[0], 1, memory_text.shape[-1]),
            #                                   memory_text_sparse)

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

class GroundingDinoLSparseTransformerEncoder(GroundingDinoTransformerEncoder):

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
        self.norm = nn.LayerNorm(self.embed_dims)
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
                position_ids: Tensor = None,
                batch_data_samples = None):
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

        # sparse the language feature, find out the label feature to cross attn
        # token_positive_map = batch_data_samples[0].token_positive_map
        # label_inds = []
        # for label in token_positive_map:
        #     label_inds += token_positive_map[label]
        # label_inds = torch.tensor(label_inds, device=output.device)
        # text_attention_mask_sparse = text_attention_mask[:, label_inds]
        # main process

        for layer_id, layer in enumerate(self.layers):
            # memory_text_sparse = memory_text[:, label_inds, :]
            if layer_id == len(self.layers) - 1:
                output_sparse = output[:, :level_start_index[1]]
                query_pos_sparse = query_pos[:, :level_start_index[1]]
                reference_points_sparse = reference_points[:, :level_start_index[1]]

                if key_padding_mask is not None:
                    key_padding_mask_sparse = key_padding_mask[:, :level_start_index[1]]
                else:
                    key_padding_mask_sparse = None

                if self.fusion_layers:
                    output_sparse, memory_text = self.fusion_layers[layer_id](
                        visual_feature=output_sparse,
                        lang_feature=memory_text,
                        attention_mask_v=key_padding_mask_sparse,
                        attention_mask_l=text_attention_mask,
                    )

                output[:, :level_start_index[1]] = output_sparse

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

                output_sparse = layer(
                    query=output_sparse,
                    query_pos=query_pos_sparse,
                    memory=output,
                    reference_points=reference_points_sparse,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    key_padding_mask=key_padding_mask)

                output[:, :level_start_index[1]] = output_sparse

            else:
                output_sparse = output[:, level_start_index[1]:]
                query_pos_sparse = query_pos[:, level_start_index[1]:]
                reference_points_sparse = reference_points[:, level_start_index[1]:]

                if key_padding_mask is not None:
                    key_padding_mask_sparse = key_padding_mask[:, level_start_index[1]:]
                else:
                    key_padding_mask_sparse = None

                if self.fusion_layers:
                    output_sparse, memory_text = self.fusion_layers[layer_id](
                        visual_feature=output_sparse,
                        lang_feature=memory_text,
                        attention_mask_v=key_padding_mask_sparse,
                        attention_mask_l=text_attention_mask,
                    )

                output[:, level_start_index[1]:] = output_sparse

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

                output_sparse = layer(
                    query=output_sparse,
                    query_pos=query_pos_sparse,
                    memory=output,
                    reference_points=reference_points_sparse,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    key_padding_mask=key_padding_mask)

                output[:, level_start_index[1]:] = output_sparse

        return output, memory_text

class GroundingDinoSparseTransformerEncoder(DeformableDetrTransformerEncoder):

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
        self.norm = nn.LayerNorm(self.embed_dims)
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
                memory_pos_ori: Tensor,
                key_padding_mask: Tensor,
                key_padding_mask_ori: Tensor,
                reference_points: Tensor,
                reference_points_ori: Tensor,
                topk_indices: Tensor,
                memory: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                memory_text: Tensor = None,
                text_attention_mask: Tensor = None,
                text_token_mask: Tensor = None,
                pos_text: Tensor = None,
                text_self_attention_masks: Tensor = None,
                position_ids: Tensor = None,
                reg_branches=None,
                cls_branches=None):
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

        if reg_branches is not None:
            intermediate = []
            intermediate_reference_points = [reference_points]
            if cls_branches is not None:
                intermediate_topk_coords = []
                intermediate_topk_scores = []
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
            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * valid_ratios[:, None]
            output = layer(
                query=output,
                query_pos=query_pos,
                memory=memory,
                reference_points=reference_points_input,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=key_padding_mask_ori)
            memory = memory.scatter(1, topk_indices.unsqueeze(-1).repeat(1, 1, output.size(-1)), output)

            if reg_branches is not None and layer_id < len(self.layers) - 1:
                if cls_branches is None:
                    tmp = reg_branches[layer_id](output)
                    assert reference_points.shape[-1] == 4
                    new_reference_points = tmp + inverse_sigmoid(
                        reference_points, eps=1e-3)
                    new_reference_points = new_reference_points.sigmoid()
                    reference_points = new_reference_points.detach()

                    intermediate.append(self.norm(query))
                    intermediate_reference_points.append(new_reference_points)
                else:
                    sparse_token_nums = query.shape[1]
                    memory = self.norm(memory)
                    enc_outputs_class = cls_branches[layer_id](memory, memory_text, text_token_mask)
                    enc_outputs_coord_unact = reg_branches[layer_id](memory) + inverse_sigmoid(reference_points_ori,
                                                                                               eps=1e-3)
                    cls_out_features = enc_outputs_class.shape[-1]
                    topk_indices = torch.topk(
                        enc_outputs_class.max(-1)[0], k=sparse_token_nums, dim=1)[1]

                    topk_scores = torch.gather(
                        enc_outputs_class, 1,
                        topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
                    topk_coords_unact = torch.gather(
                        enc_outputs_coord_unact, 1,
                        topk_indices.unsqueeze(-1).repeat(1, 1, 4))
                    topk_coords = topk_coords_unact.sigmoid()
                    intermediate_topk_coords.append(topk_coords)
                    intermediate_topk_scores.append(topk_scores)

                    topk_coords_unact = topk_coords_unact.detach()
                    reference_points = topk_coords_unact
                    reference_points = reference_points.sigmoid()
                    output = torch.gather(memory, 1, topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
                    query_pos = torch.gather(memory_pos_ori, 1, topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
                    reference_points_ori = enc_outputs_coord_unact.detach().sigmoid()

        if reg_branches is not None:
            if cls_branches is None:
                output_head_dict = dict(
                    intermediate=intermediate,
                    intermediate_reference_points=intermediate_reference_points
                    )
                return output, memory, reference_points, reference_points_ori, memory_text, output_head_dict
            else:
                output_head_dict = dict(
                    intermediate_topk_scores=intermediate_topk_scores,
                    intermediate_topk_coords=intermediate_topk_coords
                )
                return output, memory, reference_points, reference_points_ori, memory_text, output_head_dict
        else:
            return output, memory, reference_points, reference_points_ori, memory_text, None


class GroundingDinoSparseTransformerDecoder(DinoTransformerDecoder):

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            GroundingDinoSparseTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')
        self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                  self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)

class GroundingDinoLmatchTransformerEncoder(DeformableDetrTransformerEncoder):

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


class GroundingDinoLmatchTransformerDecoder(DinoTransformerDecoder):

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            GroundingDinoLmatchTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')
        self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                  self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)

    def forward(self, query: Tensor, value: Tensor, key_padding_mask: Tensor,
                self_attn_mask: Tensor, reference_points: Tensor,
                spatial_shapes: Tensor, level_start_index: Tensor,
                valid_ratios: Tensor, reg_branches: nn.ModuleList,
                position_ids: Tensor, query_mask: Tensor,
                **kwargs) -> Tuple[Tensor]:
        """Forward function of Transformer decoder.

        Args:
            query (Tensor): The input query, has shape (num_queries, bs, dim).
            value (Tensor): The input values, has shape (num_value, bs, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (num_queries, bs).
            self_attn_mask (Tensor): The attention mask to prevent information
                leakage from different denoising groups and matching parts, has
                shape (num_queries_total, num_queries_total). It is `None` when
                `self.training` is `False`.
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            reg_branches: (obj:`nn.ModuleList`): Used for refining the
                regression results.

        Returns:
            tuple[Tensor]: Output queries and references of Transformer
                decoder

            - query (Tensor): Output embeddings of the last decoder, has
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

        intermediate = []
        intermediate_text = []
        intermediate_reference_points = [reference_points]
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * valid_ratios[:, None]

            query_sine_embed = coordinate_to_encoding(
                reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)

            query, memory_text = layer(
                query,
                query_pos=query_pos,
                value=value,
                key_padding_mask=key_padding_mask,
                self_attn_mask=self_attn_mask,
                query_mask=query_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input,
                **kwargs)

            if reg_branches is not None:
                tmp = reg_branches[lid](query)
                assert reference_points.shape[-1] == 4
                new_reference_points = tmp + inverse_sigmoid(
                    reference_points, eps=1e-3)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(query))
                intermediate_text.append(self.norm(memory_text))
                intermediate_reference_points.append(new_reference_points)
                # NOTE this is for the "Look Forward Twice" module,
                # in the DeformDETR, reference_points was appended.

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_text), \
                   torch.stack(intermediate_reference_points)

        return query, memory_text, reference_points

class GroundingDinoLmatchTransformerDecoderLayer(
        DeformableDetrTransformerDecoderLayer):

    def __init__(self,
                 # cross_attn_text_cfg: OptConfigType = dict(
                 #     embed_dims=256,
                 #     num_heads=8,
                 #     dropout=0.0,
                 #     batch_first=True),
                 fusion_layer_cfg = None,
                 **kwargs) -> None:
        """Decoder layer of Deformable DETR."""
        self.fusion_layer_cfg = fusion_layer_cfg
        # self.cross_attn_text_cfg = cross_attn_text_cfg
        # if 'batch_first' not in self.cross_attn_text_cfg:
        #     self.cross_attn_text_cfg['batch_first'] = True
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize self_attn, cross-attn, ffn, and norms."""
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        # self.cross_attn_text = MultiheadAttention(**self.cross_attn_text_cfg)
        self.cross_attn = MultiScaleDeformableAttention(**self.cross_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(4)
        ]
        self.norms = ModuleList(norms_list)
        self.fusion = SingleScaleBiAttentionBlock(**self.fusion_layer_cfg)


    def forward(self,
                query: Tensor,
                key: Tensor = None,
                value: Tensor = None,
                query_pos: Tensor = None,
                key_pos: Tensor = None,
                query_mask = None,
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
        query, memory_text = self.fusion(
            visual_feature=query,
            lang_feature=memory_text,
            attention_mask_v=query_mask,
            attention_mask_l=text_attention_mask,
        )
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

        return query, memory_text

class GroundingDinoLscoreTransformerEncoder(DeformableDetrTransformerEncoder):

    def __init__(self, sparse_thres, lowest_rate, text_layer_cfg: ConfigType,
                 fusion_layer_cfg: ConfigType, record_txt=None, **kwargs) -> None:
        self.sparse_thres = sparse_thres
        self.lowest_rate = lowest_rate
        self.text_layer_cfg = text_layer_cfg
        self.fusion_layer_cfg = fusion_layer_cfg
        self.record_txt = record_txt
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

        # self.memory_sparse_trans_fc = ModuleList([nn.Linear(self.embed_dims, self.embed_dims)
        #                                           for _ in range(self.num_layers)])
        # self.memory_sparse_trans_norm = ModuleList([nn.LayerNorm(self.embed_dims)
        #                                             for _ in range(self.num_layers)])

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
                cls_branches = None,
                reg_branches = None
                ):
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

        output_proposals, output_proposals_valid = self.gen_encoder_output_proposals(
            output, key_padding_mask, spatial_shapes)
        intermediate_outputs_class = []
        intermediate_outputs_coord = []
        # main process
        for layer_id, layer in enumerate(self.layers):
            if layer_id < 3:
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

                output_memory = output
                if key_padding_mask is not None:
                    output_memory = output_memory.masked_fill(
                        key_padding_mask.unsqueeze(-1), float(0))
                output_memory = output_memory.masked_fill(~output_proposals_valid,
                                                          float(0))
                # output_memory = self.memory_sparse_trans_fc[layer_id](output_memory)
                # output_memory = self.memory_sparse_trans_norm[layer_id](output_memory)
                enc_outputs_class = cls_branches[layer_id](output_memory, memory_text,
                                                           ~text_attention_mask)
                enc_outputs_coord_unact = reg_branches[layer_id](output_memory) + output_proposals
                enc_outputs_coords = enc_outputs_coord_unact.sigmoid()

                intermediate_outputs_class.append(enc_outputs_class)
                intermediate_outputs_coord.append(enc_outputs_coords)

                if layer_id == 2:
                    bs, token_number, c = output.shape
                    self.lowest_token_nums = int(token_number * self.lowest_rate) + 1

                    scores = enc_outputs_class.sigmoid()
                    scores_p = scores.max(-1)[0]
                    score_sort, indices = scores_p.sort(dim=1, descending=True)

                    if bs > 1:
                        bs_token_number = 0
                        for i in range(bs):
                            sparse_number_single = score_sort[i][score_sort[i] > self.sparse_thres].shape[0]
                            if sparse_number_single > bs_token_number:
                                bs_token_number = sparse_number_single
                    else:
                        bs_token_number = score_sort[score_sort > self.sparse_thres].shape[0]

                    if self.lowest_token_nums > bs_token_number:
                        bs_token_number = self.lowest_token_nums
                    if self.record_txt is not None:
                        with open('./work_dirs/record_txt/' + self.record_txt, 'a') as f:
                            f.write(str(round(bs_token_number/token_number, 4)) + '\n')

                    topk_indices = indices[:, :bs_token_number]
                    output_proposals_sparse = torch.gather(output_proposals, 1,
                                                           topk_indices.unsqueeze(-1).repeat(1, 1, 4))
                    output_proposals_valid_sparse = torch.gather(output_proposals_valid, 1,
                                                           topk_indices.unsqueeze(-1))

            else:
                output_sparse = torch.gather(output, 1, topk_indices.unsqueeze(-1).repeat(1, 1, output.shape[-1]))
                if key_padding_mask is not None:
                    key_padding_mask_sparse = torch.gather(key_padding_mask, 1, topk_indices)
                else:
                    key_padding_mask_sparse = None

                if self.fusion_layers:
                    output_sparse, memory_text = self.fusion_layers[layer_id](
                        visual_feature=output_sparse,
                        lang_feature=memory_text,
                        attention_mask_v=key_padding_mask_sparse,
                        attention_mask_l=text_attention_mask,
                    )

                output = output.scatter(1, topk_indices.unsqueeze(-1).repeat(1, 1, output.shape[-1]), output_sparse)

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

                output_memory = output_sparse
                if key_padding_mask_sparse is not None:
                    output_memory = output_memory.masked_fill(
                        key_padding_mask_sparse.unsqueeze(-1), float(0))
                output_memory = output_memory.masked_fill(~output_proposals_valid_sparse,
                                                          float(0))
                # output_memory = self.memory_sparse_trans_fc[layer_id](output_memory)
                # output_memory = self.memory_sparse_trans_norm[layer_id](output_memory)
                topk_score = cls_branches[layer_id](output_memory, memory_text,
                                     ~text_attention_mask)
                topk_coords = reg_branches[layer_id](output_memory) + output_proposals_sparse
                topk_coords = topk_coords.sigmoid()
                intermediate_outputs_class.append(topk_score)
                intermediate_outputs_coord.append(topk_coords)

        return output, memory_text, intermediate_outputs_class, intermediate_outputs_coord

    # def process_mid(
    #     self,
    #     output_memory: Tensor,
    #     memory_text: Tensor,
    #     text_token_mask: Tensor,
    #     output_proposals,
    #     select_mask,
    #     cls_branch,
    #     reg_branch
    # ) -> Tuple[Dict]:
    #     bs, token_number, c = output_memory.shape
    #     self.lowest_token_nums = int(token_number * self.lowest_rate) + 1
    #
    #     enc_outputs_class = cls_branch(output_memory, memory_text,
    #                                  text_token_mask)
    #     cls_out_features = cls_branch.max_text_len
    #     enc_outputs_coord_unact = reg_branch(output_memory) + output_proposals
    #
    #     # NOTE The DINO selects top-k proposals according to scores of
    #     # multi-class classification, while DeformDETR, where the input
    #     # is `enc_outputs_class[..., 0]` selects according to scores of
    #     # binary classification.
    #     scores = enc_outputs_class.sigmoid()
    #     scores_p = scores.max(-1)[0]
    #     score_sort, indices = scores_p.sort(dim=1, descending=True)
    #
    #     if bs > 1:
    #         bs_token_number = 0
    #         for i in range(bs):
    #             sparse_number_single = score_sort[i][score_sort[i] > self.sparse_thres].shape[0]
    #             if sparse_number_single > bs_token_number:
    #                 bs_token_number = sparse_number_single
    #     else:
    #         bs_token_number = score_sort[score_sort > self.sparse_thres].shape[0]
    #
    #     topk_indices = indices[:, :bs_token_number]
    #     topk_score = torch.gather(
    #         enc_outputs_class, 1,
    #         topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
    #     topk_coords_unact = torch.gather(
    #         enc_outputs_coord_unact, 1,
    #         topk_indices.unsqueeze(-1).repeat(1, 1, 4))
    #     topk_coords = topk_coords_unact.sigmoid()
    #     select_mask = select_mask.scatter(1, topk_indices, 1)
    #
    #     return topk_indices, select_mask, topk_score, topk_coords

    # def pre_encoder(
    #     self,
    #     memory: Tensor,
    #     memory_mask: Tensor,
    #     spatial_shapes: Tensor,
    #         output_proposals,
    #         output_proposals_valid,
    #     memory_text: Tensor,
    #     text_token_mask: Tensor,
    #     cls_branch,
    #     reg_branch
    # ) -> Tuple[Dict]:
    #     bs, token_number, c = memory.shape
    #     self.lowest_token_nums = int(token_number * self.lowest_rate) + 1
    #
    #
    #     enc_outputs_class = cls_branch(output_memory, memory_text,
    #                                  text_token_mask)
    #     cls_out_features = cls_branch.max_text_len
    #     enc_outputs_coord_unact = reg_branch(output_memory) + output_proposals
    #
    #     # NOTE The DINO selects top-k proposals according to scores of
    #     # multi-class classification, while DeformDETR, where the input
    #     # is `enc_outputs_class[..., 0]` selects according to scores of
    #     # binary classification.
    #     scores = enc_outputs_class.sigmoid()
    #     scores_p = scores.max(-1)[0]
    #     score_sort, indices = scores_p.sort(dim=1, descending=True)
    #
    #     if bs > 1:
    #         bs_token_number = 0
    #         for i in range(bs):
    #             sparse_number_single = score_sort[i][score_sort[i] > self.sparse_thres].shape[0]
    #             if sparse_number_single > bs_token_number:
    #                 bs_token_number = sparse_number_single
    #     else:
    #         bs_token_number = score_sort[score_sort > self.sparse_thres].shape[0]
    #
    #     topk_indices = indices[:, :bs_token_number]
    #     topk_score = torch.gather(
    #         enc_outputs_class, 1,
    #         topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
    #     topk_coords_unact = torch.gather(
    #         enc_outputs_coord_unact, 1,
    #         topk_indices.unsqueeze(-1).repeat(1, 1, 4))
    #     topk_coords = topk_coords_unact.sigmoid()
    #
    #     select_mask = torch.zeros((bs, token_number), device=memory.device)
    #     select_mask = select_mask.scatter(1, topk_indices, 1)
    #
    #     return output_proposals, topk_indices, select_mask, topk_score, topk_coords

    def gen_encoder_output_proposals(
            self, memory: Tensor, memory_mask: Tensor,
            spatial_shapes: Tensor) -> Tuple[Tensor, Tensor]:
        """Generate proposals from encoded memory. The function will only be
        used when `as_two_stage` is `True`.

        Args:
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).

        Returns:
            tuple: A tuple of transformed memory and proposals.

            - output_memory (Tensor): The transformed memory for obtaining
              top-k proposals, has shape (bs, num_feat_points, dim).
            - output_proposals (Tensor): The inverse-normalized proposal, has
              shape (batch_size, num_keys, 4) with the last dimension arranged
              as (cx, cy, w, h).
        """

        bs = memory.size(0)
        proposals = []
        _cur = 0  # start index in the sequence of the current level
        for lvl, HW in enumerate(spatial_shapes):
            H, W = HW

            if memory_mask is not None:
                mask_flatten_ = memory_mask[:, _cur:(_cur + H * W)].view(
                    bs, H, W, 1)
                valid_H = torch.sum(~mask_flatten_[:, :, 0, 0],
                                    1).unsqueeze(-1)
                valid_W = torch.sum(~mask_flatten_[:, 0, :, 0],
                                    1).unsqueeze(-1)
                scale = torch.cat([valid_W, valid_H], 1).view(bs, 1, 1, 2)
            else:
                if not isinstance(HW, torch.Tensor):
                    HW = memory.new_tensor(HW)
                scale = HW.unsqueeze(0).flip(dims=[0, 1]).view(1, 1, 1, 2)
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(
                    0, H - 1, H, dtype=torch.float32, device=memory.device),
                torch.linspace(
                    0, W - 1, W, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
            grid = (grid.unsqueeze(0).expand(bs, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)
            proposal = torch.cat((grid, wh), -1).view(bs, -1, 4)
            proposals.append(proposal)
            _cur += (H * W)
        output_proposals = torch.cat(proposals, 1)
        # do not use `all` to make it exportable to onnx
        output_proposals_valid = (
            (output_proposals > 0.01) & (output_proposals < 0.99)).sum(
                -1, keepdim=True) == output_proposals.shape[-1]
        # inverse_sigmoid
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        if memory_mask is not None:
            output_proposals = output_proposals.masked_fill(
                memory_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(
            ~output_proposals_valid, float('inf'))

        # [bs, sum(hw), 2]
        return output_proposals, output_proposals_valid

class GroundingDinoLqueryTransformerEncoder(DeformableDetrTransformerEncoder):

    def __init__(self, text_layer_cfg: ConfigType,
                 fusion_layer_cfg: ConfigType,
                 return_intermediate, **kwargs) -> None:
        self.text_layer_cfg = text_layer_cfg
        self.fusion_layer_cfg = fusion_layer_cfg
        self.return_intermediate = return_intermediate
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
        intermediate_v = []
        intermediate_l = []
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
            if self.return_intermediate:
                intermediate_v.append(output)
                intermediate_l.append(memory_text)
        if self.return_intermediate:
            return intermediate_v, intermediate_l
        else:
            return output, memory_text

class GroundingDinoNofusionTransformerDecoderLayer(
        DeformableDetrTransformerDecoderLayer):

    def __init__(self,
                 # cross_attn_text_cfg: OptConfigType = dict(
                 #     embed_dims=256,
                 #     num_heads=8,
                 #     dropout=0.0,
                 #     batch_first=True),
                 **kwargs) -> None:
        """Decoder layer of Deformable DETR."""
        # self.cross_attn_text_cfg = cross_attn_text_cfg
        # if 'batch_first' not in self.cross_attn_text_cfg:
        #     self.cross_attn_text_cfg['batch_first'] = True
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize self_attn, cross-attn, ffn, and norms."""
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        # self.cross_attn_text = MultiheadAttention(**self.cross_attn_text_cfg)
        self.cross_attn = MultiScaleDeformableAttention(**self.cross_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(3)
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
        # query = self.cross_attn_text(
        #     query=query,
        #     query_pos=query_pos,
        #     key=memory_text,
        #     value=memory_text,
        #     key_padding_mask=text_attention_mask)
        # query = self.norms[1](query)
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
        query = self.norms[1](query)
        query = self.ffn(query)
        query = self.norms[2](query)

        return query


class GroundingDinoNofusionTransformerEncoder(DeformableDetrTransformerEncoder):

    def __init__(self,
                 # text_layer_cfg: ConfigType,
                 # fusion_layer_cfg: ConfigType,
                 **kwargs) -> None:
        # self.text_layer_cfg = text_layer_cfg
        # self.fusion_layer_cfg = fusion_layer_cfg
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList([
            DeformableDetrTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        # self.text_layers = ModuleList([
        #     DetrTransformerEncoderLayer(**self.text_layer_cfg)
        #     for _ in range(self.num_layers)
        # ])
        # self.fusion_layers = ModuleList([
        #     SingleScaleBiAttentionBlock(**self.fusion_layer_cfg)
        #     for _ in range(self.num_layers)
        # ])
        self.embed_dims = self.layers[0].embed_dims
        if self.num_cp > 0:
            if checkpoint_wrapper is None:
                raise NotImplementedError(
                    'If you want to reduce GPU memory usage, \
                    please install fairscale by executing the \
                    following command: pip install fairscale.')
            for i in range(self.num_cp):
                self.layers[i] = checkpoint_wrapper(self.layers[i])
                # self.fusion_layers[i] = checkpoint_wrapper(
                #     self.fusion_layers[i])

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
        # if self.text_layers:
        #     # generate pos_text
        #     bs, n_text, _ = memory_text.shape
        #     if pos_text is None and position_ids is None:
        #         pos_text = (
        #             torch.arange(n_text,
        #                          device=memory_text.device).float().unsqueeze(
        #                              0).unsqueeze(-1).repeat(bs, 1, 1))
        #         pos_text = get_text_sine_pos_embed(
        #             pos_text, num_pos_feats=256, exchange_xy=False)
        #     if position_ids is not None:
        #         pos_text = get_text_sine_pos_embed(
        #             position_ids[..., None],
        #             num_pos_feats=256,
        #             exchange_xy=False)

        # main process
        for layer_id, layer in enumerate(self.layers):
            # if self.fusion_layers:
            #     output, memory_text = self.fusion_layers[layer_id](
            #         visual_feature=output,
            #         lang_feature=memory_text,
            #         attention_mask_v=key_padding_mask,
            #         attention_mask_l=text_attention_mask,
            #     )
            # if self.text_layers:
            #     text_num_heads = self.text_layers[
            #         layer_id].self_attn_cfg.num_heads
            #     memory_text = self.text_layers[layer_id](
            #         query=memory_text,
            #         query_pos=(pos_text if pos_text is not None else None),
            #         attn_mask=~text_self_attention_masks.repeat(
            #             text_num_heads, 1, 1),  # note we use ~ for mask here
            #         key_padding_mask=None,
            #     )
            output = layer(
                query=output,
                query_pos=query_pos,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=key_padding_mask)
        return output, memory_text


class GroundingDinoNofusionTransformerDecoder(DinoTransformerDecoder):

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            GroundingDinoNofusionTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')
        self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                  self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)

class GroundingDinoTransformerEncoder_debug(DeformableDetrTransformerEncoder):

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
            SingleScaleBiAttentionBlock_debug(**self.fusion_layer_cfg)
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
        attn_weights_list = []
        # main process
        for layer_id, layer in enumerate(self.layers):
            if self.fusion_layers:
                output, memory_text, attn_weights = self.fusion_layers[layer_id](
                    visual_feature=output,
                    lang_feature=memory_text,
                    attention_mask_v=key_padding_mask,
                    attention_mask_l=text_attention_mask,
                )
                attn_weights_list.append(attn_weights)
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
        return output, memory_text, attn_weights_list

class GroundingDino_attn_TransformerDecoderLayer(
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
        self.cross_attn_text = MultiheadAttention_attn_weights(**self.cross_attn_text_cfg)
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
        query, attn_weights = self.cross_attn_text(
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

        return query, attn_weights


class GroundingDino_attn_TransformerDecoder(DinoTransformerDecoder):

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            GroundingDino_attn_TransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')
        self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                  self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)

    def forward(self, query: Tensor, value: Tensor, key_padding_mask: Tensor,
                self_attn_mask: Tensor, reference_points: Tensor,
                spatial_shapes: Tensor, level_start_index: Tensor,
                valid_ratios: Tensor, reg_branches: nn.ModuleList,
                **kwargs) -> Tuple[Tensor]:
        """Forward function of Transformer decoder.

        Args:
            query (Tensor): The input query, has shape (num_queries, bs, dim).
            value (Tensor): The input values, has shape (num_value, bs, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (num_queries, bs).
            self_attn_mask (Tensor): The attention mask to prevent information
                leakage from different denoising groups and matching parts, has
                shape (num_queries_total, num_queries_total). It is `None` when
                `self.training` is `False`.
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            reg_branches: (obj:`nn.ModuleList`): Used for refining the
                regression results.

        Returns:
            tuple[Tensor]: Output queries and references of Transformer
                decoder

            - query (Tensor): Output embeddings of the last decoder, has
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
        intermediate = []
        attn_weights_list = []
        intermediate_reference_points = [reference_points]
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * valid_ratios[:, None]

            query_sine_embed = coordinate_to_encoding(
                reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)

            query, attn_weights = layer(
                query,
                query_pos=query_pos,
                value=value,
                key_padding_mask=key_padding_mask,
                self_attn_mask=self_attn_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input,
                **kwargs)

            if reg_branches is not None:
                tmp = reg_branches[lid](query)
                assert reference_points.shape[-1] == 4
                new_reference_points = tmp + inverse_sigmoid(
                    reference_points, eps=1e-3)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(query))
                intermediate_reference_points.append(new_reference_points)
                attn_weights_list.append(attn_weights)
                # NOTE this is for the "Look Forward Twice" module,
                # in the DeformDETR, reference_points was appended.

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points), torch.stack(attn_weights_list)

        return query, reference_points, attn_weights

class GroundingDinoLadaptTransformerDecoder(DinoTransformerDecoder):

    def __init__(self, num_queries, Lang_adapt_cfg, **kwargs):
        self.num_queries = num_queries
        self.Lang_adapt_cfg = Lang_adapt_cfg
        if 'batch_first' not in self.Lang_adapt_cfg:
            self.Lang_adapt_cfg['batch_first'] = True
        super().__init__(**kwargs)



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
        self.norm_lang = nn.LayerNorm(self.embed_dims)
        self.Lang_content_query = nn.ModuleList([
            copy.deepcopy(nn.Embedding(self.num_queries, self.embed_dims))
            for _ in range(self.num_layers)
        ])
        for m in self.Lang_content_query.parameters():
            nn.init.zeros_(m)

        self.lang_query_trans = nn.ModuleList([
            copy.deepcopy(nn.Linear(self.embed_dims, self.embed_dims * 2))
            for _ in range(self.num_layers)
        ])

        # self.query_pos_trans = nn.ModuleList([
        #     copy.deepcopy(nn.Linear(self.embed_dims * 2, self.embed_dims))
        #     for _ in range(self.num_layers)
        # ])

        self.Lang_adapt_attns = nn.ModuleList([
            copy.deepcopy(MultiheadAttention(**self.Lang_adapt_cfg))
            for _ in range(self.num_layers)
        ])

        self.query_trans = nn.ModuleList([
            copy.deepcopy(nn.Linear(self.embed_dims * 2, self.embed_dims))
            for _ in range(self.num_layers)
        ])

    def forward(self, query: Tensor, value: Tensor, key_padding_mask: Tensor,
                self_attn_mask: Tensor, reference_points: Tensor,
                spatial_shapes: Tensor, level_start_index: Tensor,
                valid_ratios: Tensor, reg_branches: nn.ModuleList,
                **kwargs) -> Tuple[Tensor]:
        """Forward function of Transformer decoder.

        Args:
            query (Tensor): The input query, has shape (num_queries, bs, dim).
            value (Tensor): The input values, has shape (num_value, bs, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (num_queries, bs).
            self_attn_mask (Tensor): The attention mask to prevent information
                leakage from different denoising groups and matching parts, has
                shape (num_queries_total, num_queries_total). It is `None` when
                `self.training` is `False`.
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            reg_branches: (obj:`nn.ModuleList`): Used for refining the
                regression results.

        Returns:
            tuple[Tensor]: Output queries and references of Transformer
                decoder

            - query (Tensor): Output embeddings of the last decoder, has
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
        intermediate = []
        intermediate_lang = []
        intermediate_reference_points = [reference_points]
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * valid_ratios[:, None]

            query_sine_embed = coordinate_to_encoding(
                reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)

            # calculate the language feature adapt query
            lang_adapt_feat = self.Lang_content_query[lid].weight[None].repeat(query.shape[0], 1, 1)
            lang_adapt_feat = self.lang_query_trans[lid](lang_adapt_feat)
            lang_adapt_query_pos, lang_adapt_query = torch.split(lang_adapt_feat, self.embed_dims, dim=2)
            lang_adapt_query = self.Lang_adapt_attns[lid](
                query=lang_adapt_query,
                query_pos=lang_adapt_query_pos,
                key=kwargs['memory_text'],
                value=kwargs['memory_text'],
                key_padding_mask=kwargs['text_attention_mask'])
            query_ori = query[:, -self.num_queries:, :]
            query_dn = query[:, :query.shape[1] - self.num_queries, :]
            query_new = self.query_trans[lid](torch.cat((lang_adapt_query, query_ori), dim=-1))
            query = torch.cat((query_dn, query_new), dim=1)

            # query_pos_ori = query_pos[:, -self.num_queries:, :]
            # query_pos_dn = query_pos[:, :query.shape[1] - self.num_queries, :]
            # query_pos_ori = self.query_pos_trans[lid](torch.cat((lang_adapt_query_pos, query_pos_ori), dim=-1))
            # query_pos = torch.cat((query_pos_dn, query_pos_ori), dim=1)
            #############################################
            query = layer(
                query,
                query_pos=query_pos,
                value=value,
                key_padding_mask=key_padding_mask,
                self_attn_mask=self_attn_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input,
                **kwargs)

            if reg_branches is not None:
                tmp = reg_branches[lid](query)
                assert reference_points.shape[-1] == 4
                new_reference_points = tmp + inverse_sigmoid(
                    reference_points, eps=1e-3)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(query))
                intermediate_lang.append(self.norm_lang(lang_adapt_query))
                intermediate_reference_points.append(new_reference_points)
                # NOTE this is for the "Look Forward Twice" module,
                # in the DeformDETR, reference_points was appended.

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_lang), torch.stack(
                intermediate_reference_points)

        return query, lang_adapt_query, reference_points


class GroundingDinoLaugmentTransformerEncoder(DeformableDetrTransformerEncoder):

    def __init__(self,
                 # text_layer_cfg: ConfigType,
                 fusion_layer_cfg: ConfigType,
                 norm_cfg: OptConfigType = dict(type='LN'),
                 **kwargs) -> None:
        self.fusion_layer_cfg = fusion_layer_cfg
        if 'batch_first' not in self.fusion_layer_cfg:
            self.fusion_layer_cfg['batch_first'] = True
        self.norm_cfg = norm_cfg
        # self.text_layer_cfg = text_layer_cfg
        # self.fusion_layer_cfg = fusion_layer_cfg
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize encoder layers."""
        self.layers = ModuleList([
            DeformableDetrTransformerEncoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        # self.text_layers = ModuleList([
        #     DetrTransformerEncoderLayer(**self.text_layer_cfg)
        #     for _ in range(self.num_layers)
        # ])
        self.fusion_layers = ModuleList([
            MultiheadAttention(**self.fusion_layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(self.num_layers)
        ]
        self.norms = ModuleList(norms_list)

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

        # if self.text_layers:
        #     # generate pos_text
        #     bs, n_text, _ = memory_text.shape
        #     if pos_text is None and position_ids is None:
        #         pos_text = (
        #             torch.arange(n_text,
        #                          device=memory_text.device).float().unsqueeze(
        #                              0).unsqueeze(-1).repeat(bs, 1, 1))
        #         pos_text = get_text_sine_pos_embed(
        #             pos_text, num_pos_feats=256, exchange_xy=False)
        #     if position_ids is not None:
        #         pos_text = get_text_sine_pos_embed(
        #             position_ids[..., None],
        #             num_pos_feats=256,
        #             exchange_xy=False)

        # main process
        bs_L, token_num, c = memory_text.shape
        bs_v = output.shape[0]
        memory_text = torch.cat(memory_text.split(bs_v, dim=0), dim=1)
        text_attention_mask = torch.cat(text_attention_mask.split(bs_v, dim=0), dim=1)
        for layer_id, layer in enumerate(self.layers):
            if self.fusion_layers:
                output = self.fusion_layers[layer_id](
                    query=output,
                    key=memory_text,
                    value=memory_text,
                    query_pos=query_pos,
                    key_padding_mask=text_attention_mask,)
                output = self.norms[layer_id](output)
            # if self.text_layers:
            #     text_num_heads = self.text_layers[
            #         layer_id].self_attn_cfg.num_heads
            #     memory_text = self.text_layers[layer_id](
            #         query=memory_text,
            #         query_pos=(pos_text if pos_text is not None else None),
            #         attn_mask=~text_self_attention_masks.repeat(
            #             text_num_heads, 1, 1),  # note we use ~ for mask here
            #         key_padding_mask=None,
            #     )
            output = layer(
                query=output,
                query_pos=query_pos,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                key_padding_mask=key_padding_mask)
        memory_text = torch.cat(memory_text.split(token_num, dim=1), 0)
        return output, memory_text

class GroundingDinoLaugmentTransformerDecoderLayer(
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
        bs_L, token_num, c = query.shape
        bs_v = value.shape[0]
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
        if self.training:
            query = torch.cat(query.split(bs_v, dim=0), dim=1)
            query_pos = torch.cat(query_pos.split(bs_v, dim=0), dim=1)
            kwargs['reference_points'] = torch.cat(kwargs['reference_points'].split(bs_v, dim=0), dim=1)
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
        if self.training:
            query = torch.cat(query.split(token_num, dim=1), 0)
        return query


class GroundingDinoLaugmentTransformerDecoder(DinoTransformerDecoder):

    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            GroundingDinoLaugmentTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')
        self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                  self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)

    def forward(self, query: Tensor, value: Tensor, key_padding_mask: Tensor,
                self_attn_mask: Tensor, reference_points: Tensor,
                spatial_shapes: Tensor, level_start_index: Tensor,
                valid_ratios: Tensor, reg_branches: nn.ModuleList,
                **kwargs) -> Tuple[Tensor]:
        """Forward function of Transformer decoder.

        Args:
            query (Tensor): The input query, has shape (num_queries, bs, dim).
            value (Tensor): The input values, has shape (num_value, bs, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (num_queries, bs).
            self_attn_mask (Tensor): The attention mask to prevent information
                leakage from different denoising groups and matching parts, has
                shape (num_queries_total, num_queries_total). It is `None` when
                `self.training` is `False`.
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            reg_branches: (obj:`nn.ModuleList`): Used for refining the
                regression results.

        Returns:
            tuple[Tensor]: Output queries and references of Transformer
                decoder

            - query (Tensor): Output embeddings of the last decoder, has
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
        intermediate = []
        intermediate_reference_points = [reference_points]
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * valid_ratios[:, None]

            query_sine_embed = coordinate_to_encoding(
                reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)

            query = layer(
                query,
                query_pos=query_pos,
                value=value,
                key_padding_mask=key_padding_mask,
                self_attn_mask=self_attn_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input,
                **kwargs)

            if reg_branches is not None:
                tmp = reg_branches[lid](query)
                assert reference_points.shape[-1] == 4
                new_reference_points = tmp + inverse_sigmoid(
                    reference_points, eps=1e-3)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(query))
                intermediate_reference_points.append(new_reference_points)
                # NOTE this is for the "Look Forward Twice" module,
                # in the DeformDETR, reference_points was appended.

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return query, reference_points


class GroundingDinoDecoupleTransformerDecoder(DinoTransformerDecoder):


    def _init_layers(self) -> None:
        """Initialize decoder layers."""
        self.layers = ModuleList([
            GroundingDinoDecoupleTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError('There is not post_norm in '
                             f'{self._get_name()}')
        self.ref_point_head = MLP(self.embed_dims * 2, self.embed_dims,
                                  self.embed_dims, 2)
        self.norm1 = nn.LayerNorm(self.embed_dims)
        self.norm2 = nn.LayerNorm(self.embed_dims)

    def forward(self, query: Tensor, query_text: Tensor, query_text_pos: Tensor,
                value: Tensor, key_padding_mask: Tensor,
                self_attn_mask: Tensor, reference_points: Tensor,
                spatial_shapes: Tensor, level_start_index: Tensor,
                valid_ratios: Tensor, reg_branches: nn.ModuleList,
                **kwargs) -> Tuple[Tensor]:
        """Forward function of Transformer decoder.

        Args:
            query (Tensor): The input query, has shape (num_queries, bs, dim).
            value (Tensor): The input values, has shape (num_value, bs, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (num_queries, bs).
            self_attn_mask (Tensor): The attention mask to prevent information
                leakage from different denoising groups and matching parts, has
                shape (num_queries_total, num_queries_total). It is `None` when
                `self.training` is `False`.
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            reg_branches: (obj:`nn.ModuleList`): Used for refining the
                regression results.

        Returns:
            tuple[Tensor]: Output queries and references of Transformer
                decoder

            - query (Tensor): Output embeddings of the last decoder, has
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
        intermediate = []
        intermediate_text = []
        intermediate_reference_points = [reference_points]
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = \
                    reference_points[:, :, None] * torch.cat(
                        [valid_ratios, valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = \
                    reference_points[:, :, None] * valid_ratios[:, None]

            query_sine_embed = coordinate_to_encoding(
                reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)

            query, query_text = layer(
                query,
                query_text,
                query_pos=query_pos,
                query_text_pos=query_text_pos,
                value=value,
                key_padding_mask=key_padding_mask,
                self_attn_mask=self_attn_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input,
                **kwargs)

            if reg_branches is not None:
                tmp = reg_branches[lid](query)
                assert reference_points.shape[-1] == 4
                new_reference_points = tmp + inverse_sigmoid(
                    reference_points, eps=1e-3)
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(self.norm1(query))
                intermediate_text.append(self.norm2(query_text))
                intermediate_reference_points.append(new_reference_points)
                # NOTE this is for the "Look Forward Twice" module,
                # in the DeformDETR, reference_points was appended.

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_text), torch.stack(
                intermediate_reference_points)

        return query, query_text, reference_points


class GroundingDinoDecoupleTransformerDecoderLayer(
        DeformableDetrTransformerDecoderLayer):

    def __init__(self,
                 fusion_layer_cfg :OptConfigType,
                 cross_attn_text_cfg: OptConfigType = dict(
                     embed_dims=256,
                     num_heads=8,
                     dropout=0.0,
                     batch_first=True),

                 **kwargs) -> None:
        """Decoder layer of Deformable DETR."""
        self.fusion_layer_cfg = fusion_layer_cfg
        self.cross_attn_text_cfg = cross_attn_text_cfg
        if 'batch_first' not in self.cross_attn_text_cfg:
            self.cross_attn_text_cfg['batch_first'] = True
        super().__init__(**kwargs)

    def _init_layers(self) -> None:
        """Initialize self_attn, cross-attn, ffn, and norms."""
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.self_attn_text = MultiheadAttention(**self.self_attn_cfg)
        self.cross_attn_text = MultiheadAttention(**self.cross_attn_text_cfg)
        self.cross_attn_text_text = MultiheadAttention(**self.cross_attn_text_cfg)
        self.cross_attn = MultiScaleDeformableAttention(**self.cross_attn_cfg)
        self.cross_attn_2 = MultiScaleDeformableAttention(**self.cross_attn_cfg)
        self.fusion_layer = SingleScaleBiAttentionBlock(**self.fusion_layer_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn1 = FFN(**self.ffn_cfg)
        self.ffn2 = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(8)
        ]
        self.norms = ModuleList(norms_list)

    def forward(self,
                query: Tensor,
                query_text: Tensor,
                key: Tensor = None,
                value: Tensor = None,
                query_pos: Tensor = None,
                query_text_pos: Tensor = None,
                key_pos: Tensor = None,
                self_attn_mask: Tensor = None,
                cross_attn_mask: Tensor = None,
                key_padding_mask: Tensor = None,
                memory_text: Tensor = None,
                text_attention_mask: Tensor = None,
                reference_points: Tensor = None,
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
        # self attention
        query_text = self.self_attn_text(
            query=query_text,
            key=query_text,
            value=query_text,
            query_pos=query_text_pos,
            key_pos=query_text_pos,
            attn_mask=None,
            **kwargs)
        query_text = self.norms[1](query_text)

        # cross attention between query and text
        query_text = self.cross_attn_text_text(
            query=query_text,
            query_pos=query_text_pos,
            key=memory_text,
            value=memory_text,
            key_padding_mask=text_attention_mask)
        query_text = self.norms[2](query_text)

        query = self.cross_attn_text(
            query=query,
            query_pos=query_pos,
            key=memory_text,
            value=memory_text,
            key_padding_mask=text_attention_mask)
        query = self.norms[3](query)

        # cross attention between query and image
        query = self.cross_attn(
            query=query,
            key=key,
            value=value,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=cross_attn_mask,
            key_padding_mask=key_padding_mask,
            reference_points=reference_points,
            **kwargs)

        bs, num, c = query.shape
        denoise_num = num - query_text.shape[1]
        query_denoise = query[:, :denoise_num, :]
        query_v = query[:, denoise_num:, :]
        reference_points_v = reference_points[:, denoise_num:, ...]
        query_text = self.cross_attn_2(
            query=query_text,
            key=key,
            value=value,
            query_pos=query_text_pos,
            key_pos=key_pos,
            attn_mask=cross_attn_mask,
            key_padding_mask=key_padding_mask,
            reference_points=reference_points_v,
            **kwargs)

        # cross attention between visual query and text query

        query_v, query_text = self.fusion_layer(
            visual_feature=query_v,
            lang_feature=query_text,
            attention_mask_v=None,
            attention_mask_l=None,
        )

        query = torch.cat([query_denoise, query_v], dim=1)


        query = self.norms[4](query)
        query = self.ffn1(query)
        query = self.norms[5](query)
        query_text = self.norms[6](query_text)
        query_text = self.ffn2(query_text)
        query_text = self.norms[7](query_text)

        return query, query_text