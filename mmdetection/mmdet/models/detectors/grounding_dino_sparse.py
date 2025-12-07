# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from ..layers import SinePositionalEncoding
from ..layers.transformer.grounding_dino_sparse_layers import (
    GroundingDinoSparseTransformerDecoder, GroundingDinoSparseTransformerEncoder)
from .dino import DINO
from .glip import (create_positive_map, create_positive_map_label_to_token,
                   run_ner)
from ..layers import inverse_sigmoid
from mmdet.models.dense_heads.atss_vlfusion_head import convert_grounding_to_cls_scores

@MODELS.register_module()
class GroundingDINO_sparse(DINO):
    """Implementation of `Grounding DINO: Marrying DINO with Grounded Pre-
    Training for Open-Set Object Detection.

    <https://arxiv.org/abs/2303.05499>`_

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/GroundingDINO>`_.
    """

    def __init__(self, language_model, encoder_ratio, dynamic_sparsity, *args, **kwargs) -> None:

        self.language_model_cfg = language_model
        self._special_tokens = '. '
        self.encoder_ratio = encoder_ratio
        self.dynamic_sparsity = dynamic_sparsity
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = GroundingDinoSparseTransformerEncoder(**self.encoder)
        self.decoder = GroundingDinoSparseTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)
        self.memory_sparse_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_sparse_trans_norm = nn.LayerNorm(self.embed_dims)

        # text modules
        self.language_model = MODELS.build(self.language_model_cfg)
        self.text_feat_map = nn.Linear(
            self.language_model.language_backbone.body.language_dim,
            self.embed_dims,
            bias=True)

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super().init_weights()
        nn.init.constant_(self.text_feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.text_feat_map.weight.data)

    def get_tokens_and_prompts(
            self,
            original_caption: Union[str, list, tuple],
            custom_entities: bool = False) -> Tuple[dict, str, list]:
        """Get the tokens positive and prompts for the caption."""
        if isinstance(original_caption, (list, tuple)) or custom_entities:
            if custom_entities and isinstance(original_caption, str):
                original_caption = original_caption.strip(self._special_tokens)
                original_caption = original_caption.split(self._special_tokens)
                original_caption = list(
                    filter(lambda x: len(x) > 0, original_caption))

            caption_string = ''
            tokens_positive = []
            for idx, word in enumerate(original_caption):
                tokens_positive.append(
                    [[len(caption_string),
                      len(caption_string) + len(word)]])
                caption_string += word
                caption_string += self._special_tokens
            # tokens_positive_classes = tokens_positive_single[:-3]
            # tokens_positive_scales = tokens_positive_single[-3:]
            # tokens_positive = []
            # for i in range(3):
            #     for tokens_positive_class in tokens_positive_classes:
            #         mix_token_positive = [tokens_positive_class[0], tokens_positive_scales[i][0]]
            #         tokens_positive.append(mix_token_positive)
            # NOTE: Tokenizer in Grounding DINO is different from
            # that in GLIP. The tokenizer in GLIP will pad the
            # caption_string to max_length, while the tokenizer
            # in Grounding DINO will not.
            tokenized = self.language_model.tokenizer(
                [caption_string],
                padding='max_length'
                if self.language_model.pad_to_max else 'longest',
                return_tensors='pt')
            entities = original_caption
        else:
            if not original_caption.endswith('.'):
                original_caption = original_caption + self._special_tokens
            # NOTE: Tokenizer in Grounding DINO is different from
            # that in GLIP. The tokenizer in GLIP will pad the
            # caption_string to max_length, while the tokenizer
            # in Grounding DINO will not.
            tokenized = self.language_model.tokenizer(
                [original_caption],
                padding='max_length'
                if self.language_model.pad_to_max else 'longest',
                return_tensors='pt')
            tokens_positive, noun_phrases = run_ner(original_caption)
            entities = noun_phrases
            caption_string = original_caption

        return tokenized, caption_string, tokens_positive, entities
        # return tokenized, caption_string, tokens_positive, tokens_positive_single, entities

    def get_positive_map(self, tokenized, tokens_positive):
        positive_map = create_positive_map(tokenized, tokens_positive)
        positive_map_label_to_token = create_positive_map_label_to_token(
            positive_map, plus=1)
        return positive_map_label_to_token, positive_map

    def get_tokens_positive_and_prompts(
            self,
            original_caption: Union[str, list, tuple],
            custom_entities: bool = False) -> Tuple[dict, str, Tensor, list]:
        """Get the tokens positive and prompts for the caption.

        Args:
            original_caption (str): The original caption, e.g. 'bench . car .'
            custom_entities (bool, optional): Whether to use custom entities.
                If ``True``, the ``original_caption`` should be a list of
                strings, each of which is a word. Defaults to False.

        Returns:
            Tuple[dict, str, dict, str]: The dict is a mapping from each entity
            id, which is numbered from 1, to its positive token id.
            The str represents the prompts.
        """
        tokenized, caption_string, tokens_positive, entities = \
            self.get_tokens_and_prompts(
                original_caption, custom_entities)
        positive_map_label_to_token, positive_map = self.get_positive_map(
            tokenized, tokens_positive)
        # positive_map_label_to_token_single, positive_map = self.get_positive_map(
        #     tokenized, tokens_positive_single)
        return positive_map_label_to_token, caption_string, \
            positive_map, entities

    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        text_dict: Dict,
        batch_data_samples: OptSampleList = None,
    ) -> Dict:
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)

        pre_encoder_inputs_dict = dict(
            memory=encoder_inputs_dict['feat'],
            memory_pos=encoder_inputs_dict['feat_pos'],
            memory_mask=encoder_inputs_dict['feat_mask'],
            spatial_shapes=encoder_inputs_dict['spatial_shapes'],
            level_start_index=encoder_inputs_dict['level_start_index'],
            valid_ratios=encoder_inputs_dict['valid_ratios'],
            memory_text=text_dict['embedded'],
            text_token_mask=text_dict['text_token_mask'],
        )

        tmp_enc_in, pre_enc_head_inputs_dict = self.pre_encoder(**pre_encoder_inputs_dict)

        encoder_outputs_dict, enc_head_inputs_dict = self.forward_encoder(
            **tmp_enc_in, text_dict=text_dict)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        head_inputs_dict.update(pre_enc_head_inputs_dict)
        if self.training:
            head_inputs_dict.update(enc_head_inputs_dict=enc_head_inputs_dict)
        return head_inputs_dict

    def forward_encoder(self, feat: Tensor, feat_mask: Tensor, feat_mask_ori: Tensor, feat_pos_ori: Tensor,
                        feat_pos: Tensor, reference_points: Tensor, reference_points_ori: Tensor,
                        memory: Tensor, spatial_shapes: Tensor, topk_indices: Tensor,
                        level_start_index: Tensor, valid_ratios: Tensor,
                        text_dict: Dict) -> Dict:
        text_token_mask = text_dict['text_token_mask']

        memory_sparse, memory, \
        reference_points, reference_points_ori, \
        memory_text, output_head_dict = self.encoder(
            query=feat,
            query_pos=feat_pos,
            memory_pos_ori=feat_pos_ori,
            key_padding_mask=feat_mask,  # for self_attn
            key_padding_mask_ori=feat_mask_ori,  # for self_attn
            reference_points=reference_points,
            reference_points_ori=reference_points_ori,
            topk_indices=topk_indices,
            memory=memory,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            # for text encoder
            memory_text=text_dict['embedded'],
            text_attention_mask=~text_token_mask,
            text_token_mask=text_token_mask,
            position_ids=text_dict['position_ids'],
            text_self_attention_masks=text_dict['masks'],
            reg_branches=self.bbox_head.reg_branches[-(self.encoder.num_layers - 1):] if self.aux_encoder else None,
            cls_branches=self.bbox_head.cls_branches[-(self.encoder.num_layers - 1):] if self.aux_encoder and self.dynamic_sparsity else None)
        encoder_outputs_dict = dict(
            memory=memory_sparse,
            memory_ori=memory,
            reference_points=reference_points,
            reference_points_ori=reference_points_ori,
            memory_mask=feat_mask,
            memory_mask_ori=feat_mask_ori,
            spatial_shapes=spatial_shapes,
            memory_text=memory_text,
            text_token_mask=text_token_mask,
            )
        if self.aux_encoder:
            if self.dynamic_sparsity:
                encoder_head_dict = output_head_dict if self.training else dict()
            else:
                encoder_head_dict = output_head_dict if self.training else dict()
        else:
            encoder_head_dict = dict()
        return encoder_outputs_dict, encoder_head_dict

    def forward_encoder_back(self, feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor, spatial_shapes: Tensor,
                        level_start_index: Tensor, valid_ratios: Tensor,
                        text_dict: Dict) -> Dict:
        text_token_mask = text_dict['text_token_mask']
        memory, memory_text = self.encoder(
            query=feat,
            query_pos=feat_pos,
            key_padding_mask=feat_mask,  # for self_attn
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            # for text encoder
            memory_text=text_dict['embedded'],
            text_attention_mask=~text_token_mask,
            position_ids=text_dict['position_ids'],
            text_self_attention_masks=text_dict['masks'])
        encoder_outputs_dict = dict(
            memory=memory,
            memory_mask=feat_mask,
            spatial_shapes=spatial_shapes,
            memory_text=memory_text,
            text_token_mask=text_token_mask)
        return encoder_outputs_dict

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

        output_memory = memory
        if memory_mask is not None:
            output_memory = output_memory.masked_fill(
                memory_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid,
                                                  float(0))
        output_memory = self.memory_sparse_trans_fc(output_memory)
        output_memory = self.memory_sparse_trans_norm(output_memory)
        # [bs, sum(hw), 2]
        return output_memory, output_proposals

    def pre_encoder(
        self,
        memory: Tensor,
        memory_pos: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        memory_text: Tensor,
        text_token_mask: Tensor,
        **kwargs
    ) -> Tuple[Dict]:
        bs, token_number, c = memory.shape
        self.aux_encoder = self.bbox_head.aux_encoder

        self.sparse_token_nums = min(int(token_number * self.encoder_ratio) + 1, token_number)

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)

        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers + 1](output_memory, memory_text,
                                     text_token_mask)
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers + 1].max_text_len
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers + 1](output_memory) + output_proposals

        # NOTE The DINO selects top-k proposals according to scores of
        # multi-class classification, while DeformDETR, where the input
        # is `enc_outputs_class[..., 0]` selects according to scores of
        # binary classification.
        topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0], k=self.sparse_token_nums, dim=1)[1]

        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        query = torch.gather(memory, 1, topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        query_pos = torch.gather(memory_pos, 1, topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        if memory_mask is not None:
            memory_mask_ori = memory_mask
            memory_mask = torch.gather(memory_mask, 1, topk_indices)

        reference_points = topk_coords_unact
        reference_points = reference_points.sigmoid()
        reference_points_ori = enc_outputs_coord_unact.detach().sigmoid()

        encoder_inputs_dict = dict(
            feat=query,
            feat_pos=query_pos,
            feat_pos_ori=memory_pos,
            feat_mask=memory_mask,
            feat_mask_ori=memory_mask_ori if memory_mask is not None else memory_mask,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            reference_points_ori=reference_points_ori,
            memory=memory,
            level_start_index=kwargs['level_start_index'],
            valid_ratios=kwargs['valid_ratios'],
            topk_indices=topk_indices
            # memory_text=memory_text,
            # text_attention_mask=~text_token_mask,
        )
        # NOTE DINO calculates encoder losses on scores and coordinates
        # of selected top-k encoder queries, while DeformDETR is of all
        # encoder queries.
        head_inputs_dict = dict(
            pre_enc_outputs_class=topk_score,
            pre_enc_outputs_coord=topk_coords,
            ) if self.training else dict()
        # append text_feats to head_inputs_dict
        # head_inputs_dict['pre_memory_text'] = memory_text
        head_inputs_dict['text_token_mask'] = text_token_mask
        return encoder_inputs_dict, head_inputs_dict

    def pre_decoder(
            self,
            memory: Tensor,
            memory_ori: Tensor,
            reference_points: Tensor,
            reference_points_ori: Tensor,
            memory_mask: Tensor,
            memory_mask_ori: Tensor,
            spatial_shapes: Tensor,
            memory_text: Tensor,
            text_token_mask: Tensor,
            batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        bs, _, c = memory.shape

        output_memory = self.memory_trans_fc(memory_ori)
        output_memory = self.memory_trans_norm(output_memory)

        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](output_memory, memory_text,
                                     text_token_mask)
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers].max_text_len
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](output_memory) + inverse_sigmoid(reference_points_ori, eps=1e-3)

        # NOTE The DINO selects top-k proposals according to scores of
        # multi-class classification, while DeformDETR, where the input
        # is `enc_outputs_class[..., 0]` selects according to scores of
        # binary classification.
        topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]

        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        query = self.query_embedding.weight[:, None, :]
        query = query.repeat(1, bs, 1).transpose(0, 1)
        if self.training:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                         dim=1)
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=memory_ori,
            reference_points=reference_points,
            dn_mask=dn_mask,
            memory_text=memory_text,
            text_attention_mask=~text_token_mask,
        )
        # NOTE DINO calculates encoder losses on scores and coordinates
        # of selected top-k encoder queries, while DeformDETR is of all
        # encoder queries.
        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords,
            dn_meta=dn_meta) if self.training else dict()
        # append text_feats to head_inputs_dict
        head_inputs_dict['memory_text'] = memory_text
        head_inputs_dict['text_token_mask'] = text_token_mask
        return decoder_inputs_dict, head_inputs_dict

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        # TODO: Only open vocabulary tasks are supported for training now.
        text_prompts = [
            data_samples.text for data_samples in batch_data_samples
        ]

        gt_labels = [
            data_samples.gt_instances.labels
            for data_samples in batch_data_samples
        ]

        new_text_prompts = []
        positive_maps = []
        if len(set(text_prompts)) == 1:
            # All the text prompts are the same,
            # so there is no need to calculate them multiple times.
            tokenized, caption_string, tokens_positive, _ = \
                self.get_tokens_and_prompts(
                    text_prompts[0], True)
            new_text_prompts = [caption_string] * len(batch_inputs)
            for gt_label in gt_labels:
                new_tokens_positive = [
                    tokens_positive[label] for label in gt_label
                ]
                _, positive_map = self.get_positive_map(
                    tokenized, new_tokens_positive)
                positive_maps.append(positive_map)
        else:
            for text_prompt, gt_label in zip(text_prompts, gt_labels):
                tokenized, caption_string, tokens_positive, _ = \
                    self.get_tokens_and_prompts(
                        text_prompt, True)
                new_tokens_positive = [
                    tokens_positive[label] for label in gt_label
                ]
                _, positive_map = self.get_positive_map(
                    tokenized, new_tokens_positive)
                positive_maps.append(positive_map)
                new_text_prompts.append(caption_string)

        text_dict = self.language_model(new_text_prompts)
        if self.text_feat_map is not None:
            text_dict['embedded'] = self.text_feat_map(text_dict['embedded'])

        for i, data_samples in enumerate(batch_data_samples):
            positive_map = positive_maps[i].to(
                batch_inputs.device).bool().float()
            text_token_mask = text_dict['text_token_mask'][i]
            data_samples.gt_instances.positive_maps = positive_map
            data_samples.gt_instances.text_token_mask = \
                text_token_mask.unsqueeze(0).repeat(
                    len(positive_map), 1)

        visual_features = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(visual_features, text_dict,
                                                    batch_data_samples)

        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)
        return losses

    def predict(self, batch_inputs, batch_data_samples, rescale: bool = True):
        text_prompts = [
            data_samples.text for data_samples in batch_data_samples
        ]
        if 'custom_entities' in batch_data_samples[0]:
            # Assuming that the `custom_entities` flag
            # inside a batch is always the same. For single image inference
            custom_entities = batch_data_samples[0].custom_entities
        else:
            custom_entities = False
        if len(text_prompts) == 1:
            # All the text prompts are the same,
            # so there is no need to calculate them multiple times.
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(text_prompts[0],
                                                     custom_entities)
            ] * len(batch_inputs)
        else:
            _positive_maps_and_prompts = [
                self.get_tokens_positive_and_prompts(text_prompt,
                                                     custom_entities)
                for text_prompt in text_prompts
            ]
        token_positive_maps, text_prompts, _, entities = zip(
            *_positive_maps_and_prompts)
        # extract text feats
        text_dict = self.language_model(list(text_prompts))
        # text feature map layer
        if self.text_feat_map is not None:
            text_dict['embedded'] = self.text_feat_map(text_dict['embedded'])

        for i, data_samples in enumerate(batch_data_samples):
            data_samples.token_positive_map = token_positive_maps[i]

        # image feature extraction
        visual_feats = self.extract_feat(batch_inputs)

        head_inputs_dict = self.forward_transformer(visual_feats, text_dict,
                                                    batch_data_samples)
        results_list = self.bbox_head.predict(
            **head_inputs_dict,
            rescale=rescale,
            batch_data_samples=batch_data_samples)
        for data_sample, pred_instances, entity in zip(batch_data_samples,
                                                       results_list, entities):
            if len(pred_instances) > 0:
                label_names = []
                for labels in pred_instances.labels:
                    if labels >= len(entity):
                        warnings.warn(
                            'The unexpected output indicates an issue with '
                            'named entity recognition. You can try '
                            'setting custom_entities=True and running '
                            'again to see if it helps.')
                        label_names.append('unobject')
                    else:
                        label_names.append(entity[labels])
                # for visualization
                pred_instances.label_names = label_names
            data_sample.pred_instances = pred_instances
        return batch_data_samples
