# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from ..layers import SinePositionalEncoding
from ..layers.transformer.grounding_dino_layers import (
    GroundingDinoTransformerDecoder, GroundingDinoCoarseToFineTransformerEncoder)
from .dino import DINO
from .glip import (create_positive_map, create_positive_map_label_to_token,
                   )
import re

def find_noun_phrases(caption: str) -> list:
    """Find noun phrases in a caption using nltk.
    Args:
        caption (str): The caption to analyze.

    Returns:
        list: List of noun phrases found in the caption.

    Examples:
        >>> caption = 'There is two cat and a remote in the picture'
        >>> find_noun_phrases(caption) # ['cat', 'a remote', 'the picture']
    """
    try:
        import nltk
        # nltk.download('punkt')
        # nltk.download('averaged_perceptron_tagger')
    except ImportError:
        raise RuntimeError('nltk is not installed, please install it by: '
                           'pip install nltk.')

    caption = caption.lower()
    tokens = nltk.word_tokenize(caption)
    pos_tags = nltk.pos_tag(tokens)

    grammar = 'NP: {<DT>?<JJ.*>*<NN.*>+}'
    cp = nltk.RegexpParser(grammar)
    result = cp.parse(pos_tags)

    noun_phrases = []
    for subtree in result.subtrees():
        if subtree.label() == 'NP':
            noun_phrases.append(' '.join(t[0] for t in subtree.leaves()))

    return noun_phrases

def remove_punctuation(text: str) -> str:
    """Remove punctuation from a text.
    Args:
        text (str): The input text.

    Returns:
        str: The text with punctuation removed.
    """
    punctuation = [
        '|', ':', ';', '@', '(', ')', '[', ']', '{', '}', '^', '\'', '\"', '’',
        '`', '?', '$', '%', '#', '!', '&', '*', '+', ',', '.'
    ]
    for p in punctuation:
        text = text.replace(p, '')
    return text.strip()

def run_ner(caption: str) -> Tuple[list, list]:
    """Run NER on a caption and return the tokens and noun phrases.
    Args:
        caption (str): The input caption.

    Returns:
        Tuple[List, List]: A tuple containing the tokens and noun phrases.
            - tokens_positive (List): A list of token positions.
            - noun_phrases (List): A list of noun phrases.
    """
    noun_phrases = find_noun_phrases(caption)
    noun_phrases = [remove_punctuation(phrase) for phrase in noun_phrases]
    noun_phrases = [phrase for phrase in noun_phrases if phrase != '']
    relevant_phrases = noun_phrases
    labels = noun_phrases

    tokens_positive = []
    for entity, label in zip(relevant_phrases, labels):
        try:
            # search all occurrences and mark them as different entities
            # TODO: Not Robust
            for m in re.finditer(entity, caption.lower()):
                tokens_positive.append([[m.start(), m.end()]])
        except Exception:
            print('noun entities:', noun_phrases)
            print('entity:', entity)
            print('caption:', caption.lower())
    return tokens_positive, noun_phrases

@MODELS.register_module()
class GroundingDINO_ref_sparse(DINO):
    """Implementation of `Grounding DINO: Marrying DINO with Grounded Pre-
    Training for Open-Set Object Detection.

    <https://arxiv.org/abs/2303.05499>`_

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/GroundingDINO>`_.
    """

    def __init__(self, language_model, *args, **kwargs) -> None:

        self.language_model_cfg = language_model
        self._special_tokens = '. '
        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = GroundingDinoCoarseToFineTransformerEncoder(**self.encoder)
        self.decoder = GroundingDinoTransformerDecoder(**self.decoder)
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
            tokens_positive = [[_[0] for _ in tokens_positive]]
            entities = noun_phrases
            caption_string = original_caption

        return tokenized, caption_string, tokens_positive, entities

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
        return positive_map_label_to_token, caption_string, \
            positive_map, entities

    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        text_dict: Dict,
        batch_data_samples: OptSampleList = None,
    ) -> Dict:
        encoder_inputs_dict, decoder_inputs_dict, head_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)

        encoder_outputs_dict, encoder_head_inputs_dict = self.forward_encoder(
            **encoder_inputs_dict, text_dict=text_dict)
        head_inputs_dict.update(encoder_head_inputs_dict)

        tmp_dec_in, pre_decoder_head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)
        head_inputs_dict.update(pre_decoder_head_inputs_dict)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict

    def forward_encoder(self, feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor, spatial_shapes: Tensor,
                        level_start_index: Tensor, valid_ratios: Tensor,
                        text_dict: Dict) -> Dict:
        text_token_mask = text_dict['text_token_mask']
        memory, memory_text, binary_class = self.encoder(
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
            text_self_attention_masks=text_dict['masks'],
            # sparse cls branch
            sparse_cls_branch=self.bbox_head.sparse_cls_branches
        )
        encoder_outputs_dict = dict(
            memory=memory,
            memory_mask=feat_mask,
            spatial_shapes=spatial_shapes,
            memory_text=memory_text,
            text_token_mask=text_token_mask,
        )
        head_inputs_dict = dict(binary_class=binary_class) if self.training else dict()

        return encoder_outputs_dict, head_inputs_dict


    def pre_transformer(
            self,
            mlvl_feats: Tuple[Tensor],
            batch_data_samples: OptSampleList = None) -> Tuple[Dict]:
        """Process image features before feeding them to the transformer.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            mlvl_feats (tuple[Tensor]): Multi-level features that may have
                different resolutions, output from neck. Each feature has
                shape (bs, dim, h_lvl, w_lvl), where 'lvl' means 'layer'.
            batch_data_samples (list[:obj:`DetDataSample`], optional): The
                batch data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[dict]: The first dict contains the inputs of encoder and the
            second dict contains the inputs of decoder.

            - encoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_encoder()`, which includes 'feat', 'feat_mask',
              and 'feat_pos'.
            - decoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_decoder()`, which includes 'memory_mask'.
        """
        batch_size = mlvl_feats[0].size(0)

        # construct binary masks for the transformer.
        assert batch_data_samples is not None
        batch_input_shape = batch_data_samples[0].batch_input_shape
        input_img_h, input_img_w = batch_input_shape
        img_shape_list = [sample.img_shape for sample in batch_data_samples]
        same_shape_flag = all([
            s[0] == input_img_h and s[1] == input_img_w for s in img_shape_list
        ])
        # support torch2onnx without feeding masks
        if torch.onnx.is_in_onnx_export() or same_shape_flag:
            mlvl_masks = []
            mlvl_pos_embeds = []
            for feat in mlvl_feats:
                mlvl_masks.append(None)
                mlvl_pos_embeds.append(
                    self.positional_encoding(None, input=feat))
        else:
            masks = mlvl_feats[0].new_ones(
                (batch_size, input_img_h, input_img_w))
            for img_id in range(batch_size):
                img_h, img_w = img_shape_list[img_id]
                masks[img_id, :img_h, :img_w] = 0
            # NOTE following the official DETR repo, non-zero
            # values representing ignored positions, while
            # zero values means valid positions.

            mlvl_masks = []
            mlvl_pos_embeds = []
            for feat in mlvl_feats:
                mlvl_masks.append(
                    F.interpolate(masks[None], size=feat.shape[-2:]).to(
                        torch.bool).squeeze(0))
                mlvl_pos_embeds.append(
                    self.positional_encoding(mlvl_masks[-1]))

        feat_flatten = []
        lvl_pos_embed_flatten = []
        mask_flatten = []
        spatial_shapes = []
        binary_mask_xy_list = []

        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            batch_size, c, h, w = feat.shape
            spatial_shape = torch._shape_as_tensor(feat)[2:].to(feat.device)
            # [bs, c, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl, c]
            feat = feat.view(batch_size, c, -1).permute(0, 2, 1)
            pos_embed = pos_embed.view(batch_size, c, -1).permute(0, 2, 1)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            # [bs, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl]
            if mask is not None:
                mask = mask.flatten(1)

            feat_flatten.append(feat)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            mask_flatten.append(mask)
            spatial_shapes.append(spatial_shape)

            # create results
            scale = torch.tensor([input_img_w / w, input_img_h / h], device=feat.device)
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(
                    0, w - 1, w, dtype=torch.float32, device=feat.device),
                torch.linspace(
                    0, h - 1, h, dtype=torch.float32, device=feat.device),
            )

            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
            grid = (grid.unsqueeze(0).expand(batch_size, -1, -1, -1) + 0.5) * scale
            grid = grid.view(batch_size, -1, 2)
            binary_mask_xy_list.append(grid)

        # (bs, num_feat_points, dim)
        feat_flatten = torch.cat(feat_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        binary_mask_xy = torch.cat(binary_mask_xy_list, 1)
        # (bs, num_feat_points), where num_feat_points = sum_lvl(h_lvl*w_lvl)
        if mask_flatten[0] is not None:
            mask_flatten = torch.cat(mask_flatten, 1)
        else:
            mask_flatten = None

        # (num_level, 2)
        spatial_shapes = torch.cat(spatial_shapes).view(-1, 2)
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1, )),  # (num_level)
            spatial_shapes.prod(1).cumsum(0)[:-1]))
        if mlvl_masks[0] is not None:
            valid_ratios = torch.stack(  # (bs, num_level, 2)
                [self.get_valid_ratio(m) for m in mlvl_masks], 1)
        else:
            valid_ratios = mlvl_feats[0].new_ones(batch_size, len(mlvl_feats),
                                                  2)

        encoder_inputs_dict = dict(
            feat=feat_flatten,
            feat_mask=mask_flatten,
            feat_pos=lvl_pos_embed_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios)
        decoder_inputs_dict = dict(
            memory_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios)

        head_inputs_dict = dict(binary_mask_xy=binary_mask_xy) if self.training else dict()
        return encoder_inputs_dict, decoder_inputs_dict, head_inputs_dict


    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        memory_text: Tensor,
        text_token_mask: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        bs, _, c = memory.shape

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)

        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](output_memory, memory_text,
                                     text_token_mask)
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers].max_text_len
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](output_memory) + output_proposals

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
            # no dn part
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
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
        # if len(set(text_prompts)) == 1:
        #     # All the text prompts are the same,
        #     # so there is no need to calculate them multiple times.
        #     tokenized, caption_string, tokens_positive, _ = \
        #         self.get_tokens_and_prompts(
        #             text_prompts[0], True)
        #     new_text_prompts = [caption_string] * len(batch_inputs)
        #     # for gt_label in gt_labels:
        #     #     new_tokens_positive = [
        #     #         tokens_positive[label] for label in gt_label
        #     #     ]
        #     #     _, positive_map = self.get_positive_map(
        #     #         tokenized, new_tokens_positive)
        #     #     positive_maps.append(positive_map)
        # else:
        for text_prompt in text_prompts:
            tokenized, caption_string, tokens_positive, _ = \
                self.get_tokens_and_prompts(
                    text_prompt, True)
            # new_tokens_positive = [
            #     tokens_positive[label] for label in gt_label
            # ]
            _, positive_map = self.get_positive_map(
                tokenized, tokens_positive)
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

        # create grounding truth
        for i, data_samples in enumerate(batch_data_samples):
            bs, _, h, w = batch_inputs.shape
            gt_mask = torch.ones([h, w], device=batch_inputs.device) + 1
            gt_bboxes = data_samples.gt_instances.bboxes
            for gt_bbox in gt_bboxes:
                x1, y1, x2, y2 = gt_bbox.int()
                gt_mask[torch.clamp(y1, 0, h - 1).int():torch.clamp(y2, 0, h - 1).int(),
                torch.clamp(x1, 0, w - 1).int():torch.clamp(x2, 0, w - 1).int()] = 0
            data_samples.gt_binary_mask = gt_mask

        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)
        return losses

    def predict(self, batch_inputs, batch_data_samples, rescale: bool = True):
        text_prompts = [
            data_samples.text for data_samples in batch_data_samples
        ]
        # if 'custom_entities' in batch_data_samples[0]:
        #     # Assuming that the `custom_entities` flag
        #     # inside a batch is always the same. For single image inference
        #     custom_entities = batch_data_samples[0].custom_entities
        # else:
        custom_entities = True
        # if len(text_prompts) == 1:
        #     # All the text prompts are the same,
        #     # so there is no need to calculate them multiple times.
        #     _positive_maps_and_prompts = [
        #         self.get_tokens_positive_and_prompts(text_prompts[0],
        #                                              custom_entities)
        #     ] * len(batch_inputs)
        # else:
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
            # if len(pred_instances) > 0:
                # label_names = []
                # for labels in pred_instances.labels:
                #     if labels >= len(entity):
                #         warnings.warn(
                #             'The unexpected output indicates an issue with '
                #             'named entity recognition. You can try '
                #             'setting custom_entities=True and running '
                #             'again to see if it helps.')
                #         label_names.append('unobject')
                #     else:
                #         label_names.append(entity[labels])
                # for visualization
                # pred_instances.label_names = label_names
            data_sample.pred_instances = pred_instances
        return batch_data_samples
