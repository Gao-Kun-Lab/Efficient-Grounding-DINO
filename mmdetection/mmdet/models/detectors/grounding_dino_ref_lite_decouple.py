# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, Tuple, Union, List

import torch
import torch.nn as nn
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from ..layers import SinePositionalEncoding
from ..layers.transformer.grounding_dino_layers import (
    GroundingDinoTransformerDecoder, GroundingDinoTransformerEncoder, GroundingDinoLiteDecoupleTransformerEncoder)
from .dino import DINO
from .glip import (create_positive_map, create_positive_map_label_to_token,
                   )
import re
import os
import os.path as osp
import cv2
import numpy as np
import torch.nn.functional as F

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
class GroundingDINO_ref_lite_decouple(DINO):
    """Implementation of `Grounding DINO: Marrying DINO with Grounded Pre-
    Training for Open-Set Object Detection.

    <https://arxiv.org/abs/2303.05499>`_

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/GroundingDINO>`_.
    """

    def __init__(self, language_model, show_feature=False, show_result=False, *args, **kwargs) -> None:
        self.language_model_cfg = language_model
        self._special_tokens = '. '
        self.show_feature = show_feature
        self.show_result = show_result
        super().__init__(*args, **kwargs)

        if self.show_feature:
            self.activations = []
            def forward_hook(module, input, output):
                self.activations.append(output)
                return None
            # def backward_hook(module, grad_input, grad_output):
            #     self.gradients['value'] = grad_output[0]
            #     return None
            # for i in range(self.encoder.num_layers):
            #     self.encoder.fusion_layers[i].register_forward_hook(forward_hook)
            self.encoder.fusion_layers[-1].register_forward_hook(forward_hook)


    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = GroundingDinoLiteDecoupleTransformerEncoder(**self.encoder)
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
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)

        encoder_outputs_dict = self.forward_encoder(
            **encoder_inputs_dict, text_dict=text_dict)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict

    def forward_encoder(self, feat: Tensor, feat_mask: Tensor,
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
        ################################ start imshow feature #################################
        if self.show_result:
            output_dir = './work_dirs/RSVG_out_imshow/baseline_imshow_RSVG_HR/'
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            img_path = batch_data_samples[0].img_path
            img = cv2.imread(img_path)
            bbox = results_list[0]['bboxes'][0]
            text = batch_data_samples[0].text
            cv2.rectangle(img,
                          (int(bbox[0].item()), int(bbox[1].item())),
                          (int(bbox[2].item()),int(bbox[3].item())),
                          (0, 255, 0), 3)
            cv2.imwrite(osp.join(output_dir, text + '.jpg'),img)
        if self.show_feature:
            norm_img = batch_inputs
            bs, _, h, w = norm_img.shape
            gt_list = [_.gt_instances.bboxes for _ in batch_data_samples]
            gt_masks = torch.zeros([bs, h, w], device=norm_img.device)
            pred_masks = torch.zeros([bs, h, w], device=norm_img.device)
            for i in range(bs):
                images = []
                result = results_list[i]
                gt_bboxes = gt_list[i]
                gt_mask = gt_masks[i]
                pred_mask = pred_masks[i]
                bboxes = result.bboxes
                scores = result.scores
                pred_bboxes = bboxes[0].unsqueeze(0)
                overlap, union = self.compute_iou(pred_bboxes, gt_bboxes)
                iou = overlap.reshape(bs, -1).sum(-1) * 1.0 / union.reshape(
                    bs, -1).sum(-1)

                if rescale:
                    assert batch_data_samples[i].get('scale_factor') is not None
                    scale_factor = batch_data_samples[i].get('scale_factor')
                    gt_bboxes = gt_bboxes * gt_bboxes.new_tensor(
                        scale_factor).repeat((1, 2))
                    pred_bboxes = pred_bboxes * pred_bboxes.new_tensor(
                        scale_factor).repeat((1, 2))
                for layer in range(len(self.activations)):
                    feature_i = self.activations[layer][0][i]
                    _, _, vT = torch.linalg.svd(feature_i)
                    cam = feature_i @ vT[0, :]
                    # cam = feature_i.mean(1)
                    start = 0
                    if iou.max() < 0.5:
                        # print(batch_data_samples[i].text)
                        show = False
                    else:
                        show = False

                    mean = [123.675, 116.28, 103.53]
                    std = [58.395, 57.12, 57.375]
                    ori_img = norm_img[i].cpu().clone()
                    for c in range(ori_img.shape[0]):
                        ori_img[c] = norm_img[i].cpu()[c] * std[c] + mean[c]
                    image = np.zeros((h, w, 3), dtype=np.uint8)
                    ori_img = np.array(ori_img.permute(1, 2, 0), dtype=np.uint8)
                    image = image + ori_img
                    for bbox in pred_bboxes:
                        x1, y1, x2, y2 = bbox.int()
                        pred_mask[y1:y2, x1:x2] = 1
                        cv2.rectangle(image, (x1.cpu().numpy().item(), y1.cpu().numpy().item()),
                                      (x2.cpu().numpy().item(), y2.cpu().numpy().item()),
                                      (0, 255, 0), 3)

                    for gt_bbox in gt_bboxes:
                        x1, y1, x2, y2 = gt_bbox.int()
                        gt_mask[y1:y2, x1:x2] = 1
                        cv2.rectangle(image, (x1.cpu().numpy().item(), y1.cpu().numpy().item()),
                                      (x2.cpu().numpy().item(), y2.cpu().numpy().item()),
                                      (0, 0, 255), 3)


                    out_str = ''
                    images.append(image)
                    for level, vf in enumerate(visual_feats):
                        f_w, f_h = vf.shape[-2], vf.shape[-1]
                        gap = f_w * f_h
                        saliency_map_ = cam[start:start + gap].reshape(1, 1, f_w, f_h)
                        saliency_map_ = F.interpolate(saliency_map_, size=(h, w), mode='bilinear', align_corners=False)
                        saliency_map_min, saliency_map_max = saliency_map_.min(), saliency_map_.max()
                        saliency_map_ = (saliency_map_ - saliency_map_min).div(saliency_map_max - saliency_map_min + 1e-8).squeeze().squeeze()
                        start += gap

                        gt_cam = saliency_map_[gt_mask == 1]
                        # pred_cam = saliency_map_[pred_mask == 1]
                        # bg_cam = saliency_map_[pred_mask == 0]
                        bg_cam = saliency_map_[gt_mask == 0]
                        delta = gt_cam.mean() - bg_cam.mean()
                        out_str += str(delta.item()) + ' '
                        # if show:
                        heatmap = cv2.applyColorMap(np.uint8(255 * saliency_map_.clone().detach().squeeze().cpu()), cv2.COLORMAP_JET)
                        heatmap = heatmap[..., ::-1]
                        result = np.int_(heatmap * 0.7 + image * 0.3)
                        # images.append(np.concatenate([image, heatmap, result], axis=1))
                        images.append(result)
                    # output_dir = './work_dirs/RSVG_out_imshow/'
                    # with open(osp.join(output_dir, 'gt_bg_delta_grounding_dino.txt'), 'a') as f:
                    #     f.write(out_str + '\n')
                # if show:
                images = np.concatenate(images, axis=1)
                # plt.imshow(images)
                # plt.show()
                output_dir = 'WORK_DIR/mmdetection/work_dirs/RSVG_out_imshow/cam_RSVG_baseline'

                # print(batch_data_samples[i].text)
                os.makedirs(output_dir, exist_ok=True)
                text = batch_data_samples[i].text
                img_id = batch_data_samples[i].img_id
                import random
                id = '_' + str(random.randint(0, 10000)) + '_' + str(random.randint(0, 1000)) + str(
                    random.randint(0, 100))
                output_name = text + '_' + id + '.jpg'
                output_path = os.path.join(output_dir, output_name)

                cv2.imwrite(output_path,images)

                    # images.append(saliency_map_.squeeze().squeeze())

            self.activations.clear()
        ################################ end imshow feature #################################

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

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs, has shape (bs, dim, H, W).
            batch_data_samples (List[:obj:`DetDataSample`], optional): The
                batch data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[Tensor]: A tuple of features from ``bbox_head`` forward.
        """
        text_prompts = [
            data_samples.text for data_samples in batch_data_samples
        ]
        custom_entities = True
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

        # head_inputs_dict = self.forward_transformer(visual_feats, text_dict,
        #                                             batch_data_samples)
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            visual_feats, batch_data_samples)

        encoder_outputs_dict = self.forward_encoder(
            **encoder_inputs_dict, text_dict=text_dict)
        encoder_output_results = (encoder_outputs_dict['memory'], encoder_outputs_dict['memory_text'])
        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)
        #
        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        #
        # results = self.bbox_head.forward(**head_inputs_dict)

        # return (visual_feats, text_dict['embedded'])
        # return encoder_output_results
        return decoder_outputs_dict['hidden_states']
        # return results

    def compute_iou(self, pred_bbox: torch.Tensor,
                    gt_bbox: torch.Tensor) -> tuple:

        area1 = (pred_bbox[..., 2] - pred_bbox[..., 0]) * (
                pred_bbox[..., 3] - pred_bbox[..., 1])
        area2 = (gt_bbox[..., 2] - gt_bbox[..., 0]) * (
                gt_bbox[..., 3] - gt_bbox[..., 1])

        lt = torch.max(pred_bbox[..., :, None, :2],
                       gt_bbox[..., None, :, :2])  # [B, rows, cols, 2]
        rb = torch.min(pred_bbox[..., :, None, 2:],
                       gt_bbox[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        union = area1[..., None] + area2[..., None, :] - overlap
        return overlap, union

def fp16_clamp(x, min=None, max=None):
    if not x.is_cuda and x.dtype == torch.float16:
        # clamp for cpu float16, tensor fp16 has no clamp implementation
        return x.float().clamp(min, max).half()

    return x.clamp(min, max)