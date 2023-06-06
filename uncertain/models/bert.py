from copyreg import pickle
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, KLDivLoss, CosineSimilarity
from torch.nn.init import xavier_uniform_
import torch.nn.functional as F
import os

import pickle
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput
from .utils import LabelSmoothingLoss, VAE, BNN, pairwise_loss, kl_loss, orthogonal_loss, kl_pairwise_loss, orthogonal_pairwise_loss, kl_orthogonal_pairwise_loss, SequenceAnalysisOutput, DropoutMC, write_latent, write_representations, HLoss

class BertLebelSmoothingSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = BertModel(config)
        
        self.labelsmoothing = LabelSmoothingLoss(self.num_labels, 0.01)

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.sm = nn.Softmax(dim=1)

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        cls_representation = outputs[1]

        pooled_output = self.dropout(cls_representation)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = CrossEntropyLoss()
                cross_entropy = loss_fct(logits.view(-1, self.num_labels), pesudo_labels)
                loss = cross_entropy + klloss
                # loss = klloss
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                loss = self.labelsmoothing(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class BertVAEForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = BertModel(config)

        self.vae = VAE(config, hidden_size=config.hidden_size, latent_size=100, var_scale=0.1, eta_bn_prop=0.0)

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.sm = nn.Softmax(dim=1)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        cls_representation = outputs[1]

        latent_z= self.vae.encoder(input=cls_representation)
        cls_representation_recon = self.vae.decoder(z=latent_z)

        # write_latent(latent_z,labels)

        pooled_output = self.dropout(cls_representation_recon)
        logits = self.classifier(pooled_output)
        pesudo_labels = torch.argmax(self.sm(self.classifier(self.dropout(cls_representation))), dim=1)

        pesudo_labels = torch.argmax(self.sm(self.classifier(self.dropout(cls_representation))), dim=1)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = CrossEntropyLoss()
                cross_entropy = loss_fct(logits.view(-1, self.num_labels), pesudo_labels)
                opd = orthogonal_pairwise_loss(self.vae.latent_size, cls_representation, cls_representation_recon, self.vae.bate_layer)
                loss = cross_entropy + opd
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), pesudo_labels)
                loss +=  orthogonal_pairwise_loss(self.vae.latent_size, cls_representation, cls_representation_recon, self.vae.bate_layer)
                # loss +=  kl_orthogonal_pairwise_loss(self.vae.latent_size, cls_representation, cls_representation_recon,latent_z, self.vae.bate_layer)
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class BertVAEForSequenceClassificationMaskAnalysis(BertVAEForSequenceClassification):
    def __init__(self, config, mask_strategy):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = BertModel(config)
        self.latent_size = 100

        self.vae = VAE(config, hidden_size=config.hidden_size, latent_size=self.latent_size, var_scale=0.1, eta_bn_prop=0.0)

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.cossim = nn.CosineSimilarity(dim=1, eps=1e-6)

        self.sm = nn.Softmax(dim=1)
        self.mask_strategy = mask_strategy

        # Initialize weights and apply final processing
        self.post_init()
    
    # Note this method requires batchsize 1
    def mask_by_delta_e(self, latent_z, delta_e):
        latent_zs = latent_z[0]
        ezs_scores = []
        for idx, zi in enumerate(latent_zs):
            mask = [0.]*self.latent_size
            mask = torch.FloatTensor([mask]).cuda()
            mask[0][idx] = zi
            ezi = self.vae.decoder(z=mask)
            score = self.cossim(ezi, delta_e)
            ezs_scores += [score]
        ezs_scores = torch.FloatTensor(ezs_scores).cuda()
        indices = torch.argsort(ezs_scores)[self.mask_strategy[0]:self.mask_strategy[1]]
        mask = [1.]*self.latent_size
        mask = [0. if idx in indices else each for idx, each in enumerate(mask)]
        mask = torch.FloatTensor([mask]).cuda()
        return mask * latent_z 
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        cls_representation = outputs[1]

        latent_z = self.vae.encoder(input=cls_representation)
        cls_representation_recon = self.vae.decoder(z=latent_z)

        masked_z = self.mask_by_delta_e(latent_z, cls_representation_recon-cls_representation)
        cls_representation_recon = self.vae.decoder(z=masked_z)

        pooled_output = self.dropout(cls_representation_recon)
        logits = self.classifier(pooled_output)

        pooled_output = self.dropout(cls_representation)
        ori_logits = self.classifier(pooled_output)

        pesudo_labels = torch.argmax(self.sm(ori_logits), dim=1)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = CrossEntropyLoss()
                cross_entropy = loss_fct(logits.view(-1, self.num_labels), pesudo_labels)
                klloss = orthogonal_pairwise_loss(self.vae.latent_size, cls_representation, cls_representation_recon, self.vae.bate_layer)
                loss = cross_entropy + klloss

            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                cross_entropy = loss_fct(logits.view(-1, self.num_labels), pesudo_labels)
                klloss = orthogonal_pairwise_loss(self.vae.latent_size, cls_representation, cls_representation_recon, self.vae.bate_layer)
                loss = cross_entropy + klloss
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class BertVAEForSequenceClassificationAnalysis(BertVAEForSequenceClassification):

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        cls_representation = outputs[1]

        latent_z = self.vae.encoder(input=cls_representation)
        cls_representation_recon = self.vae.decoder(z=latent_z)

        pooled_output = self.dropout(cls_representation_recon)
        logits = self.classifier(pooled_output)

        pooled_output = self.dropout(cls_representation)
        ori_logits = self.classifier(pooled_output)

        perturbed = self.vae.latent_perturb(z=latent_z)
        influntial_idxs, influntial_idx_coss, word_cosss = self.vae.cos_delta_by_dimension_batches(cls_representation_recon-cls_representation, perturbed, outputs.last_hidden_state)
        recon_labels = torch.argmax(self.sm(logits), dim=1)
        pesudo_labels = torch.argmax(self.sm(ori_logits), dim=1)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = CrossEntropyLoss()
                cross_entropy = loss_fct(logits.view(-1, self.num_labels), pesudo_labels)
                klloss = orthogonal_pairwise_loss(self.vae.latent_size, cls_representation, cls_representation_recon, self.vae.bate_layer)
                loss = cross_entropy + klloss

            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                cross_entropy = loss_fct(logits.view(-1, self.num_labels), pesudo_labels)
                klloss = orthogonal_pairwise_loss(self.vae.latent_size, cls_representation, cls_representation_recon, self.vae.bate_layer)
                loss = cross_entropy + klloss
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceAnalysisOutput(
            loss=loss,
            logits=pesudo_labels,
            recon_logits=recon_labels,
            influntial_ids=torch.tensor(influntial_idxs),
            influntial_idx_coss=torch.tensor(influntial_idx_coss),
            part_input_idss=input_ids,
            word_cosss=torch.tensor(word_cosss)
        )

class BertBNNForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = BertModel(config)

        self.bnn = BNN(config, hidden_size=config.hidden_size, latent_size=100)

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.sm = nn.Softmax(dim=1)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        cls_representation = outputs[1]

        latent_z = self.bnn.encoder(input=cls_representation)
        # write_latent(latent_z,labels)
        cls_representation_recon = self.bnn.decoder(z=latent_z)

        pooled_output = self.dropout(cls_representation_recon)
        logits = self.classifier(pooled_output)
        # write_representations(cls_representation, cls_representation_recon, labels)

        pesudo_labels = torch.argmax(self.sm(self.classifier(self.dropout(cls_representation))), dim=1)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = CrossEntropyLoss()
                cross_entropy = loss_fct(logits.view(-1, self.num_labels), pesudo_labels)
                klloss = orthogonal_pairwise_loss(self.bnn.latent_size, cls_representation, cls_representation_recon, self.bnn.linear)
                loss = cross_entropy + klloss
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                cross_entropy = loss_fct(logits.view(-1, self.num_labels), pesudo_labels)
                klloss = orthogonal_pairwise_loss(self.bnn.latent_size, cls_representation, cls_representation_recon, self.bnn.linear)
                loss = cross_entropy + klloss 
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class BertBNNForSequenceClassificationMaskAnalysis(BertBNNForSequenceClassification):
    def __init__(self, config, mask_strategy):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = BertModel(config)
        self.latent_size = 100

        self.bnn = BNN(config, hidden_size=config.hidden_size, latent_size=100)

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.cossim = nn.CosineSimilarity(dim=1, eps=1e-6)

        self.sm = nn.Softmax(dim=1)
        self.mask_strategy = mask_strategy

        # Initialize weights and apply final processing
        self.post_init()
    
    # Note this method requires batchsize 1
    def mask_by_delta_e(self, latent_z, delta_e):
        latent_zs = latent_z[0]
        ezs_scores = []
        for idx, zi in enumerate(latent_zs):
            mask = [0.]*self.latent_size
            mask = torch.FloatTensor([mask]).cuda()
            mask[0][idx] = zi
            ezi = self.bnn.decoder(z=mask)
            score = self.cossim(ezi, delta_e)
            ezs_scores += [score]
        ezs_scores = torch.FloatTensor(ezs_scores).cuda()
        indices = torch.argsort(ezs_scores)[self.mask_strategy[0]:self.mask_strategy[1]]
        mask = [1.]*self.latent_size
        mask = [0. if idx in indices else each for idx, each in enumerate(mask)]
        mask = torch.FloatTensor([mask]).cuda()
        return mask * latent_z 
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        cls_representation = outputs[1]

        latent_z = self.bnn.encoder(input=cls_representation)
        cls_representation_recon = self.bnn.decoder(z=latent_z)

        masked_z = self.mask_by_delta_e(latent_z, cls_representation_recon-cls_representation)
        cls_representation_recon = self.bnn.decoder(z=masked_z)

        pooled_output = self.dropout(cls_representation_recon)
        logits = self.classifier(pooled_output)

        pooled_output = self.dropout(cls_representation)
        ori_logits = self.classifier(pooled_output)

        pesudo_labels = torch.argmax(self.sm(ori_logits), dim=1)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = CrossEntropyLoss()
                cross_entropy = loss_fct(logits.view(-1, self.num_labels), pesudo_labels)
                klloss = orthogonal_pairwise_loss(self.bnn.latent_size, cls_representation, cls_representation_recon, self.bnn.linear)
                loss = cross_entropy + klloss

            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                cross_entropy = loss_fct(logits.view(-1, self.num_labels), pesudo_labels)
                klloss = orthogonal_pairwise_loss(self.vae.latent_size, cls_representation, cls_representation_recon, self.vae.bate_layer)
                loss = cross_entropy + klloss
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class BertBNNForSequenceClassificationAnalysis(BertBNNForSequenceClassification):

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        cls_representation = outputs[1]

        latent_z = self.bnn.encoder(input=cls_representation)
        cls_representation_recon = self.bnn.decoder(z=latent_z)

        pooled_output = self.dropout(cls_representation_recon)
        logits = self.classifier(pooled_output)

        pooled_output = self.dropout(cls_representation)
        ori_logits = self.classifier(pooled_output)

        perturbed = self.bnn.latent_perturb(z=latent_z)
        influntial_idxs, influntial_idx_coss, word_cosss = self.bnn.cos_delta_by_dimension_batches(cls_representation_recon-cls_representation, perturbed, outputs.last_hidden_state)
        recon_labels = torch.argmax(self.sm(logits), dim=1)
        pesudo_labels = torch.argmax(self.sm(ori_logits), dim=1)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = CrossEntropyLoss()
                cross_entropy = loss_fct(logits.view(-1, self.num_labels), pesudo_labels)
                klloss = orthogonal_pairwise_loss(self.bnn.latent_size, cls_representation, cls_representation_recon, self.bnn.linear)
                loss = cross_entropy + klloss

                grad_x = torch.autograd.grad(loss, cls_representation_recon, retain_graph=True)

                self.analysis(perturbed, grad_x)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), pesudo_labels) + orthogonal_pairwise_loss(self.bnn.latent_size, cls_representation, cls_representation_recon, self.bnn.linear)
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceAnalysisOutput(
            loss=loss,
            logits=pesudo_labels,
            recon_logits=recon_labels,
            influntial_ids=torch.tensor(influntial_idxs),
            influntial_idx_coss=torch.tensor(influntial_idx_coss),
            part_input_idss=input_ids,
            word_cosss=torch.tensor(word_cosss)
        )


class BertMCDForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        self.dropout = DropoutMC(0.1, True)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )