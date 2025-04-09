from .kdpl_utils import load_teacher_to_cpu, load_classnames_dictionary, sampling, CustomCLIPTeacher, \
    get_K_max
from .coop import CustomCLIP, CoOp, load_clip_to_cpu,TextEncoder

import os.path as osp
import time
import datetime
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
# from training.reweighting import weight_learner

_tokenizer = _Tokenizer()



class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, is_specific = True):
        super().__init__()
        n_cls = len(classnames)
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        n_ctx = 5
        if is_specific:
            ctx_init  = ctx_init
        else:
            ctx_init  = "a photo of no"
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:

            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class StudentPromptLearner(PromptLearner):

    def __init__(self, cfg, classnames, student_model, is_specific = True):
        super().__init__(cfg, classnames, student_model, is_specific=is_specific)

    def forward(self, prompts_indices=None):
        prompts = super().forward()

        if prompts_indices is not None:
            prompts = prompts[prompts_indices]
        return prompts


# class my_prompt_learner(nn.Module):
#     def __init__(self,  cfg, classnames, student_model):
#         super().__init__()
#         self.prompt_learner_pos =  StudentPromptLearner(cfg, classnames, student_model, is_specific=True)
#         self.prompt_learner_neg =  StudentPromptLearner(cfg, classnames, student_model, is_specific=False)
#
#     def forward(self, prompts_indices=None):


class CustomCLIPStudent(CustomCLIP):
    def __init__(self, cfg, classnames, student_model):
        super().__init__(cfg, classnames, student_model)
        # del self.prompt_learner
        self.prompt_learner = StudentPromptLearner(cfg, classnames, student_model, is_specific=True)
        self.prompt_learner_generic = StudentPromptLearner(cfg, classnames, student_model, is_specific=False)
        self.text_encoder_naga =  TextEncoder(student_model)
        # temp_dim = self.prompt_learner_specific.token_suffix.shape[1] + self.prompt_learner_specific.token_prefix.shape[
        #     1] + self.prompt_learner_specific.n_ctx
        # self.conv_layer = nn.Conv1d(in_channels=temp_dim * 2, out_channels=temp_dim, kernel_size=1)
        self.register_buffer('pre_features', torch.zeros((cfg.DATALOADER.TRAIN_X.BATCH_SIZE, 512), dtype=torch.float16))
        self.register_buffer('pre_weight1', torch.ones((cfg.DATALOADER.TRAIN_X.BATCH_SIZE, 1), dtype=torch.float16))


    def forward(self, image, prompts_indices=None):
        image_features_ = self.image_encoder(image.type(self.dtype))

        prompts_specific = self.prompt_learner(prompts_indices)
        prompts_generic = self.prompt_learner_generic(prompts_indices)
        # combined_prompts = torch.cat([prompts_specific, prompts_generic], dim=1)

        # if prompts_indices is not None:
        #     tokenized_prompts = self.tokenized_prompts[prompts_indices]
        # else:
        #     tokenized_prompts = self.tokenized_prompts
        if prompts_indices is not None:
            text_features_pos = self.text_encoder(prompts_specific, self.prompt_learner.tokenized_prompts[prompts_indices])
            text_features_neg = self.text_encoder_naga(prompts_generic, self.prompt_learner_generic.tokenized_prompts[prompts_indices])
        else:
            text_features_pos = self.text_encoder(prompts_specific, self.prompt_learner.tokenized_prompts)
            text_features_neg = self.text_encoder_naga(prompts_generic, self.prompt_learner_generic.tokenized_prompts)

        image_features = image_features_ / image_features_.norm(dim=-1, keepdim=True)

        text_features_pos = text_features_pos / text_features_pos.norm(dim=-1, keepdim=True)
        text_features_neg = text_features_neg / text_features_neg.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_pos = logit_scale * image_features @ text_features_pos.t()
        logits_neg = logit_scale * image_features @ text_features_neg.t()

        # print("hello")
        return logits_pos, logits_neg, image_features, text_features_pos,text_features_neg
        # if self.training:
        #     return logits_pos, logits_neg, image_features_
        # else:
        #     return logits_pos
        # if self.training:
        #     return logits, image_features_
        # else:
        #     return logits


@TRAINER_REGISTRY.register()
class CoOp_OFF(CoOp):
    """CoOp+KDPL.
       Improving Zero-shot Generalization of Learned Prompts via Unsupervised Knowledge Distillation
       M. Mistretta et al.
       https://arxiv.org/abs/2407.03056
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.KDPL.PREC in ["fp16", "fp32"]

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch, self.epoch, self.batch_idx)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()



    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        self.k_max, classnames = get_K_max(cfg, classnames)

        print(f"Loading Student CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        student_model = load_clip_to_cpu(cfg)
        print(f"Loading Teacher CLIP (backbone: {cfg.TRAINER.KDPL.TEACHER})")
        teacher_model = load_teacher_to_cpu(cfg)
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            student_model.float()
            teacher_model.float()

        print("Building Student CLIP")
        self.student_model = CustomCLIPStudent(cfg, classnames, student_model)
        print("Building Teacher CLIP")
        self.teacher_model = CustomCLIPTeacher(cfg, classnames, teacher_model, self.student_model.logit_scale)

        print("Turning off gradients in the teacher")
        for name, param in self.teacher_model.named_parameters():
            param.requires_grad_(False)

        print("Turning off gradients in the student")
        for name, param in self.student_model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        self.student_model.to(self.device)
        self.teacher_model.to(self.device)

        param_groups = [{'params': self.student_model.prompt_learner.parameters()}, {'params': self.student_model.prompt_learner_generic.parameters()}]
        # self.optim = build_optimizer(self.student_model.prompt_learner, cfg.OPTIM, param_groups)
        self.optim = build_optimizer(None, cfg.OPTIM, param_groups)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.student_model.prompt_learner, self.optim, self.sched)

        self.teacher_model.init_teacher_text_features()

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        hard_label = batch["hard_label"]
        input = input.to(self.device)
        label = label.to(self.device)
        hard_label = hard_label.to(self.device)
        return input, label,hard_label

    def after_train(self):
        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")

        # Close writer
        self.close_writer()

    def cosine_distance_loss(self, logits_pos, logits_neg):
        # 归一化处理
        logits_pos_norm = F.normalize(logits_pos, p=2, dim=1)
        logits_neg_norm = F.normalize(logits_neg, p=2, dim=1)
        # 计算余弦相似度（范围为[-1, 1]）
        cosine_sim = torch.sum(logits_pos_norm * logits_neg_norm, dim=1)
        # 最小化均值，使相似度趋近于-1（方向相反）
        return cosine_sim

    def forward_backward(self, batch, epoch = 0, idx = 0):
        image, label,hard_label = self.parse_batch_train(batch)
        teacher_logits = label
        # teacher_logits_ = self.teacher_model(image)
        propmt_indices = sampling(teacher_logits, K=self.k_max)
        # propmt_indices = None
        logits_pos, logits_neg, image_features, text_features_pos,text_features_neg = self.student_model(image, propmt_indices)
        teacher_logits = teacher_logits[:, propmt_indices]
        teacher_labels = torch.argmax(teacher_logits, dim=1)
        batch_size, num_classes = logits_pos.size()
        teacher_log_prob = F.log_softmax(teacher_logits, dim=-1)
        student_log_prob = F.log_softmax(logits_pos, dim=-1)

        p_teacher = F.softmax(teacher_logits, dim=-1)  # (batch_size, num_classes)
        entropy = -torch.sum(p_teacher * torch.log(p_teacher + 1e-6), dim=-1)  # (batch_size,)
        num_classes = teacher_logits.shape[-1]
        max_entropy = torch.log(torch.tensor(num_classes))
        w = 1 - (entropy / max_entropy)  # (batch_size,)
        # w = w.


        log_probs = F.log_softmax(logits_neg, dim=1)
        probs = log_probs.exp()
        entropy = - (probs * log_probs).sum(dim=1)  # 各样本的熵
        loss_neg = - entropy  # 最小化负熵

        T = 1.0

        # uniform = torch.ones_like(p_teacher, dtype=torch.float32) / num_classes  # (batch_size, num_classes)
        #
        # loss_neg = F.kl_div(
        #     F.log_softmax(logits_neg / T, dim=-1),
        #     uniform,
        #     reduction='none',
        #
        # ).sum(dim=-1, dtype=torch.float32)  # (batch_size,)

        # 一阶差异
        first_order_diff = text_features_pos - text_features_neg  # [num_classes, feature_dim]

        # 类别间的二阶差异（差异向量的差异）
        cascade_diff_loss = 0
        second_order_diffs = []
        unique_classes = teacher_labels.unique().tolist()
        for i in unique_classes:
            for j in unique_classes:
                # 两个类别的差异向量之间的差异
                if i != j:
                    second_diff = first_order_diff[i] - first_order_diff[j]
                    # second_diff = second_diff / second_diff.norm()
                    # second_order_diffs.append((i, j, second_diff))
                    cascade_diff_loss +=  second_diff.mean()

        # for i in range(batch_size):
        #     label_i = teacher_labels[i]
        #     for j in range(batch_size):
        #         label_j = teacher_labels[j]
        #         for idx, (l1, l2, second_diff) in enumerate(second_order_diffs):
        #             if l1 == label_i and l2 == label_j:
        #                 cascade_diff_loss += w[i] * w[j] * second_diff.mean()
        # 级联差异损失
        # cascade_diff_loss = 0
        # # 一阶差异损失 - 已在其他部分实现
        # # 二阶差异损失 - 捕捉类别间的相对关系
        # for i in range(batch_size):
        #     label_i = teacher_labels[i]
        #     for j in range(batch_size):
        #         label_j = teacher_labels[j]
        #         if label_i < label_j:
        #             # 查找这两个类别的二阶差异向量
        #             for idx, (l1, l2, second_diff) in enumerate(second_order_diffs):
        #                 if l1 == label_i and l2 == label_j:
        #                     # 图像特征与二阶差异的相似度
        #                     sim_i = image_features[i] @ second_diff
        #                     sim_j = image_features[j] @ second_diff
        #                     # 标签i的样本应与二阶差异更相似
        #                     cascade_diff_loss += w[i] * w[j] * torch.relu(sim_j - sim_i + 0.2)
        #                     break


        loss_JS = self.cosine_distance_loss(logits_pos, logits_neg)

        loss_F = F.kl_div(student_log_prob, teacher_log_prob, log_target=True, reduction="none").sum(dim=-1)
        loss_R = F.kl_div(teacher_log_prob, student_log_prob, log_target=True, reduction="none").sum(dim=-1)
        alfa = 0.5
        beta = 0.5
        # loss = (alfa * loss_F + beta * loss_R + 0.1 * loss_N + 0.001 * loss_JS)
        loss = (w * alfa * loss_F).mean()  + (w * beta * loss_R).mean() + ((1 - w) * loss_neg).mean() + (w * 0.001 * loss_JS).mean() + 0.1 * cascade_diff_loss
        # loss = (w * alfa * loss_F).mean() + (w * beta * loss_R).mean() + (w * 0.001 * loss_JS).mean()



        self.model_backward_and_update(loss)

        loss_summary = {
            "loss symmetric KL": loss.item(),
            # "loss KL forward": loss_F.item(),
            # "loss KL reverse": loss_R.item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def model_inference(self, images):
        return self.student_model(images)[0]
