import os
import pdb
import copy
import torch

from torch import nn
from torch.nn import functional as F
from safetensors.torch import save_file
from huggingface_hub import PyTorchModelHubMixin

import torch
from torchaudio.models import hubert_pretrain_base

import sys
from pathlib import Path
sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1])))

class BabyHuBERTWrapper(
    nn.Module,
    PyTorchModelHubMixin, 
    repo_url="https://github.com/tiantiaf0627/childvox-release"
):
    def __init__(
        self, 
        pretrain_model="babyhubert", 
        hidden_dim=256,
        finetune_method="lora",
        lora_rank=16,
        freeze_params=True,
        output_class_num=4,
        max_audio_duration=15,
        fold_idx=1,
        load_pretrained=False,
        label_list=None,
        id2label=None,
    ):
        super(BabyHuBERTWrapper, self).__init__()

        if id2label is not None:
            id2label = {int(k): v for k, v in id2label.items()}
        elif label_list is not None:
            id2label = {i: l for i, l in enumerate(label_list)}

        if label_list is None and id2label is not None:
            label_list = [id2label[i] for i in sorted(id2label)]

        self.label_list = label_list
        self.id2label   = id2label
        self.label2id   = {v: k for k, v in id2label.items()} if id2label else None

        if label_list is not None: output_class_num = len(label_list)

        model = hubert_pretrain_base(num_classes=500)

        if load_pretrained:
            state_dict = torch.load("/scratch1/tiantiaf/model/models--MarvinLvn--BabyHuBERT/snapshots/3736078db7fc4703222e4bb52da9f59871425243/BabyHuBERT.ckpt", map_location="cpu")
            state_dict = {k.replace("model.", ""): v for k, v in state_dict["state_dict"].items()}
            model.load_state_dict(state_dict)
        encoder = model.wav2vec2
        encoder.eval()

        self.backbone_model             = model
        self.finetune_method            = finetune_method

        self.freeze_params = freeze_params
        for _, p in self.backbone_model.named_parameters(): p.requires_grad = False
        
        # 6. Downstream models
        # hidden_dim = 768
        self.model_seq = nn.Sequential(
            nn.Conv1d(768, hidden_dim, 1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv1d(hidden_dim, hidden_dim, 1, padding=0),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv1d(hidden_dim, hidden_dim, 1, padding=0)
        )
        num_layers = 12
        self.weights = nn.Parameter(torch.ones(num_layers)/num_layers)
        self.fold_idx = fold_idx
        
        self.out_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_class_num),
        )
        
    def forward(self, x, length=None, return_feature=False):
        features, length = self.backbone_model.wav2vec2.extract_features(x, lengths=length)
        
        # 2. get length and mask
        if length is not None:
            length = length.cuda()

        # 4. stacked feature
        stacked_feature = torch.stack(features, dim=0)
        
        # 5. Weighted sum
        _, *origin_shape = stacked_feature.shape
        # Return transformer enc outputs [num_enc_layers, B, T, D]
        stacked_feature = stacked_feature.view(12, -1)
        norm_weights = F.softmax(self.weights, dim=-1)
        
        # Perform weighted average
        weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
        features = weighted_feature.view(*origin_shape)
        
        # 6. Pass the weighted average to point-wise 1D Conv
        # B x T x D
        features = features.transpose(1, 2)
        features = self.model_seq(features)
        features = features.transpose(1, 2)
        
        # 7. Pooling
        if length is not None:
            mean, std = list(), list()
            for snt_id in range(features.shape[0]):
                # Avoiding padded time steps
                actual_size = length[snt_id]
                mean.append(torch.mean(features[snt_id, 0:actual_size, ...], dim=0))
            features = torch.stack(mean)
        else:
            features = torch.mean(features, dim=1)

        # 8. Output predictions
        # B x D
        predicted = self.out_layer(features)
        if return_feature: return predicted, features
        return predicted
    
    # From huggingface
    def get_feat_extract_output_lengths(self, input_length):
        """
        Computes the output length of the convolutional layers
        """
        conv_kernel = [
            10,
            3,
            3,
            3,
            3,
            2,
            2
        ]

        conv_stride = [
            5,
            2,
            2,
            2,
            2,
            2,
            2
        ]
    
        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (input_length - kernel_size) // stride + 1
        for kernel_size, stride in zip(conv_kernel, conv_stride):
            input_length = _conv_out_length(input_length, kernel_size, stride)
        return input_length

    def _save_pretrained(self, save_directory: Path) -> None:
        state_dict = {
            k: v.contiguous() for k, v in self.state_dict().items()
        }
        out_dir = Path(save_directory) / f"fold_{self.fold_idx}"
        out_dir.mkdir(parents=True, exist_ok=True)
        save_file(state_dict, str(out_dir / "model.safetensors"))

    @classmethod
    def _from_pretrained(cls, *, model_id, revision, cache_dir, force_download,
                        proxies, resume_download, local_files_only, token,
                        map_location="cpu", strict=False, fold_idx=1, **model_kwargs):
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_model

        model = cls(**model_kwargs)
        model_file = hf_hub_download(
            repo_id=model_id, filename=f"fold_{fold_idx}/model.safetensors",
            revision=revision, cache_dir=cache_dir, force_download=force_download,
            proxies=proxies, resume_download=resume_download,
            local_files_only=local_files_only, token=token,
        )
        load_model(model, model_file, strict=strict, device=map_location)
        return model
