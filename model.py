import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Function
from torch.utils.checkpoint import checkpoint as grad_checkpoint
import os
from torch import Tensor
import numpy as np
from torch.utils import data
from collections import OrderedDict
from torch.nn.parameter import Parameter
from pytorch_model_summary import summary
import math


class AttentionConv1d(nn.Module):
    def __init__(self, kernel_size, out_channels):
        super(AttentionConv1d, self).__init__()
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.cosine_similarity = nn.CosineSimilarity(dim=1)

    def calculate_similarity(self, embedding, embedding_neighbor):
        similarity = self.cosine_similarity(embedding, embedding_neighbor)
        similarity = torch.unsqueeze(similarity, dim=1)
        return similarity

    def cal_local_attenttion(self, embedding, feature, kernel_size):
        embedding_l = torch.zeros_like(embedding)
        embedding_l[:, :, 1:] = embedding[:, :, :-1]
        similarity_l = self.calculate_similarity(embedding, embedding_l)
        similarity_c = self.calculate_similarity(embedding, embedding)
        embedding_r = torch.zeros_like(embedding)
        embedding_r[:, :, :-1] = embedding[:, :, 1:]
        similarity_r = self.calculate_similarity(embedding, embedding_r)
        similarity = torch.cat([similarity_l, similarity_c, similarity_r], dim=1)  
        # expand for D times
        batch, channel, temporal_length = feature.size()
        similarity_tile = torch.zeros(batch, kernel_size * channel, temporal_length).type_as(feature)
        similarity_tile[:, :channel * 1, :] = similarity[:, :1, :]
        similarity_tile[:, channel * 1:channel * 2, :] = similarity[:, 1:2, :]
        similarity_tile[:, channel * 2:, :] = similarity[:, 2:, :]
        return similarity_tile

    def forward(self, feature, embedding, weight):
        batch, channel, temporal_length = feature.size()
        inp = torch.unsqueeze(feature, dim=3)
        w = torch.unsqueeze(weight, dim=3)

        unfold = nn.Unfold(kernel_size=(self.kernel_size, 1), stride=1, padding=[1, 0])
        inp_unf = unfold(inp)
        # local attention
        attention = self.cal_local_attenttion(embedding, feature, kernel_size=self.kernel_size)
        inp_weight = inp_unf * attention
        inp_unf_t = inp_weight.transpose(1, 2)
        w_t = w.view(w.size(0), -1).t()
        results = torch.matmul(inp_unf_t, w_t)
        out_unf = results.transpose(1, 2)
        out = out_unf.view(batch, self.out_channels, temporal_length)
        return out


class FilterModule(nn.Module):
    def __init__(self):
        super(FilterModule, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        return out

class BaseModule(nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()
        self.conv_1 = nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_1_att = AttentionConv1d(kernel_size=3, out_channels=1024)
        self.conv_2 = nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_2_att = AttentionConv1d(kernel_size=3, out_channels=1024)
        self.lrelu = nn.LeakyReLU()
        self.drop_out = nn.Dropout(0.7)
    def forward(self, x, embedding):
        feat1 = self.lrelu(self.conv_1_att(x, embedding, self.conv_1.weight))
        feat2 = self.lrelu(self.conv_2_att(feat1, embedding, self.conv_2.weight))
        feature = self.drop_out(feat2)
        return feat1, feature

class ClassifierModule(nn.Module):
    def __init__(self):
        super(ClassifierModule, self).__init__()
        self.conv = nn.Conv1d(in_channels=1024, out_channels=2, kernel_size=1, stride=1, padding=0, bias=False)
        self.fc = nn.Linear(2100, 132)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        out = self.sig(x)
        return out
class EmbeddingModule(nn.Module):
    def __init__(self):
        super(EmbeddingModule, self).__init__()
        self.conv_1 = nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv1d(in_channels=512, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU()
    def forward(self, x):
        out = self.lrelu(self.conv_1(x))
        out = self.conv_2(out)
        embedding = F.normalize(out, p=2, dim=1)
        return embedding
class TDL(nn.Module):
    def __init__(self):
        super(TDL, self).__init__()
        self.filter_module = FilterModule()
        self.base_module = BaseModule()
        self.classifier_module = ClassifierModule()
        self.softmax = nn.Softmax(dim=1)
        self.embedding_module = EmbeddingModule()
        self.sig = nn.Sigmoid()
    def forward(self, x):
        #weights = self.filter_module(x)
        #x = weights * x
        embedding = self.embedding_module(x)
        feature1, feature2 = self.base_module(x, embedding)
        feature2 = self.classifier_module(feature2)
        return embedding, feature2


# ---------------------------------------------------------------------------
# Aşama 2 – Pure PyTorch Mamba (mamba-ssm kütüphanesi GEREKTİRMEZ)
# ---------------------------------------------------------------------------

class PureMambaSSM(nn.Module):
    """Selective SSM (S6) çekirdeği – ekstra C++ derlemesi yok."""

    def __init__(self, d_inner: int, d_state: int = 16):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        dt_rank = max(1, d_inner // 16)

        # A: (d_inner, d_state) – log-parameterised, negatif tutmak için -exp
        self.A_log = nn.Parameter(
            torch.log(
                torch.arange(1, d_state + 1, dtype=torch.float32)
                .unsqueeze(0).repeat(d_inner, 1)
            )
        )
        self.D = nn.Parameter(torch.ones(d_inner))                  # skip connection ölçeği
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        # dt bias'ı küçük başlatmak sayısal kararlılığı artırır
        nn.init.uniform_(self.dt_proj.bias, -4.0, -1.0)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, d_inner)
        orig_dtype = x.dtype
        device_type = 'cuda' if x.is_cuda else 'cpu'

        # SSM çekirdeğini her zaman float32'de çalıştır (fp16 overflow önlemi)
        with torch.amp.autocast(device_type=device_type, enabled=False):
            x_f = x.float()
            B, T, D = x_f.shape
            A = -torch.exp(self.A_log.float())                      # (D, N)

            x_dbl = self.x_proj(x_f)                                # (B, T, dt_rank+2N)
            dt_rank = self.dt_proj.in_features
            dt_raw, B_ssm, C = x_dbl.split(
                [dt_rank, self.d_state, self.d_state], dim=-1
            )
            dt = F.softplus(self.dt_proj(dt_raw))                   # (B, T, D)

            # Sıralı seçici tarama (sequential selective scan)
            h = torch.zeros(B, D, self.d_state, device=x.device, dtype=torch.float32)
            ys = []
            for t in range(T):
                # ZOH ayrıklaştırma
                dA_t = torch.exp(dt[:, t].unsqueeze(-1) * A)        # (B, D, N)
                dB_t = dt[:, t].unsqueeze(-1) * B_ssm[:, t].unsqueeze(1)  # (B, D, N)
                h = dA_t * h + dB_t * x_f[:, t].unsqueeze(-1)      # (B, D, N)
                ys.append((h * C[:, t].unsqueeze(1)).sum(-1))        # (B, D)

            y = torch.stack(ys, dim=1)                              # (B, T, D)
            y = y + x_f * self.D                                    # skip connection

        return y.to(orig_dtype)


class MambaBlock(nn.Module):
    """
    Tam Mamba bloğu: norm → in_proj → depthwise-conv → SiLU → SSM → gate → out_proj → artık bağlantı.
    Giriş/Çıkış: (B, T, d_model)
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        d_inner = int(expand * d_model)
        self.norm     = nn.LayerNorm(d_model)
        self.in_proj  = nn.Linear(d_model, d_inner * 2, bias=False)
        # Nedensel (causal) derinlik-boyutlu evrişim; padding = d_conv-1 sağa kırpılır
        self.conv1d   = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=d_inner, bias=True
        )
        self.ssm      = PureMambaSSM(d_inner, d_state)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, T, d_model)
        residual = x
        T = x.shape[1]
        x = self.norm(x)

        xz = self.in_proj(x)                                        # (B, T, d_inner*2)
        x_branch, z = xz.chunk(2, dim=-1)                           # ikisi de (B, T, d_inner)

        # Nedensel depthwise conv: (B, T, d_inner) → (B, d_inner, T) → trim → geri
        x_branch = self.conv1d(x_branch.transpose(1, 2))[..., :T].transpose(1, 2)
        x_branch = F.silu(x_branch)                                 # (B, T, d_inner)

        y = self.ssm(x_branch)                                      # (B, T, d_inner)
        y = y * F.silu(z)                                           # kapı (gate)
        y = self.out_proj(y)                                        # (B, T, d_model)
        return y + residual


class TDL_Mamba(nn.Module):
    """
    Aşama 2 modeli: TCONV (BaseModule) → 2× MambaBlock.

    Boyut akışı (train loop'u feat.transpose(1,2) yapıyor):
        girdi  : (B, 1024, T)
        ESM    : (B, 32,   T)  – yalnızca LESM kaybı için, Mamba'ya GİRMEZ
        Mamba  : (B, T, 1024) → (B, T, 1024)  [iç transpose'lar burada]
        output : (B, 1024, T) → ClassifierModule → (B, 132)
    """

    def __init__(self):
        super(TDL_Mamba, self).__init__()
        self.embedding_module  = EmbeddingModule()                  # ESM: (B,32,T)
        self.mamba_1           = MambaBlock(d_model=1024, d_state=16, d_conv=4, expand=1)
        self.dropout           = nn.Dropout(0.3)
        self.mamba_2           = MambaBlock(d_model=1024, d_state=16, d_conv=4, expand=1)
        self.classifier_module = ClassifierModule()

    def forward(self, x: Tensor):
        # x: (B, 1024, T)
        embedding = self.embedding_module(x)        # (B, 32, T) – LESM loss için ayrı

        x_t = x.transpose(1, 2)                     # (B, T, 1024) – Mamba formatı
        x_t = grad_checkpoint(self.mamba_1, x_t, use_reentrant=False)  # bellek tasarrufu
        x_t = self.dropout(x_t)
        x_t = grad_checkpoint(self.mamba_2, x_t, use_reentrant=False)
        x_t = x_t.transpose(1, 2)                   # (B, 1024, T) – conv formatına geri

        feat_out = self.classifier_module(x_t)      # (B, 132)
        return embedding, feat_out


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    #print(summary(Network(), torch.randn((16, 1024, 1050)), show_input=False))

