import math
from typing import Optional

import torch as th
import torch.nn as nn
import torch.nn.functional as fn


def print_net_size(net: nn.Module):
    param_size = sum(p.numel() for p in net.parameters() if p.requires_grad)
    model_size = sum(p.numel() * p.element_size() for p in net.parameters())
    return param_size, model_size


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = th.exp(th.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = th.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class StepMLP(nn.Module):
    def __init__(self, hidden_dim: int, scale: int = 4, bias: bool = True):
        super().__init__()
        self.net = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * scale, bias=bias),
            nn.Mish(),
            nn.Linear(hidden_dim * scale, hidden_dim, bias=bias),
        )

    def forward(self, x):
        return self.net(x)


class FrLinear(nn.Module):
    def __init__(self, hidden_size, embeds_size, scale, sparsity_threshold=0.01):
        super().__init__()
        self.hidden_size = hidden_size
        self.embeds_size = embeds_size

        self.r_wgt = nn.Parameter(scale * th.randn(embeds_size, embeds_size))
        self.i_wgt = nn.Parameter(scale * th.randn(embeds_size, embeds_size))
        self.r_bias = nn.Parameter(scale * th.randn(embeds_size))
        self.i_bias = nn.Parameter(scale * th.randn(embeds_size))
        self.e_embs = nn.Parameter(th.randn(1, self.embeds_size))
        self.fc = nn.Linear(self.hidden_size * self.embeds_size, self.hidden_size)
        self.sparsity_threshold = sparsity_threshold

    def forward(self, x):
        # input x shape: (n_batch, len, dim)
        b, l = x.shape[0], x.shape[1]
        x = self.frequency_embeding(x)  # (n_batch, dim, len, emb)
        x = th.fft.rfft(x, dim=2, norm='ortho')  # FFT on N dimension
        y = self.frequency_relu_linear(x)
        x = th.fft.irfft(y, n=l, dim=2, norm="ortho")
        x = x.permute(0, 2, 1, 3).reshape(b, l, -1)
        x = self.fc(x)
        return x

    def frequency_embeding(self, x):
        # x shape : (n_batch, n_len, n_dim)
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(-1)
        y = x * self.e_embs  # N*T*1 x 1*D = N*T*D
        return y

    def frequency_relu_linear(self, x):
        o1_real = fn.relu(
            th.einsum('bijd,dd->bijd', x.real, self.r_wgt) - \
            th.einsum('bijd,dd->bijd', x.imag, self.i_wgt) + \
            self.r_bias
        )
        o1_imag = fn.relu(
            th.einsum('bijd,dd->bijd', x.imag, self.r_wgt) + \
            th.einsum('bijd,dd->bijd', x.real, self.i_wgt) + \
            self.i_bias
        )
        y = th.stack([o1_real, o1_imag], dim=-1)
        y = fn.softshrink(y, lambd=self.sparsity_threshold)
        y = th.view_as_complex(y)
        return y


class FrAttention(nn.Module):
    """Frequency Multi-Head Attention module"""

    def __init__(self, n_head, d_model, d_embeds=32, dropout=0.1, normalize_before=True, bias=True):
        super().__init__()
        assert d_model % n_head == 0
        self.hidden_size = d_model
        self.embeds_size = d_embeds
        self.p_dropout = dropout

        self.normalize_before = normalize_before
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.d_v = d_model // n_head

        self.w_qs = FrLinear(hidden_size=self.hidden_size, embeds_size=self.embeds_size, scale=0.05, sparsity_threshold=0.01)
        self.w_ks = FrLinear(hidden_size=self.hidden_size, embeds_size=self.embeds_size, scale=0.05, sparsity_threshold=0.01)
        self.w_vs = FrLinear(hidden_size=self.hidden_size, embeds_size=self.embeds_size, scale=0.05, sparsity_threshold=0.01)

        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=bias)
        nn.init.xavier_uniform_(self.out_proj.weight)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def scaled_dot_production(self, q, k, v, attn_mask=None, dropout=0.0, is_causal=False, need_weights=False):
        bsz, embed_dim = q.shape[0], self.hidden_size
        src_len, tgt_len = v.shape[1], q.shape[1]
        if need_weights:
            B, Nt, E = q.shape
            q_scaled = q * math.sqrt(1.0 / float(E))

            assert not (is_causal and attn_mask is None), "FIXME: is_causal not implemented for need_weights"

            if attn_mask is not None:
                attn_output_weights = th.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
            else:
                attn_output_weights = th.bmm(q_scaled, k.transpose(-2, -1))
            attn_output_weights = fn.softmax(attn_output_weights, dim=-1)
            if dropout > 0.0:
                attn_output_weights = fn.dropout(attn_output_weights, p=dropout)

            attn_output = th.bmm(attn_output_weights, v)

            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
            attn_output = fn.linear(attn_output, self.out_proj.weight, self.out_proj.bias)
            attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
            return attn_output.permute(1, 0, 2), attn_output_weights

        else:
            q = q.view(bsz, tgt_len, self.n_head, self.d_k)
            k = k.view(bsz, src_len, self.n_head, self.d_v)
            v = v.view(bsz, src_len, self.n_head, self.d_v)
            q, k, v = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)  # Transpose for attention dot product: b x n x lq x dv

            attn_output = fn.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout, is_causal=is_causal)
            attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)

            attn_output = fn.linear(attn_output, self.out_proj.weight, self.out_proj.bias)
            attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
            return attn_output.permute(1, 0, 2), None

    def forward(self, q, k, v, mask=None, need_weights=False, is_causal=False):
        residual = q
        if self.normalize_before:
            q = self.layer_norm(q)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q)
        k = self.w_ks(k)
        v = self.w_vs(v)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        output, attn = self.scaled_dot_production(q, k, v, attn_mask=mask, dropout=self.p_dropout, need_weights=need_weights, is_causal=is_causal)
        output = self.dropout(output)
        output += residual

        if not self.normalize_before:
            output = self.layer_norm(output)
        return output, attn

    def interpolate(self, q, k, v, t, qt, mask=None):
        d_k, n_head = self.d_k, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        len_qt = qt.size(1)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs.interpolate(q, t, qt, mask=mask).view(sz_b, len_qt, len_q, -1, n_head, d_k)
        k = self.w_ks.interpolate(k, t, qt).view(sz_b, len_qt, len_k, -1, n_head, d_k)
        v = self.w_vs.interpolate(v, t, qt).view(sz_b, len_qt, len_v, -1, n_head, d_k)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.permute(0, 4, 1, 2, 3, 5), k.permute(0, 4, 1, 2, 3, 5), v.permute(0, 4, 1, 2, 3, 5)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        output, _ = self.attention.interpolate(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_qt, -1)
        output = self.fc(output)

        return output


class FrEncoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_feedforward=2048, d_embeds=32, dropout=0.1, norm_first=True, bias=True, layer_norm_eps=1e-8, activation=fn.gelu, rotary=None):
        super().__init__()
        self.self_attn = FrAttention(n_head=n_head, d_model=d_model, d_embeds=d_embeds, dropout=dropout, bias=bias)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, d_feedforward, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_feedforward, d_model, bias=bias)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if activation is fn.relu or isinstance(activation, th.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is fn.gelu or isinstance(activation, th.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation
        self.rotary = nn.Identity() if rotary is None else rotary

    def forward(self, src, src_mask: Optional[th.Tensor] = None, src_key_padding_mask: Optional[th.Tensor] = None, is_causal: bool = False):
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, is_causal=is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, is_causal=is_causal))
            x = self.norm2(x + self._ff_block(x))

        return x

    def _sa_block(self, x: th.Tensor, attn_mask: Optional[th.Tensor], is_causal=False):
        qk = self.rotary(x)
        x = self.self_attn(qk, qk, x, mask=attn_mask, need_weights=False, is_causal=is_causal)[0]
        return self.dropout1(x)

    def _ff_block(self, x: th.Tensor) -> th.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class FrDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_feedforward=2048, d_embeds=32, dropout=0.1, norm_first=True, bias=True, layer_norm_eps=1e-8, activation=fn.gelu, rotary=None):
        super().__init__()
        self.self_attn = FrAttention(n_head=n_head, d_model=d_model, d_embeds=d_embeds, dropout=dropout, bias=bias)
        self.cross_attn = FrAttention(n_head=n_head, d_model=d_model, d_embeds=d_embeds, dropout=dropout, bias=bias)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, d_feedforward, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_feedforward, d_model, bias=bias)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if activation is fn.relu or isinstance(activation, th.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is fn.gelu or isinstance(activation, th.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation
        self.rotary = nn.Identity() if rotary is None else rotary

    def forward(self, src, mem, src_mask: Optional[th.Tensor] = None, mem_mask: Optional[th.Tensor] = None, is_causal: bool = False):
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, is_causal=is_causal)
            x = x + self._mha_block(self.norm1(x), mem, mem_mask, is_causal=is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, is_causal=is_causal))
            x = self.norm1(x + self._mha_block(x, mem, mem_mask, is_causal=is_causal))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(self, x: th.Tensor, attn_mask: Optional[th.Tensor], is_causal=False):
        qk = self.rotary(x)
        x = self.self_attn(qk, qk, x, mask=attn_mask, need_weights=False, is_causal=is_causal)[0]
        return self.dropout1(x)

    def _mha_block(self, x: th.Tensor, mem: th.Tensor, attn_mask: Optional[th.Tensor], is_causal=False):
        qx = self.rotary(x)
        km = self.rotary(mem)
        x = self.cross_attn(qx, km, mem, mask=attn_mask, need_weights=False, is_causal=is_causal)[0]
        return self.dropout1(x)

    def _ff_block(self, x: th.Tensor) -> th.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


def ut_freformer():
    fl = FrLinear(hidden_size=256, embeds_size=32, scale=0.02)
    x = th.randn(size=(8, 150, 256))  #
    m = th.randn(size=(8, 120, 256))  #
    y = fl(x)
    print("x {}, y {}".format(x.shape, y.shape))

    attn = FrAttention(n_head=4, d_model=256)
    y, at = attn(x, x, x, need_weights=False)
    print("attention : y shape {}, attn shape {}".format(y.shape, (None if at is None else at.shape)))

    enc = FrEncoderLayer(n_head=4, d_model=256)
    _ = enc.train()
    yenc = enc(x)
    print("after enc : {}".format(yenc.shape))

    dec = FrDecoderLayer(n_head=4, d_model=256)
    _ = dec.train()
    ydec = dec(x, m)
    print("after edc : {}".format(ydec.shape))


def main():
    print('test code todo delete')
    ut_freformer()


if __name__ == '__main__':
    main()
