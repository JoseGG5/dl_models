import torch
import torch.nn as nn


from vit import MultiHeadAttention

class FFN(nn.Module):
    """ A simple multilayer percentron """
    def __init__(
            self,
            d: int,
            intermediate_dim: int,
            dropout_rate: float
            ) -> None:
        super().__init__()
        
        self.proj1 = nn.Linear(d, intermediate_dim)
        self.swish = nn.SiLU()
        self.proj2 = nn.Linear(intermediate_dim, d)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.proj1(x)
        x = self.swish(x)
        x = self.dropout(x)
        x = self.proj2(x)
        x = self.dropout(x)

        return x


class Conv(nn.Module):
    def __init__(self, d: int, dropout_rate: float):
        super().__init__()

        self.ln = nn.LayerNorm(normalized_shape=(d))
        self.pointwise_conv_dup = nn.Conv1d(in_channels=d, out_channels=d*2, kernel_size=1)  # 1x1 standard conv duplicating channels dim (d)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(in_channels=d, out_channels=d, kernel_size=31, padding=15, groups=d)  # conv with groups=d so that each channel uses its own kernel 
        self.bn = nn.BatchNorm1d(num_features=d)
        self.swish = nn.SiLU()
        self.pointwise_conv = nn.Conv1d(in_channels=d, out_channels=d, kernel_size=1)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ x -> (B, P, D)"""

        x = self.ln(x)
        x = x.transpose(1, 2)  # so that d is the channel dim
        x = self.pointwise_conv_dup(x)
        x = self.glu(x)
        x = self.depthwise_conv(x)
        x = self.bn(x)
        x = self.swish(x)
        x = self.pointwise_conv(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)

        return x


class ConformerBlock(nn.Module):
    def __init__(self, d: int, dropout_rate: float, attention_head_size: int = 64, n_heads: int = 12, bias: bool = True):
        super().__init__()
        
        self.ffn = FFN(d=d, intermediate_dim=4*d, dropout_rate=dropout_rate)
        self.mhsa = MultiHeadAttention(
            d=d,
            dropout_rate=dropout_rate,
            attention_head_size=attention_head_size,
            n_heads=n_heads,
            bias=bias
            )
        self.conv = Conv(d=d, dropout_rate=dropout_rate)
        self.ln = nn.LayerNorm(normalized_shape=(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = x + 0.5*self.ffn(x)
        x = x + self.mhsa(x)[0]
        x = x + self.conv(x)
        x = x + 0.5*self.ffn(x)
        x = self.ln(x)

        return x
    

class Conformer(nn.Module):
    def __init__(self, input_dim: int, d: int, dropout_rate: float, n_blocks: int = 12, attention_head_size: int = 64, n_heads: int = 12, bias: bool = True):
        super().__init__()

        self.linear = nn.Linear(in_features=input_dim, out_features=768)  # in case the frontend + tokenizer doesn't yield d=768
        self.dropout = nn.Dropout(p=dropout_rate)
        self.blocks = nn.ModuleList([ConformerBlock(
            d=d,
            dropout_rate=dropout_rate,
            attention_head_size=attention_head_size,
            n_heads=n_heads,
            bias=bias
            ) for _ in range(n_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ x -> (B, P, D)"""
        x = self.linear(x)
        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x)

        return x


        
if __name__ == "__main__":
    # get a signal
    signal = torch.rand(size=(1, 48_000*5,))

    # audio frontend (could be log-mel, linear spec, scott, etc)
    spec = torch.rand(size=(1, 256, 512))

    # patchify + embed
    x = torch.rand(size=(1, 512, 768))

    cf = Conformer(input_dim=768, d=768, dropout_rate=0.5)
    with torch.no_grad():
        x = cf(x)

    print(x.shape)