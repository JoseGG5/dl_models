import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import librosa
from matplotlib import pyplot as plt

from diff_vit import DiffVisionTransformer

"""Based on https://arxiv.org/pdf/2105.04906.

    VICReg (Variance Invariance Covariance Regularization) is a SSL framework that uses
    an encoder to generate two embeddings of two different views of a given image. It's way more efficient than
    methods like SimCLR as it does not need negative samples and is way more explainable than methods like BYOL.

    It that prevents collapsing to a trivial solution (constant embeddings) by applying the following regularization terms:
    1. Invariance: Ensures that the representations of the two augmented views are similar
    2. Covariance: Ensures that the dimensions of each embedding are not correlated (this means that each dimension stores meaningful information)
    3. Variance: Ensures that for a given batch of embeddings, each dimension is not collapsed.
    
    We will try to adapt this to audio data.
    """

class Augmentations(nn.Module):
    def __init__(
            self,

            p_time_mask: float = 0.25,
            time_mask_n: int = 1,
            time_bins: int = 500,
            time_mask_ratio: float = 0.08,

            p_freq_mask: float = 0.25,
            freq_mask_n: int = 1,
            freq_bins: int = 256,
            freq_mask_ratio: float = 0.08,

            p_time_shift: float = 0.4,
            max_time_shift_ratio: float = 0.2,

            p_add_noise: float = 0.8,
            noise_std: float = 0.1,

            p_gain: float = 0.4,
            min_gain: float = 0.7,
            max_gain: float = 1.3
            ):
        """Class that handles the augmentations for each spectrogram.

        Args:
            p_time_mask (float, optional): The probability of applying TimeMasking. Defaults to 0.4.
            time_mask_n (int, optional): Number of times to apply TimeMasking. Defaults to 2.
            time_bins (int, optional): The length of the time dimension in our specs. Defaults to 500.
            time_mask_ratio (float, optional): The ratio of the bins to mask. Defaults to 0.2.
            
            p_freq_mask (float, optional): The probability of applying FreqMasking. Defaults to 0.4.
            freq_mask_n (int, optional): Number of times to apply FreqMasking. Defaults to 2.
            freq_bins (int, optional): The length of the freq dimension in our specs. Defaults to 256.
            freq_mask_ratio (float, optional): The ratio of the bins to mask. Defaults to 0.2.
            
            p_time_shift (float, optional): The probability of applying time shift. Defaults to 0.4.
            max_time_shift_ratio (float, optional): The maximum ratio of the signal to shift. Defaults to 0.2.
            
            p_add_noise (float, optional): The probability to add noise. Defaults to 0.8.
            noise_std (float, optional): The std of the noise to add (the bigger the more random the noise). Defaults to 0.8.

            p_gain (float, optional): The probability to add random gain. Defaults to 0.4.
            min_gain (float, optional): The minimum value that we can get as gain. Defaults to 0.7.
            max_gain (float, optional): The maximum value that we can get as gain. Defaults to 0.7.
        """
        super().__init__()

        self.p_time_mask = p_time_mask
        self.time_mask_n = time_mask_n
        self.time_mask_param = max(1, int(time_bins * time_mask_ratio))
        self.time_mask = T.TimeMasking(time_mask_param=self.time_mask_param)

        self.p_freq_mask = p_freq_mask
        self.freq_mask_n = freq_mask_n
        self.freq_mask_param = max(1, int(freq_bins * freq_mask_ratio))
        self.freq_mask = T.FrequencyMasking(freq_mask_param=self.freq_mask_param)    

        self.p_time_shift = p_time_shift
        self.max_time_shift_ratio = max_time_shift_ratio

        self.p_add_noise = p_add_noise
        self.noise_std = noise_std

        self.p_gain = p_gain
        self.min_gain = min_gain
        self.max_gain = max_gain


    def _apply_time_mask(self, x: torch.tensor) -> torch.tensor:
        for _ in range(self.time_mask_n):
            x = self.time_mask(x)
        return x


    def _apply_freq_mask(self, x: torch.tensor) -> torch.tensor:
        for _ in range(self.freq_mask_n):
            x = self.freq_mask(x)
        return x


    def _add_noise(self, x: torch.tensor):
        if self.noise_std <= 0:
            return x
        noise = torch.randn_like(x) * self.noise_std
        return x + noise
    

    def _time_shift(self, x: torch.tensor):
        if self.max_time_shift_ratio <= 0:
            return x

        _, _, _, T = x.shape  # B C F T
        max_shift = int(T * self.max_time_shift_ratio)
        if max_shift < 1:
            return x

        shift = random.randint(-max_shift, max_shift)
        if shift == 0:
            return x

        return torch.roll(x, shifts=shift, dims=-1)
    

    def _apply_gain(self, x: torch.tensor):
        gain = random.uniform(self.min_gain, self.max_gain)
        return x * gain


    def forward(self, x: torch.tensor) -> torch.tensor:
        
        x_og = x.clone()

        x_btft = x.squeeze(1)  # without channel dim for TimeMasking and FreqMasking to work on the whole batch
        
        if random.random() < self.p_time_mask:
            x_btft = self._apply_time_mask(x_btft)
        if random.random() < self.p_freq_mask:
            x_btft = self._apply_freq_mask(x_btft)
        
        x_aug = x_btft.unsqueeze(1)  # B C F T

        if random.random() < self.p_add_noise:
            x_aug = self._add_noise(x_aug)
        if random.random() < self.p_time_shift:
            x_aug = self._time_shift(x_aug)
        if random.random() < self.p_gain:
            x_aug = self._apply_gain(x_aug)

        return x_og, x_aug


class Expander(nn.Module):
    def __init__(
            self,
            d: int,
            n_mlp: int = 3,
            expanded_d: int = 8192,
            dropout_rate: float = 0.0,

    ):
        """ The expander of the embeddings (h_phi). Architecture details are shown in page 5, section 4.2

        Args:
            d (int): The number of dimensions of the embeddings generated by the encoder
            n_mlp (int, optional): Number of mlps. Defaults to 3.
            expanded_d (int, optional): The dimensionality that we expand our embeddings. Defaults to 8192.
            dropout_rate (float, optional): Dropout rate, typically 0 in SSL. Defaults to 0.0.
        """
        super().__init__()

        layers = []
        for i in range(n_mlp):
            in_dim = d if i == 0 else expanded_d
            out_dim = expanded_d

            if i != n_mlp - 1:
                layers.append(
                    nn.Sequential(
                        nn.Linear(in_dim, out_dim),
                        nn.BatchNorm1d(out_dim),
                        nn.ReLU(inplace=True),
                        nn.Dropout(dropout_rate)
                    )
                )
            else:
                layers.append(
                    nn.Linear(in_dim, out_dim)
                )

        self.mlps = nn.ModuleList(layers)
    

    def forward(self, x: torch.tensor):
        for mlp in self.mlps:
            x = mlp(x)
        return x


class VicReg(nn.Module):
    def __init__(
            self,
            encoder,
            expander: Expander,

            p_time_mask: float = 0.5,
            time_mask_n: int = 1,
            time_bins: int = 500,
            time_mask_ratio: float = 0.1,

            p_freq_mask: float = 0.5,
            freq_mask_n: int = 1,
            freq_bins: int = 256,
            freq_mask_ratio: float = 0.1,

            p_time_shift: float = 0.4,
            max_time_shift_ratio: float = 0.2,

            p_add_noise: float = 0.8,
            noise_std: float = 0.1,

            p_gain: float = 0.4,
            min_gain: float = 0.7,
            max_gain: float = 1.3
            ):
        super().__init__()
        
        self.encoder = encoder
        self.expander = expander

        self.augmenter = Augmentations(
            p_time_mask=p_time_mask,
            time_mask_n=time_mask_n,
            time_bins=time_bins,
            time_mask_ratio=time_mask_ratio,
            p_freq_mask=p_freq_mask,
            freq_mask_n=freq_mask_n,
            freq_bins=freq_bins,
            freq_mask_ratio=freq_mask_ratio,
            p_time_shift=p_time_shift,
            max_time_shift_ratio=max_time_shift_ratio,
            p_add_noise=p_add_noise,
            noise_std=noise_std,
            p_gain=p_gain,
            min_gain=min_gain,
            max_gain=max_gain
            )
        
    
    def forward(self, x: torch.tensor):
        _, x_aug0 = self.augmenter(x)
        _, x_aug1 = self.augmenter(x)

        emb0, _ = self.encoder(x_aug0)
        emb1, _ = self.encoder(x_aug1)

        expanded_emb0 = self.expander(emb0)
        expanded_emb1 = self.expander(emb1)

        return expanded_emb0, expanded_emb1



###################### Loss function ######################


class VicRegLoss(nn.Module):
    def __init__(self, eps: float = 1e-4, gamma: float = 1, var_coeff: float = 25.0, sim_coeff: float = 25.0):
        super(VicRegLoss, self).__init__()

        self.eps = eps
        self.gamma = gamma
        self.var_coeff = var_coeff
        self.sim_coeff = sim_coeff

    def forward(self, emb_0: torch.tensor, emb_1: torch.tensor) -> float:
        
        # embeddings shape -> (B, D)
        B, D = emb_0.shape

        # Invariance (this ensures embedings are similar)
        inv_term = ((emb_0 - emb_1) ** 2).mean()

        # Variance (ensures that each dimension thas not collapse to a constant)
        sigma_0 = torch.sqrt(emb_0.var(dim=0) + self.eps)
        sigma_1 = torch.sqrt(emb_1.var(dim=0) + self.eps)
        var_term_0 = torch.sum(F.relu(self.gamma - sigma_0)) / D
        var_term_1 = torch.sum(F.relu(self.gamma - sigma_1)) / D
        var_term = var_term_0 + var_term_1

        # Covariance (avoid all dimensions of one embedding to by copies)
        emb_0_centered = emb_0 - emb_0.mean(dim=0)
        cov_emb_0 = (emb_0_centered.T @ emb_0_centered) / (B - 1)
        emb_1_centered = emb_1 - emb_1.mean(dim=0)
        cov_emb_1 = (emb_1_centered.T @ emb_1_centered) / (B - 1)
        
        diagonal_mask = torch.eye(cov_emb_0.size(0), dtype=torch.bool, device=emb_0.device)
        off_diagonal_mask = ~diagonal_mask
        off_diagonal_elements_0 = cov_emb_0[off_diagonal_mask]
        off_diagonal_elements_1 = cov_emb_1[off_diagonal_mask]
        cov_term_0 = (1 / D) * torch.sum(off_diagonal_elements_0 ** 2)
        cov_term_1 = (1 / D) * torch.sum(off_diagonal_elements_1 ** 2)
        cov_term = cov_term_0 + cov_term_1

        loss = self.sim_coeff * inv_term + self.var_coeff * var_term + self.gamma * cov_term
        return loss


if __name__ == "__main__":

    PATH = r"C:\Users\Jose Antonio\Desktop\SanctSound_OC02_02_671117349_190828074416._12425_12430.wav"
    y, sr = torchaudio.load(PATH)

    spectrogram = T.Spectrogram(n_fft=1024, power=1, hop_length=234)  # to obtain a 513 1024 spec size (divisible by 16)
    spec = spectrogram(y)[..., :1024]
    print(sr)
    print(spec.shape)
    
    spec = torch.Tensor(librosa.amplitude_to_db(spec))
    spec_resized = nn.MaxPool2d(kernel_size=2)(spec).unsqueeze(0).repeat(10, 1, 1, 1)  # 10, 1, 256, 512 (B, C, F, T)

    augmenter = Augmentations()

    def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
        if ax is None:
            _, ax = plt.subplots(1, 1)
        if title is not None:
            ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.imshow(specgram[0, 0], origin="lower", aspect="auto", interpolation="nearest")

    for i in range(5):
        og_spec, aug_spec = augmenter(spec_resized)

        fig, axs = plt.subplots(2, 1)
        plot_spectrogram(og_spec, title="original", ax=axs[0])
        plot_spectrogram(aug_spec, title="augmented", ax=axs[1])

        plt.show()

    encoder = DiffVisionTransformer(d=768, img_size=(256,512), patch_size=(16,16), intermediate_dim=768*4, dropout_rate_blocks=0)
    expander = Expander(d=768)
    vicreg = VicReg(encoder=encoder, expander=expander)

    criterion = VicRegLoss()

    vicreg.eval()
    with torch.no_grad():
        expanded_0, expanded_1 = vicreg(og_spec)

        print(expanded_0.shape)
        print(expanded_1.shape)

        loss = criterion(expanded_0, expanded_1)
        print(loss)




