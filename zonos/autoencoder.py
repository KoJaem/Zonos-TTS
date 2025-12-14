import math

import torch
import torchaudio
from transformers.models.dac import DacModel


class DACAutoencoder:
    def __init__(self):
        super().__init__()
        super().__init__()
        from config import USE_OFFLINE, LOCAL_MODELS_DIR
        
        dac_repo = "descript/dac_44khz"
        if USE_OFFLINE:
            # Construct the local path: models/descript/dac_44khz
            local_dac_path = LOCAL_MODELS_DIR / dac_repo
            if local_dac_path.exists():
                print(f"Loading '{dac_repo}' from local disk: {local_dac_path} (OFFLINE MODE)")
                self.dac = DacModel.from_pretrained(str(local_dac_path))
            else:
                 raise FileNotFoundError(f"Offline mode is enabled but local path for {dac_repo} is missing: {local_dac_path}")
        else:
            self.dac = DacModel.from_pretrained(dac_repo)
        self.dac.eval().requires_grad_(False)
        self.codebook_size = self.dac.config.codebook_size
        self.num_codebooks = self.dac.quantizer.n_codebooks
        self.sampling_rate = self.dac.config.sampling_rate

    def preprocess(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        wav = torchaudio.functional.resample(wav, sr, 44_100)
        right_pad = math.ceil(wav.shape[-1] / 512) * 512 - wav.shape[-1]
        return torch.nn.functional.pad(wav, (0, right_pad))

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        return self.dac.encode(wav).audio_codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        with torch.autocast(self.dac.device.type, torch.float16, enabled=self.dac.device.type != "cpu"):
            return self.dac.decode(audio_codes=codes).audio_values.unsqueeze(1).float()
