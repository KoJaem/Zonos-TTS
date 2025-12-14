import torch
import torchaudio
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device

# model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-hybrid", device=device)
repo_id = "Zyphra/Zonos-v0.1-transformer"
from config import get_local_model_path, has_local_model, USE_OFFLINE

if USE_OFFLINE:
    if has_local_model(repo_id):
        print(f"Loading {repo_id} from local disk (OFFLINE MODE)...")
        model = Zonos.from_local(
            config_path=get_local_model_path(repo_id, "config.json"),
            model_path=get_local_model_path(repo_id, "model.safetensors"),
            device=device
        )
    else:
        raise FileNotFoundError(f"Offline mode is enabled but local model files for {repo_id} are missing.")
else:
    print(f"Loading {repo_id} from Hugging Face Hub (cache)...")
    model = Zonos.from_pretrained(repo_id, device=device)

wav, sampling_rate = torchaudio.load("assets/exampleaudio.mp3")
speaker = model.make_speaker_embedding(wav, sampling_rate)

torch.manual_seed(421)

cond_dict = make_cond_dict(text="안녕하세요, 여러분. 이번엔 오프라인 모델 이랍니다! 고재민님이 테스트 중이에요.", speaker=speaker, language="ko")
conditioning = model.prepare_conditioning(cond_dict)

codes = model.generate(conditioning)

wavs = model.autoencoder.decode(codes).cpu()
torchaudio.save("sample.wav", wavs[0], model.autoencoder.sampling_rate)
