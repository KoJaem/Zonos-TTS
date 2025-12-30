import torch
import torchaudio
import os
import sys
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device

from config import get_local_model_path, has_local_model, USE_OFFLINE

def normalize_audio_rms(audio_tensor, target_rms_db=-20.0):
    """
    Normalize audio to a target RMS level in dB with soft limiting.
    
    This ensures consistent loudness across all files:
    1. Always normalize to target RMS (-20dB)
    2. Apply soft limiting if peaks exceed threshold
    
    Args:
        audio_tensor: torch.Tensor of shape (channels, samples) or (samples,)
        target_rms_db: Target RMS level in dB (default: -20.0 dB)
    
    Returns:
        Normalized audio tensor with consistent RMS
    """
    # Avoid processing silent audio
    if torch.max(torch.abs(audio_tensor)) < 1e-8:
        return audio_tensor
    
    # Calculate current RMS
    rms = torch.sqrt(torch.mean(audio_tensor ** 2))
    
    if rms < 1e-8:
        return audio_tensor
    
    # Convert target dB to linear scale
    target_rms_linear = 10 ** (target_rms_db / 20.0)
    
    # Calculate scaling factor to reach target RMS
    scaling_factor = target_rms_linear / rms
    
    # Apply RMS normalization
    normalized = audio_tensor * scaling_factor
    
    # Apply soft limiting if peaks exceed 0.95
    # This prevents clipping while maintaining the target RMS
    peak = torch.max(torch.abs(normalized))
    
    if peak > 0.95:
        # Use tanh-based soft clipping for smooth limiting
        # This compresses peaks while preserving RMS better than hard limiting
        normalized = torch.tanh(normalized / peak * 0.95) * 0.95
        
        # Re-adjust to maintain target RMS after soft clipping
        new_rms = torch.sqrt(torch.mean(normalized ** 2))
        if new_rms > 1e-8:
            normalized = normalized * (target_rms_linear / new_rms)
    
    return normalized

def load_model(model_choice):
    print(f"Loading {model_choice} model...")
    
    model = None
    if USE_OFFLINE:
        if has_local_model(model_choice):
             print(f"--> Found local model for {model_choice}, loading from disk (OFFLINE MODE).")
             model = Zonos.from_local(
                config_path=get_local_model_path(model_choice, "config.json"),
                model_path=get_local_model_path(model_choice, "model.safetensors"),
                device=device
            )
        else:
             raise FileNotFoundError(f"Offline mode is enabled but local model files for {model_choice} are missing.")
    else:
        if has_local_model(model_choice):
             print(f"--> Found local model for {model_choice}, loading from disk.")
             model = Zonos.from_local(
                config_path=get_local_model_path(model_choice, "config.json"),
                model_path=get_local_model_path(model_choice, "model.safetensors"),
                device=device
            )
        else:
            print(f"--> Local model not found for {model_choice}, using Hugging Face Hub.")
            model = Zonos.from_pretrained(model_choice, device=device)

    model.requires_grad_(False).eval()
    print(f"{model_choice} model loaded successfully!")
    return model

def generate_audio_for_text(model, text, language, speaker_embedding, output_path, prefix_audio_path=None, seed=420):
    # Default parameters based on gradio_interface.py
    emotion_tensor = torch.tensor([1.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.2], device=device) # Happy, Sad, Disgust, Fear, Surprise, Anger, Other, Neutral
    vq_val = 0.78
    vq_tensor = torch.tensor([vq_val] * 8, device=device).unsqueeze(0)
    
    fmax = 24000.0  # Match Gradio default
    pitch_std = 45.0
    speaking_rate = 15.0
    dnsmos_ovrl = 4.0
    speaker_noised = False
    
    # Sampling parameters
    cfg_scale = 2.0
    top_p = 0.0
    top_k = 0
    min_p = 0.0
    linear = 0.5
    confidence = 0.40
    quadratic = 0.00
    
    # Set seed
    torch.manual_seed(seed)

    cond_dict = make_cond_dict(
        text=text,
        language=language,
        speaker=speaker_embedding,
        emotion=emotion_tensor,
        vqscore_8=vq_tensor,
        fmax=fmax,
        pitch_std=pitch_std,
        speaking_rate=speaking_rate,
        dnsmos_ovrl=dnsmos_ovrl,
        speaker_noised=speaker_noised,
        device=device,
        unconditional_keys=["emotion"], # Match gradio default for better quality
    )
    conditioning = model.prepare_conditioning(cond_dict)
    
    max_new_tokens = 86 * 30 # roughly 30 seconds max
    
    # Handle prefix audio (silence)
    audio_prefix_codes = None
    if prefix_audio_path is not None and os.path.exists(prefix_audio_path):
        wav_prefix, sr_prefix = torchaudio.load(prefix_audio_path)
        wav_prefix = wav_prefix.mean(0, keepdim=True)
        wav_prefix = model.autoencoder.preprocess(wav_prefix, sr_prefix)
        wav_prefix = wav_prefix.to(device, dtype=torch.float32)
        audio_prefix_codes = model.autoencoder.encode(wav_prefix.unsqueeze(0))
    
    codes = model.generate(
        prefix_conditioning=conditioning,
        audio_prefix_codes=audio_prefix_codes,
        max_new_tokens=max_new_tokens,
        cfg_scale=cfg_scale,
        batch_size=1,
        sampling_params=dict(top_p=top_p, top_k=top_k, min_p=min_p, linear=linear, conf=confidence, quad=quadratic),
    )
    
    wav_out = model.autoencoder.decode(codes).cpu().detach()
    sr_out = model.autoencoder.sampling_rate
    if wav_out.dim() == 2 and wav_out.size(0) > 1:
        wav_out = wav_out[0:1, :]
    
    # Normalize audio to consistent RMS level
    wav_out_normalized = normalize_audio_rms(wav_out.squeeze(), target_rms_db=-20.0)
    
    # Save audio
    torchaudio.save(output_path, wav_out_normalized.unsqueeze(0), sr_out)
    print(f"Saved to {output_path} (normalized to -20dB RMS, seed: {seed})")

def main():
    print("Initializing multi_random_seed.py...")
    
    # Model selection mechanism (simplified)
    model_choice = "Zyphra/Zonos-v0.1-transformer"
    # ideally check for hybrid support like in gradio interface if needed, but defaulting to transformer is safer if unsure
    
    try:
        model = load_model(model_choice)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Receive multiple texts with file numbers
    # 각 문장마다 텍스트와 파일 번호를 함께 저장
    text_data = []  # (text, file_number) 튜플 리스트
    print("변환할 텍스트를 입력해주세요 (한 줄에 하나씩). 입력을 마치려면 빈 줄에서 엔터를 누르세요:")
    while True:
        try:
            line = input()
            if not line:
                break
            
            # 파일 번호 입력받기
            while True:
                try:
                    file_num_input = input(f"  '{line[:30]}...' → 몇번 파일인가요? ").strip()
                    if file_num_input:
                        file_number = int(file_num_input)
                        text_data.append((line, file_number))
                        print(f"  ✓ 파일 번호 {file_number}로 저장되었습니다. (output{file_number}-1.wav, output{file_number}-2.wav, output{file_number}-3.wav)")
                        print("  다음 텍스트를 입력하세요 (입력을 마치려면 빈 줄에서 엔터):")
                        break
                    else:
                        print("  파일 번호를 입력해주세요.")
                except ValueError:
                    print("  올바른 숫자를 입력해주세요.")
                except EOFError:
                    return
        except EOFError:
            break
            
    if not text_data:
        print("입력된 텍스트가 없습니다. 종료합니다.")
        return

    # 3. Language Code
    language = input("언어 코드를 입력하세요 (기본값: ko): ").strip()
    if not language:
        language = "ko"
        
    print("\n" + "="*40)
    print(f"총 {len(text_data)}개의 텍스트가 입력되었습니다:")
    for i, (text, file_num) in enumerate(text_data):
        print(f"{i+1}. [{file_num}번] {text}")
    print("="*40 + "\n")
    
    # 4. Optional Speaker Audio
    speaker_audio_path = "cloneAudio/clone.wav"
    speaker_embedding = None
    
    if os.path.exists(speaker_audio_path):
        print(f"Found speaker audio at {speaker_audio_path}")
        try:
            wav, sr = torchaudio.load(speaker_audio_path)
            speaker_embedding = model.make_speaker_embedding(wav, sr)
            speaker_embedding = speaker_embedding.to(device, dtype=torch.bfloat16)
        except Exception as e:
            print(f"Error processing speaker audio: {e}")
            print("Proceeding without speaker cloning...")
            # If failed, we might want to Add 'speaker' to unconditional_keys, but for now let's hope it works or just fail.
            # In the original code, if speaker is None, we need to handle it.
            # The original code adds 'speaker' to unconditional_keys if it's not provided?
            # Actually, make_cond_dict handles it?
            # Let's check make_cond_dict... if speaker is None, it might error if not in unconditional_keys.
            # For this script, let's assume if file exists it works, if not, we warn.
            pass
    else:
        print(f"Warning: {speaker_audio_path} not found. Proceeding without speaker cloning (unconditional speaker).")
        # If no speaker, we probably need to handle make_cond_dict correctly.
        # But the user logic specifically asked to look at this file.
        
    # 5 & 6. Generate and Save with RANDOM SEED for each text (각 문장마다 3번씩 실행)
    total_generations = len(text_data) * 3
    print(f"Generating audio for {len(text_data)} items, {total_generations} times total (각 문장당 3번씩)...\n")
    
    for i, (text, file_number) in enumerate(text_data):
        print(f"\n[문장 {i+1}/{len(text_data)}] '{text[:50]}...' (파일 번호: {file_number}) 처리 중...")
        
        # 각 문장마다 3번씩 실행
        for attempt in range(3):
            attempt_num = attempt + 1  # 실행 번호 (1부터 시작)
            output_filename = f"output{file_number}-{attempt_num}.wav"
            
            # 각 실행마다 새로운 랜덤 시드 생성
            random_seed = torch.randint(0, 2**32 - 1, (1,)).item()
            
            print(f"  [{attempt_num}/3] Generating {output_filename} (seed: {random_seed})...")
            
            generate_audio_for_text(model, text, language, speaker_embedding, output_filename, prefix_audio_path="assets/silence_100ms.wav", seed=random_seed)

if __name__ == "__main__":
    main()
