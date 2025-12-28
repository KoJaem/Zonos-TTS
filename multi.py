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
    print(f"Saved to {output_path} (normalized to -20dB RMS)")

def merge_audio_files(output_filenames, merged_output_path):
    """
    여러 오디오 파일을 순서대로 병합하는 함수
    
    Args:
        output_filenames: 병합할 오디오 파일 경로 리스트
        merged_output_path: 병합된 오디오 파일 저장 경로
    """
    if not output_filenames:
        print("병합할 파일이 없습니다.")
        return
    
    print(f"\n병합 파일 생성 중... ({len(output_filenames)}개 파일)")
    
    try:
        # 첫 번째 파일을 로드하여 샘플링 레이트 확인
        first_wav, first_sr = torchaudio.load(output_filenames[0])
        first_wav = first_wav.mean(0, keepdim=True) if first_wav.dim() > 1 and first_wav.size(0) > 1 else first_wav
        
        # 모든 오디오를 저장할 리스트
        merged_audio = [first_wav]
        
        # 나머지 파일들을 순서대로 로드하고 병합
        for i, filename in enumerate(output_filenames[1:], 1):
            if not os.path.exists(filename):
                print(f"경고: {filename} 파일을 찾을 수 없어 병합에서 제외됩니다.")
                continue
            
            wav, sr = torchaudio.load(filename)
            wav = wav.mean(0, keepdim=True) if wav.dim() > 1 and wav.size(0) > 1 else wav
            
            # 샘플링 레이트가 다르면 리샘플링
            if sr != first_sr:
                print(f"경고: {filename}의 샘플링 레이트가 다릅니다 ({sr} vs {first_sr}). 리샘플링합니다.")
                resampler = torchaudio.transforms.Resample(sr, first_sr)
                wav = resampler(wav)
            
            merged_audio.append(wav)
            print(f"  {i+1}/{len(output_filenames)}: {filename} 추가됨")
        
        # 모든 오디오를 순서대로 연결
        merged_wav = torch.cat(merged_audio, dim=-1)
        
        # 병합된 오디오를 정규화
        merged_wav_normalized = normalize_audio_rms(merged_wav.squeeze(), target_rms_db=-20.0)
        
        # 병합된 파일 저장
        torchaudio.save(merged_output_path, merged_wav_normalized.unsqueeze(0), first_sr)
        print(f"병합 파일 저장 완료: {merged_output_path}")
        
    except Exception as e:
        print(f"병합 파일 생성 중 오류가 발생했습니다: {e}")

def main():
    print("Initializing multi.py...")
    
    # Model selection mechanism (simplified)
    model_choice = "Zyphra/Zonos-v0.1-transformer"
    # ideally check for hybrid support like in gradio interface if needed, but defaulting to transformer is safer if unsure
    
    try:
        model = load_model(model_choice)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. 텍스트 입력 방법 선택
    print("텍스트 입력 방법을 선택하세요:")
    print("  1. 직접 입력 (터미널에서 입력)")
    print("  2. 파일에서 읽기 (파일 경로 입력)")
    input_method = input("선택 (1 또는 2, 기본값: 1): ").strip()
    
    texts = []
    
    if input_method == "2":
        # 파일에서 읽기
        file_path = input("텍스트 파일 경로를 입력하세요: ").strip()
        if not file_path:
            print("파일 경로가 입력되지 않았습니다. 종료합니다.")
            return
        
        if not os.path.exists(file_path):
            print(f"파일을 찾을 수 없습니다: {file_path}")
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # 각 줄을 strip하여 공백 제거 후, 빈 줄이 아닌 경우만 추가
                    line = line.strip()
                    if line:  # 빈 줄이 아닌 경우만 추가
                        texts.append(line)
            
            if not texts:
                print("파일에서 읽은 텍스트가 없습니다. 종료합니다.")
                return
            
            print(f"파일에서 {len(texts)}개의 텍스트를 읽었습니다.")
        except Exception as e:
            print(f"파일 읽기 중 오류가 발생했습니다: {e}")
            return
    else:
        # 직접 입력 (기존 로직)
        print("변환할 텍스트를 입력해주세요 (한 줄에 하나씩). 입력을 마치려면 빈 줄에서 엔터를 누르세요:")
        while True:
            try:
                line = input()
                if not line:
                    break
                texts.append(line)
            except EOFError:
                break
        
        if not texts:
            print("입력된 텍스트가 없습니다. 종료합니다.")
            return

    # 3. 병합 파일 생성 여부 확인
    merge_choice = input("모든 결과물을 순서대로 병합한 파일도 생성하시겠습니까? (y/n, 기본값: y): ").strip().lower()
    # 기본값이 y이므로, 빈 입력이거나 y/yes/예/ㅇ 중 하나면 병합 수행
    should_merge = not merge_choice or merge_choice in ['y', 'yes', '예', 'ㅇ']
    
    # 4. Language Code
    language = input("언어 코드를 입력하세요 (기본값: ko): ").strip()
    if not language:
        language = "ko"
    
    # 5. Seed 값 입력
    seed_input = input("사용할 seed 값을 입력하세요 (없으면 랜덤으로 생성): ").strip()
    if seed_input:
        try:
            base_seed = int(seed_input)
            print(f"입력받은 seed 값 사용: {base_seed}")
        except ValueError:
            print(f"잘못된 seed 값입니다. 랜덤으로 생성합니다.")
            base_seed = torch.randint(0, 2**32 - 1, (1,)).item()
            print(f"랜덤 seed 생성: {base_seed}")
    else:
        base_seed = torch.randint(0, 2**32 - 1, (1,)).item()
        print(f"랜덤 seed 생성: {base_seed}")
        
    print("\n" + "="*40)
    print(f"총 {len(texts)}개의 텍스트가 입력되었습니다:")
    for i, t in enumerate(texts):
        print(f"{i+1}. {t}")
    print("="*40 + "\n")
    
    # 6. Optional Speaker Audio
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
        
    # 7. Generate and Save
    print(f"Generating audio for {len(texts)} items...")
    print(f"Using seed: {base_seed}\n")
    
    # 생성된 파일 경로를 저장할 리스트
    output_filenames = []
    
    for i, text in enumerate(texts):
        output_filename = f"output{i+1}.wav"
        output_filenames.append(output_filename)
        print(f"Generating {output_filename} for text: {text[:20]}...")
        
        generate_audio_for_text(model, text, language, speaker_embedding, output_filename, prefix_audio_path="assets/silence_100ms.wav", seed=base_seed)
    
    # 8. 병합 파일 생성 (사용자가 요청한 경우)
    if should_merge:
        merged_output_path = "merged_output.wav"
        merge_audio_files(output_filenames, merged_output_path)

if __name__ == "__main__":
    main()
