# Multi Scripts 사용 가이드

이 문서는 여러 텍스트를 한 번에 음성으로 변환하는 스크립트들의 사용법을 설명합니다.

## 📋 스크립트 종류

### 1. `multi.py` - 일관된 음성 생성
**용도**: 동일한 화자 특성으로 여러 음성 파일 생성

**특징**:
- ✅ 모든 텍스트에 **동일한 시드** 사용
- ✅ 일관된 음성 특성 (톤, 억양, 속도)
- ✅ 영상 내레이션처럼 **통일감이 필요한 경우** 적합

**사용 예시**:
```bash
uv run multi.py
```

### 2. `multi_random_seed.py` - 다양한 음성 생성
**용도**: 각 텍스트마다 다른 음성 특성으로 생성

**특징**:
- ✅ 각 텍스트마다 **랜덤 시드** 생성
- ✅ 다양한 음성 변화 (톤, 억양, 속도)
- ✅ 대화형 콘텐츠나 **다양성이 필요한 경우** 적합

**사용 예시**:
```bash
uv run multi_random_seed.py
```

## 🎯 사용 방법

### 1. 스크립트 실행
```bash
uv run multi.py
# 또는
uv run multi_random_seed.py
```

### 2. 텍스트 입력
```
변환할 텍스트를 입력해주세요 (한 줄에 하나씩). 입력을 마치려면 빈 줄에서 엔터를 누르세요:
안녕하세요.
오늘 날씨가 좋네요.
감사합니다.
[빈 줄에서 엔터]
```

### 3. 언어 코드 입력
```
언어 코드를 입력하세요 (기본값: ko): 
[엔터 또는 ko/en/ja 등 입력]
```

### 4. 생성 완료
```
Saved to output1.wav (normalized to -20dB RMS)
Saved to output2.wav (normalized to -20dB RMS)
Saved to output3.wav (normalized to -20dB RMS)
```

## 🔊 음성 클로닝

두 스크립트 모두 `cloneAudio/clone.wav` 파일이 있으면 자동으로 음성 클로닝을 수행합니다.

**설정 방법**:
1. 클로닝할 음성 파일을 `cloneAudio/clone.wav`에 저장
2. 스크립트 실행 시 자동으로 감지 및 적용

## 📊 오디오 정규화

두 스크립트 모두 **RMS 정규화**를 자동으로 적용합니다.

**정규화 특징**:
- 🎚️ 모든 파일이 **-20dB RMS**로 통일
- 🎵 일관된 음량 레벨
- 🚫 클리핑 방지 (Soft Limiting)
- ✅ 영상 편집 시 별도 볼륨 조정 불필요

## 🆚 어떤 스크립트를 사용해야 할까?

| 상황 | 추천 스크립트 | 이유 |
|------|--------------|------|
| 영상 내레이션 | `multi.py` | 일관된 화자 느낌 |
| 튜토리얼/강의 | `multi.py` | 전문적이고 통일된 음성 |
| 대화형 콘텐츠 | `multi_random_seed.py` | 자연스러운 변화 |
| 샘플 생성 | `multi_random_seed.py` | 다양한 결과 확인 |
| 오디오북 | `multi.py` | 일관된 독서 경험 |

## 🛠️ 고급 설정

### 생성 파라미터 (코드 수정 필요)

두 스크립트의 `generate_audio_for_text()` 함수에서 다음 파라미터를 조정할 수 있습니다:

```python
# 감정 설정 (Happy, Sad, Disgust, Fear, Surprise, Anger, Other, Neutral)
emotion_tensor = torch.tensor([1.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.2], device=device)

# 음성 품질
vq_val = 0.78  # 0.5 ~ 0.8 (높을수록 고품질)
fmax = 24000.0  # 최대 주파수
pitch_std = 45.0  # 피치 변화 (0 ~ 300)
speaking_rate = 15.0  # 말하기 속도 (5 ~ 30)
dnsmos_ovrl = 4.0  # 전체 품질 (1 ~ 5)

# 생성 설정
cfg_scale = 2.0  # CFG 스케일 (1 ~ 5)
linear = 0.5  # 샘플링 선형성
confidence = 0.40  # 샘플링 신뢰도
```

### 정규화 레벨 변경

기본값 `-20dB`를 변경하려면:

```python
# 더 큰 소리: -18dB
wav_out_normalized = normalize_audio_rms(wav_out.squeeze(), target_rms_db=-18.0)

# 더 작은 소리: -23dB
wav_out_normalized = normalize_audio_rms(wav_out.squeeze(), target_rms_db=-23.0)
```

## 📝 출력 파일

생성된 파일은 현재 디렉토리에 저장됩니다:
- `output1.wav`
- `output2.wav`
- `output3.wav`
- ...

## ⚠️ 주의사항

1. **메모리**: 긴 텍스트나 많은 파일 생성 시 GPU 메모리 부족 가능
2. **시간**: 각 텍스트당 약 10~30초 소요
3. **덮어쓰기**: 기존 `output*.wav` 파일은 자동으로 덮어씌워짐

## 🔍 트러블슈팅

### 음량이 여전히 불일치
- 정규화 로직이 최신 버전인지 확인
- `-20dB` 값을 `-18dB`로 조정 시도

### 음성 품질이 낮음
- `vq_val`을 `0.78`로 설정 확인
- `dnsmos_ovrl`을 `4.0` 이상으로 설정

### GPU 메모리 부족
- 텍스트를 더 작은 배치로 나누어 실행
- 한 번에 5~10개 텍스트만 처리

## 📚 관련 파일

- [multi.py](file:///Users/kojaemin/Documents/Zunos-TTS/multi.py) - 일관된 시드 버전
- [multi_random_seed.py](file:///Users/kojaemin/Documents/Zunos-TTS/multi_random_seed.py) - 랜덤 시드 버전
- [gradio_interface.py](file:///Users/kojaemin/Documents/Zunos-TTS/gradio_interface.py) - 웹 UI 버전
