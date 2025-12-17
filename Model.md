# 모델 다운로드 가이드 (`Model.md`)

이 프로젝트에서 사용하는 모든 모델에 대한 주소와 다운로드 방법입니다.
로컬 환경(오프라인)에서 실행하려면 아래 파일들을 `models/` 폴더 내에 정해진 구조로 다운로드 받아야 합니다.

## 1. Zonos Main Models (TTS 핵심 모델)
둘 중 하나(또는 둘 다) 다운로드 하시면 됩니다. 보통 `transformer` 모델이 기본값입니다.

### **Zyphra/Zonos-v0.1-transformer**
- **Hugging Face 주소**: [https://huggingface.co/Zyphra/Zonos-v0.1-transformer](https://huggingface.co/Zyphra/Zonos-v0.1-transformer)
- **저장 위치**: `models/Zyphra/Zonos-v0.1-transformer/`
- **필수 파일**:
    - `config.json`
    - `model.safetensors`

### **Zyphra/Zonos-v0.1-hybrid** (선택 사항)
- **설명**: `mamba-ssm` 라이브러리가 필요하며, 고급 사용자용입니다. **일반적으로는 위 Transformer 모델만 쓰셔도 충분합니다.**
- **주의**: 이 모델은 **NVIDIA GPU(CUDA)** 가 필요합니다. **macOS에서는 설치가 어렵거나 불가능할 수 있으므로 Transformer 모델 사용을 강력 권장합니다.**
- **설치 방법** (Linux/Windows + NVIDIA GPU):
    ```bash
    uv pip install "mamba-ssm>=2.2.4" "causal-conv1d>=1.5.0.post8"
    ```
- **Hugging Face 주소**: [https://huggingface.co/Zyphra/Zonos-v0.1-hybrid](https://huggingface.co/Zyphra/Zonos-v0.1-hybrid)
- **저장 위치**: `models/Zyphra/Zonos-v0.1-hybrid/`
- **필수 파일**:
    - `config.json`
    - `model.safetensors`

---

## 2. Speaker Embedding Models (목소리 복제용)
목소리 복제 기능을 위해 필요합니다.

### **Zyphra/Zonos-v0.1-speaker-embedding**
- **Hugging Face 주소**: [https://huggingface.co/Zyphra/Zonos-v0.1-speaker-embedding](https://huggingface.co/Zyphra/Zonos-v0.1-speaker-embedding)
- **저장 위치**: `models/Zyphra/Zonos-v0.1-speaker-embedding/`
- **필수 파일**:
    - `ResNet293_SimAM_ASP_base.pt`
    - `ResNet293_SimAM_ASP_base_LDA-128.pt`

---

## 3. DAC Autoencoder (오디오 디코더)
생성된 코드를 오디오 파형으로 변환하는 데 사용됩니다.

### **descript/dac_44khz**
- **Hugging Face 주소**: [https://huggingface.co/descript/dac_44khz](https://huggingface.co/descript/dac_44khz)
- **저장 위치**: `models/descript/dac_44khz/`
- **필수 파일** (가능하면 전부 받는 것을 권장하지만, 최소한 아래 파일들은 필요합니다):
    - `config.json`
    - `model.safetensors` (또는 `pytorch_model.bin`)
    - `preprocessor_config.json` (있다면)

---

## 디렉토리 구조 예시
`models` 폴더를 만들고 아래와 같이 파일을 배치하면 됩니다.

```
models/
├── Zyphra/
│   ├── Zonos-v0.1-transformer/
│   │   ├── config.json
│   │   └── model.safetensors
│   ├── Zonos-v0.1-hybrid/
│   │   ├── config.json
│   │   └── model.safetensors
│   └── Zonos-v0.1-speaker-embedding/
│       ├── ResNet293_SimAM_ASP_base.pt
│       └── ResNet293_SimAM_ASP_base_LDA-128.pt
└── descript/
    └── dac_44khz/
        ├── config.json
        ├── model.safetensors
        └── ...
```
