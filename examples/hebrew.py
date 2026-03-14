"""
wget https://huggingface.co/thewh1teagle/zipvoice-heb/resolve/main/zipvoice-onnx.tar.gz
wget https://github.com/thewh1teagle/zipvoice-onnx/releases/download/model-files-v1.0/prompt_hebrew_male1.wav -O prompt.wav
wget https://github.com/thewh1teagle/zipvoice-onnx/releases/download/model-files-v1.0/vocos_24khz.onnx
mkdir -p model-he
tar -xf zipvoice-onnx.tar.gz -C ./model-he --strip-components=1
uv run examples/hebrew.py
"""

import soundfile as sf
from zipvoice_onnx import ZipVoice, ZipVoiceOptions

# Example usage with zipvoice_distill model
options = ZipVoiceOptions(
    text_encoder_path="./model-he/text_encoder.onnx",
    fm_decoder_path="./model-he/fm_decoder.onnx",
    text_encoder_int8_path="./model-he/text_encoder_int8.onnx",
    fm_decoder_int8_path="./model-he/fm_decoder_int8.onnx",
    model_json_path="./model-he/model.json",
    tokens_path="./model-he/tokens.txt",
    vocoder_path="./vocos_24khz.onnx",
)

zipvoice = ZipVoice(options)

# Example usage
ref_wav = "prompt.wav"
ref_phonemes = "halˈaχti lamakˈolet liknˈot lˈeχem veχalˈav, ubadˈeʁeχ paɡˈaʃti χavˈeʁ jaʃˈan ʃelˈo ʁaʔˈiti haʁbˈe zmˈan."
target_phonemes = "halˈaχti lamakˈolet liknˈot lˈeχem veχalˈav, ubadˈeʁeχ paɡˈaʃti χavˈeʁ jaʃˈan ʃelˈo ʁaʔˈiti haʁbˈe zmˈan."

samples, sample_rate = zipvoice.create(ref_wav, ref_phonemes, target_phonemes, num_steps=8)
print(f"Generated audio: {samples.shape} samples at {sample_rate} Hz")

sf.write("audio.wav", samples, sample_rate)
print("Saved to audio.wav")
