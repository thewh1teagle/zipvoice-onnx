"""
wget https://github.com/thewh1teagle/zipvoice-onnx/releases/download/model-files-v1.0/prompt_hebrew_male1.wav -O prompt.wav
uv run examples/hebrew.py
"""

import soundfile as sf
from zipvoice_onnx import ZipVoice, ZipVoiceOptions

# Example usage with zipvoice_distill model
options = ZipVoiceOptions(
    text_encoder_path="./model/text_encoder.onnx",
    fm_decoder_path="./model/fm_decoder.onnx",
    text_encoder_int8_path="./model/text_encoder_int8.onnx",
    fm_decoder_int8_path="./model/fm_decoder_int8.onnx",
    model_json_path="./model/model.json",
    tokens_path="./model/tokens.txt",
)

zipvoice = ZipVoice(options)

# Example usage
ref_wav = "prompt.wav"
ref_phonemes = "halˈaχti lamakˈolet liknˈot lˈeχem veχalˈav, ubadˈeʁeχ paɡˈaʃti χavˈeʁ jaʃˈan ʃelˈo ʁaʔˈiti haʁbˈe zmˈan."
target_phonemes = "halˈaχti lamakˈolet liknˈot lˈeχem veχalˈav, ubadˈeʁeχ paɡˈaʃti χavˈeʁ jaʃˈan ʃelˈo ʁaʔˈiti haʁbˈe zmˈan."

samples, sample_rate = zipvoice.create(ref_wav, ref_phonemes, target_phonemes)
print(f"Generated audio: {samples.shape} samples at {sample_rate} Hz")

sf.write("audio.wav", samples, sample_rate)
print("Saved to audio.wav")
