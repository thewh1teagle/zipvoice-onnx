
"""
mkdir -p ./model
wget https://huggingface.co/k2-fsa/ZipVoice/resolve/main/zipvoice/text_encoder.onnx -O ./model/text_encoder.onnx
wget https://huggingface.co/k2-fsa/ZipVoice/resolve/main/zipvoice/fm_decoder.onnx -O ./model/fm_decoder.onnx
wget https://huggingface.co/k2-fsa/ZipVoice/resolve/main/zipvoice/text_encoder_int8.onnx -O ./model/text_encoder_int8.onnx
wget https://huggingface.co/k2-fsa/ZipVoice/resolve/main/zipvoice/fm_decoder_int8.onnx -O ./model/fm_decoder_int8.onnx
wget https://huggingface.co/k2-fsa/ZipVoice/resolve/main/zipvoice/model.json -O ./model/model.json
wget https://huggingface.co/k2-fsa/ZipVoice/resolve/main/zipvoice/tokens.txt -O ./model/tokens.txt


mkdir -p ./model_distilled
wget https://huggingface.co/k2-fsa/ZipVoice/resolve/main/zipvoice_distill/text_encoder.onnx -O ./model_distilled/text_encoder.onnx
wget https://huggingface.co/k2-fsa/ZipVoice/resolve/main/zipvoice_distill/fm_decoder.onnx -O ./model_distilled/fm_decoder.onnx
wget https://huggingface.co/k2-fsa/ZipVoice/resolve/main/zipvoice_distill/text_encoder_int8.onnx -O ./model_distilled/text_encoder_int8.onnx
wget https://huggingface.co/k2-fsa/ZipVoice/resolve/main/zipvoice_distill/fm_decoder_int8.onnx -O ./model_distilled/fm_decoder_int8.onnx
wget https://huggingface.co/k2-fsa/ZipVoice/resolve/main/zipvoice_distill/model.json -O ./model_distilled/model.json
wget https://huggingface.co/k2-fsa/ZipVoice/resolve/main/zipvoice_distill/tokens.txt -O ./model_distilled/tokens.txt


wget https://github.com/thewh1teagle/zipvoice-onnx/releases/download/model-files-v1.0/vocos_24khz.onnx
wget https://github.com/thewh1teagle/zipvoice-onnx/releases/download/model-files-v1.0/prompt_english_female2.wav

uv pip install phonemizer-fork espeakng-loader
uv run examples/english_espeak.py
"""

import soundfile as sf
from zipvoice_onnx import ZipVoice, ZipVoiceOptions
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from phonemizer import phonemize
import espeakng_loader

EspeakWrapper.set_library(espeakng_loader.get_library_path())
EspeakWrapper.set_data_path(espeakng_loader.get_data_path())

# Example usage with zipvoice_distill model
options = ZipVoiceOptions(
    text_encoder_path="./model/text_encoder.onnx",
    fm_decoder_path="./model/fm_decoder.onnx",
    text_encoder_int8_path="./model/text_encoder_int8.onnx",
    fm_decoder_int8_path="./model/fm_decoder_int8.onnx",
    model_json_path="./model/model.json",
    tokens_path="./model/tokens.txt",
    vocoder_path="./vocos_24khz.onnx",
)

zipvoice = ZipVoice(options)

# Example usage
ref_wav = "prompt_english_female2.wav"
ref_text = "In order to win, you must expect to win."
target_text = "the most technological aspect of the book."
ref_phonemes = phonemize(text=ref_text, language="en-us", backend="espeak")
target_phonemes = phonemize(text=target_text, language="en-us", backend="espeak")

samples, sample_rate = zipvoice.create(ref_wav, ref_phonemes, target_phonemes)
print(f"Generated audio: {samples.shape} samples at {sample_rate} Hz")

sf.write("audio.wav", samples, sample_rate)
print("Saved to audio.wav")
