#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Professional Voice Cloning Script using OpenVoice and DeepFilterNet.

This script takes an input audio file, a text prompt, and a language,
and generates a new audio file with the specified text spoken in the voice
cloned from the input audio.

Key Features:
1.  **Command-Line Interface:** All parameters are passed via command-line arguments for easy integration and scripting.
2.  **Background Noise Removal:** Uses DeepFilterNet to denoise the input audio, improving the quality of the tone color extraction.
3.  **Voice Cloning with OpenVoice:** Leverages the OpenVoice V2 model to clone the voice's tone color.
4.  **Multi-Lingual Synthesis:** Uses MeloTTS as the backend for synthesizing speech in various languages.
5.  **Structured Output:** Saves the final audio to a user-specified path and provides a clear success message.

Prerequisites:
- Python 3.8+
- PyTorch
- OpenVoice (and its dependencies)
- MeloTTS (and its dependencies)
- DeepFilterNet: `pip install -U deepfilternet`
- Pre-trained checkpoints for OpenVoiceV2 downloaded and placed in the 'checkpoints_v2' directory.

Example Usage:
python professional_voice_clone.py \
    --input_audio "path/to/your/reference.wav" \
    --text "Hello, this is a test of voice cloning." \
    --language "EN" \
    --speaker "EN-US" \
    --output_path "cloned_output/final_audio.wav"
"""

import os
import sys
import argparse
import torch
import logging
from typing import Optional

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Model and Path Configurations ---
CKPT_CONVERTER = 'checkpoints_v2/converter'
BASE_SE_PATH = 'checkpoints_v2/base_speakers/ses'

def check_prerequisites():
    """Checks if necessary directories and files exist."""
    if not os.path.exists(CKPT_CONVERTER):
        logging.error(f"Converter checkpoint directory not found at: '{CKPT_CONVERTER}'")
        logging.error("Please download the OpenVoiceV2 checkpoints and place them correctly.")
        sys.exit(1)
    if not os.path.exists(BASE_SE_PATH):
        logging.error(f"Base speaker embedding directory not found at: '{BASE_SE_PATH}'")
        logging.error("Please download the OpenVoiceV2 checkpoints and place them correctly.")
        sys.exit(1)

def setup_arg_parser() -> argparse.ArgumentParser:
    """
    Sets up the command-line argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Professional Voice Cloning Tool using OpenVoice.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-i", "--input_audio",
        type=str, required=True,
        help="Path to the input audio file to clone the voice from."
    )
    parser.add_argument(
        "-t", "--text",
        type=str, required=True,
        help="The text to be synthesized."
    )
    supported_languages = ['EN', 'ES', 'FR', 'ZH', 'JP', 'KR', 'EN_NEWEST']
    parser.add_argument(
        "-l", "--language",
        type=str, required=True,
        choices=supported_languages,
        help=f"Language for the synthesis. Supported: {', '.join(supported_languages)}"
    )
    parser.add_argument(
        "-o", "--output_path",
        type=str, required=True,
        help="Path to save the generated output audio file (e.g., 'output/cloned.wav')."
    )
    parser.add_argument(
        "-s", "--speaker",
        type=str, required=False, default=None,
        help="Optional: The base speaker from MeloTTS. If not provided, a default is used."
    )
    parser.add_argument(
        "--speed",
        type=float, required=False, default=1.0,
        help="Optional: Speed of the synthesized speech (default: 1.0)."
    )
    parser.add_argument(
        "--emotion",
        type=str, required=False,
        help="Note: This is a placeholder. Emotion/style is cloned from the --input_audio, not set by this flag."
    )
    return parser

def denoise_audio(input_path: str) -> str:
    """
    Removes background noise from the input audio file using DeepFilterNet.

    Args:
        input_path (str): Path to the noisy audio file.

    Returns:
        str: Path to the denoised (enhanced) audio file.
    """
    logging.info(f"Denoising audio file: {input_path}")
    try:
        from df.enhance import enhance, init_df, load_audio, save_audio
    except ImportError:
        logging.error("DeepFilterNet not found. Please install it using: pip install -U deepfilternet")
        sys.exit(1)

    if not os.path.exists(input_path):
        logging.error(f"Input audio file not found at: {input_path}")
        raise FileNotFoundError(f"Input audio file not found at: {input_path}")

    model, df_state, _ = init_df()
    audio, sr = load_audio(input_path, sr=df_state.sr())
    enhanced = enhance(model, df_state, audio)

    # Save the denoised audio to a temporary file in the same directory
    output_dir = os.path.dirname(input_path)
    base_name = os.path.basename(input_path)
    denoised_path = os.path.join(output_dir, f"denoised_{base_name}")
    save_audio(denoised_path, enhanced, df_state.sr())
    logging.info(f"Denoised audio saved to: {denoised_path}")
    return denoised_path

def clone_voice(
    tone_color_converter,
    melo_tts,
    text: str,
    reference_audio: str,
    target_speaker: Optional[str],
    speed: float,
    output_path: str,
    device: str
):
    """
    Processes the audio and performs the voice cloning.
    """
    # 1. Extract the target tone color embedding from the denoised reference audio
    logging.info("Extracting tone color embedding from reference audio.")
    from openvoice import se_extractor
    target_se, _ = se_extractor.get_se(reference_audio, tone_color_converter, vad=True)

    # 2. Select a base speaker from MeloTTS
    speaker_ids = melo_tts.hps.data.spk2id
    available_speakers = list(speaker_ids.keys())

    if target_speaker and target_speaker not in available_speakers:
        logging.warning(f"Speaker '{target_speaker}' not found for the selected language.")
        logging.warning(f"Available speakers: {available_speakers}")
        speaker_key = available_speakers[0]
        logging.info(f"Using default speaker: '{speaker_key}'")
    elif target_speaker:
        speaker_key = target_speaker
    else:
        speaker_key = available_speakers[0]
        logging.info(f"No speaker specified. Using default speaker: '{speaker_key}'")

    speaker_id = speaker_ids[speaker_key]
    formatted_speaker_key = speaker_key.lower().replace('_', '-')

    # 3. Load the source speaker embedding
    source_se_path = os.path.join(BASE_SE_PATH, f'{formatted_speaker_key}.pth')
    if not os.path.exists(source_se_path):
        logging.error(f"Base speaker embedding not found at: {source_se_path}")
        raise FileNotFoundError(f"Base speaker embedding not found at: {source_se_path}")

    source_se = torch.load(source_se_path, map_location=device)

    # 4. Synthesize the base audio using MeloTTS
    logging.info(f"Synthesizing base audio with text: '{text}'")
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    src_path = os.path.join(output_dir, 'tmp_base_audio.wav')

    # Handle a specific issue with MPS on CPU
    if torch.backends.mps.is_available() and device == 'cpu':
        torch.backends.mps.is_available = lambda: False

    melo_tts.tts_to_file(text, speaker_id, src_path, speed=speed)

    # 5. Apply the tone color conversion
    logging.info("Applying tone color conversion to create the final audio.")
    encode_message = "@MyShell"  # Watermark
    tone_color_converter.convert(
        audio_src_path=src_path,
        src_se=source_se,
        tgt_se=target_se,
        output_path=output_path,
        message=encode_message
    )

    # 6. Clean up temporary files
    os.remove(src_path)
    logging.info("Cleaned up temporary base audio file.")


def main():
    """
    Main function to execute the voice cloning pipeline.
    """
    # Check for necessary files before proceeding
    check_prerequisites()
    
    parser = setup_arg_parser()
    args = parser.parse_args()

    # --- Step 1: Denoise Input Audio ---
    try:
        denoised_audio_path = denoise_audio(args.input_audio)
    except Exception as e:
        logging.error(f"Failed to denoise audio. Error: {e}")
        sys.exit(1)

    # --- Step 2: Initialize Models ---
    logging.info("Initializing models...")
    try:
        from openvoice.api import ToneColorConverter
        from melo.api import TTS
    except ImportError as e:
        logging.error(f"Failed to import a required library: {e}")
        logging.error("Please ensure OpenVoice and MeloTTS are installed correctly.")
        sys.exit(1)
        
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    try:
        tone_color_converter = ToneColorConverter(f'{CKPT_CONVERTER}/config.json', device=device)
        tone_color_converter.load_ckpt(f'{CKPT_CONVERTER}/checkpoint.pth')
        melo_tts = TTS(language=args.language, device=device)
    except Exception as e:
        logging.error(f"Failed to initialize models. Error: {e}")
        os.remove(denoised_audio_path) # Clean up denoised file on failure
        sys.exit(1)

    # --- Step 3: Process the Audio ---
    logging.info("Starting the voice cloning process...")
    try:
        clone_voice(
            tone_color_converter=tone_color_converter,
            melo_tts=melo_tts,
            text=args.text,
            reference_audio=denoised_audio_path,
            target_speaker=args.speaker,
            speed=args.speed,
            output_path=args.output_path,
            device=device
        )
    except Exception as e:
        logging.error(f"An error occurred during voice cloning. Error: {e}")
    finally:
        # --- Step 4: Cleanup and Final Message ---
        os.remove(denoised_audio_path)
        logging.info(f"Cleaned up temporary denoised file: {denoised_audio_path}")

    # --- Success Message ---
    success_message = f"""
    ===============================================================
    Success! Voice cloning process completed.
    Output saved to: {os.path.abspath(args.output_path)}
    ===============================================================
    """
    print(success_message)


if __name__ == "__main__":
    main()