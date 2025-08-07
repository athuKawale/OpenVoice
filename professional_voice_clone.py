#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Professional Voice Cloning Script using OpenVoice and DeepFilterNet.

This script takes an input audio file, a text prompt, and a language,
and generates a new audio file with the specified text spoken in the voice
cloned from the input audio.
"""

import os
import sys
import argparse
import torch
import logging
from typing import Optional, Tuple

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Model and Path Configurations ---
CKPT_CONVERTER = 'checkpoints_v2/converter'
BASE_SE_PATH = 'checkpoints_v2/base_speakers/ses'
SUPPORTED_LANGUAGES = ['EN', 'ES', 'FR', 'ZH', 'JP', 'KR']

# --- Lazy Import Dependencies ---
try:
    from df.enhance import enhance, init_df, load_audio, save_audio
    from openvoice.api import ToneColorConverter
    from melo.api import TTS
    from openvoice import se_extractor
except ImportError as e:
    logging.error(f"Failed to import a required library: {e}")
    logging.error("Please ensure all dependencies are installed correctly.")
    logging.error("You may need to run: pip install -U deepfilternet openvoice-tts melo-tts")
    sys.exit(1)


def check_prerequisites():
    """Checks if necessary checkpoint directories exist."""
    if not os.path.exists(CKPT_CONVERTER):
        logging.error(f"Converter checkpoint directory not found: '{CKPT_CONVERTER}'")
        logging.error("Please download the OpenVoiceV2 checkpoints and place them correctly.")
        sys.exit(1)
    if not os.path.exists(BASE_SE_PATH):
        logging.error(f"Base speaker embedding directory not found: '{BASE_SE_PATH}'")
        logging.error("Please download the OpenVoiceV2 checkpoints and place them correctly.")
        sys.exit(1)


def setup_arg_parser() -> argparse.ArgumentParser:
    """Sets up and returns the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="A professional voice cloning tool using OpenVoice.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-i", "--input_audio", type=str, required=True,
        help="Path to the reference audio file for voice cloning."
    )
    parser.add_argument(
        "-t", "--text", type=str, required=True,
        help="The text to be synthesized."
    )
    parser.add_argument(
        "-l", "--language", type=str, required=True, choices=SUPPORTED_LANGUAGES,
        help=f"Language for synthesis. Supported: {', '.join(SUPPORTED_LANGUAGES)}"
    )
    parser.add_argument(
        "-o", "--output_path", type=str, required=True,
        help="Path to save the generated audio file (e.g., 'output/cloned.wav')."
    )
    parser.add_argument(
        "-s", "--speaker", type=str, required=False, default=None,
        help="Optional: The base speaker from MeloTTS. Defaults to the first available speaker for the language."
    )
    parser.add_argument(
        "--speed", type=float, required=False, default=1.0,
        help="Optional: Speed of the synthesized speech (default: 1.0)."
    )
    return parser


def denoise_audio(input_path: str) -> str:
    """
    Removes background noise from an audio file using DeepFilterNet.

    Args:
        input_path: Path to the noisy audio file.

    Returns:
        Path to the denoised (enhanced) audio file.
    """
    logging.info(f"Denoising audio file: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input audio file not found at: {input_path}")

    model, df_state, _ = init_df()
    audio, _ = load_audio(input_path, sr=df_state.sr())
    enhanced = enhance(model, df_state, audio)

    output_dir = os.path.dirname(input_path) or '.'
    base_name = os.path.basename(input_path)
    denoised_path = os.path.join(output_dir, f"denoised_{base_name}")
    
    save_audio(denoised_path, enhanced, df_state.sr())
    logging.info(f"Denoised audio saved to: {denoised_path}")
    return denoised_path


def select_speaker(melo_tts: TTS, target_speaker: Optional[str]) -> Tuple[str, int]:
    """Selects the speaker, defaulting to the first available if the target is not found."""
    speaker_ids = melo_tts.hps.data.spk2id
    available_speakers = list(speaker_ids.keys())

    if target_speaker and target_speaker in available_speakers:
        speaker_key = target_speaker
        logging.info(f"Using specified speaker: '{speaker_key}'")
    else:
        if target_speaker:
            logging.warning(f"Speaker '{target_speaker}' not found for the selected language.")
            logging.warning(f"Available speakers: {available_speakers}")
        speaker_key = available_speakers[0]
        logging.info(f"Using default speaker: '{speaker_key}'")
        
    return speaker_key, speaker_ids[speaker_key]


def clone_voice(
    tone_color_converter: ToneColorConverter,
    melo_tts: TTS,
    text: str,
    reference_audio: str,
    target_speaker: Optional[str],
    speed: float,
    output_path: str,
    device: str
):
    """Processes the audio and performs voice cloning."""
    # 1. Extract tone color embedding from the reference audio.
    logging.info("Extracting tone color embedding from reference audio.")
    target_se, _ = se_extractor.get_se(reference_audio, tone_color_converter, vad=True)

    # 2. Select a base speaker.
    speaker_key, speaker_id = select_speaker(melo_tts, target_speaker)
    formatted_speaker_key = speaker_key.lower().replace('_', '-')

    # 3. Load the source speaker embedding.
    source_se_path = os.path.join(BASE_SE_PATH, f'{formatted_speaker_key}.pth')
    if not os.path.exists(source_se_path):
        raise FileNotFoundError(f"Base speaker embedding not found at: {source_se_path}")
    source_se = torch.load(source_se_path, map_location=device)

    # 4. Synthesize the base audio.
    logging.info(f"Synthesizing base audio with text: '{text}'")
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    src_path = os.path.join(output_dir or '.', 'tmp_base_audio.wav')
    
    melo_tts.tts_to_file(text, speaker_id, src_path, speed=speed)

    # 5. Apply tone color conversion.
    logging.info("Applying tone color conversion to create the final audio.")
    tone_color_converter.convert(
        audio_src_path=src_path,
        src_se=source_se,
        tgt_se=target_se,
        output_path=output_path,
        message="@MyShell"  # Watermark
    )

    # 6. Clean up temporary base audio file.
    os.remove(src_path)
    logging.info("Cleaned up temporary base audio file.")


def main():
    """Main function to execute the voice cloning pipeline."""
    check_prerequisites()
    
    parser = setup_arg_parser()
    args = parser.parse_args()

    denoised_audio_path = None
    try:
        # Step 1: Denoise Input Audio
        denoised_audio_path = denoise_audio(args.input_audio)

        # Step 2: Initialize Models
        logging.info("Initializing models...")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # Handle a specific issue with MPS on CPU
        if torch.backends.mps.is_available() and device == 'cpu':
            torch.backends.mps.is_available = lambda: False

        converter = ToneColorConverter(f'{CKPT_CONVERTER}/config.json', device=device)
        converter.load_ckpt(f'{CKPT_CONVERTER}/checkpoint.pth')
        tts_model = TTS(language=args.language, device=device)

        # Step 3: Process and Clone Voice
        logging.info("Starting the voice cloning process...")
        clone_voice(
            tone_color_converter=converter,
            melo_tts=tts_model,
            text=args.text,
            reference_audio=denoised_audio_path,
            target_speaker=args.speaker,
            speed=args.speed,
            output_path=args.output_path,
            device=device
        )
        
        # --- Success Message ---
        success_message = f"""
        ===============================================================
        Success! Voice cloning process completed.
        Output saved to: {os.path.abspath(args.output_path)}
        ===============================================================
        """
        print(success_message)

    except Exception as e:
        logging.error(f"An error occurred during the process: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Step 4: Cleanup
        if denoised_audio_path and os.path.exists(denoised_audio_path):
            os.remove(denoised_audio_path)
            logging.info(f"Cleaned up temporary denoised file: {denoised_audio_path}")


if __name__ == "__main__":
    main()
