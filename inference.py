import glob
import os
import argparse
import json
import torch
import librosa
from models.stfts import mag_phase_stft, mag_phase_istft
from models.generator import MambAttention
import numpy as np
import soundfile as sf
import time

from utils.util import (
    load_ckpts, load_optimizer_states, save_checkpoint,
    build_env, load_config, initialize_seed,
    print_gpu_info, log_model_info, initialize_process_group,
)

h = None
device = None


def process_chunk(chunk, model, n_fft, hop_size, win_size, compress_factor, device):
    """Run a single audio chunk through the model and return the enhanced audio."""
    chunk_tensor = torch.FloatTensor(chunk).to(device)
    norm_factor = torch.sqrt(len(chunk_tensor) / torch.sum(chunk_tensor ** 2.0)).to(device)
    chunk_tensor = (chunk_tensor * norm_factor).unsqueeze(0)

    noisy_amp, noisy_pha, noisy_com = mag_phase_stft(chunk_tensor, n_fft, hop_size, win_size, compress_factor)
    amp_g, pha_g, com_g = model(noisy_amp, noisy_pha)
    audio_g = mag_phase_istft(amp_g, pha_g, n_fft, hop_size, win_size, compress_factor)
    audio_g = audio_g / norm_factor

    return audio_g.squeeze().cpu().numpy()


def inference(args, device):
    cfg = load_config(args.config)
    n_fft, hop_size, win_size = cfg['stft_cfg']['n_fft'], cfg['stft_cfg']['hop_size'], cfg['stft_cfg']['win_size']
    compress_factor = cfg['model_cfg']['compress_factor']
    sampling_rate = cfg['stft_cfg']['sampling_rate']

    # 2 seconds at the configured sampling rate
    chunk_seconds = 2.0
    chunk_len = int(chunk_seconds * sampling_rate)

    model = MambAttention(cfg).to(device)
    state_dict = torch.load(args.checkpoint_file, map_location=device)
    model.load_state_dict(state_dict['generator'])

    os.makedirs(args.output_folder, exist_ok=True)

    model.eval()

    with torch.no_grad():
        # You can use data.json instead of input_folder with:
        # ---------------------------------------------------- #
        # with open("data/test_noisy.json", 'r') as json_file:
        #     test_files = json.load(json_file)
        # for i, fname in enumerate( test_files ):
        #     folder_path = os.path.dirname(fname)
        #     fname = os.path.basename(fname)
        #     noisy_wav, _ = librosa.load(os.path.join( folder_path, fname ), sr=sampling_rate)
        #     noisy_wav = torch.FloatTensor(noisy_wav).to(device)
        # ---------------------------------------------------- #
        for i, fname in enumerate(sorted(f for f in os.listdir(args.input_folder) if f.endswith('.wav'))):
            noisy_wav, _ = librosa.load(os.path.join(args.input_folder, fname), sr=sampling_rate)

            total_len = len(noisy_wav)
            processed_chunks = []

            # Split into chunk_len-sized pieces, process each, then stitch back together.
            # The final chunk may be shorter than chunk_len — it is zero-padded before
            # being passed to the model and the padding is trimmed from the output.
            for start in range(0, total_len, chunk_len):
                end = start + chunk_len
                chunk = noisy_wav[start:end]

                # Pad the last (possibly short) chunk so the model always gets a
                # full-length input, then remember how much padding we added so we
                # can strip it from the output.
                pad_len = chunk_len - len(chunk)
                if pad_len > 0:
                    chunk = np.pad(chunk, (0, pad_len), mode='constant')

                enhanced = process_chunk(chunk, model, n_fft, hop_size, win_size, compress_factor, device)

                # Remove padding from the output to match the original chunk length
                if pad_len > 0:
                    enhanced = enhanced[:len(enhanced) - pad_len]

                processed_chunks.append(enhanced)

            # Concatenate all chunks back into a single waveform
            audio_out = np.concatenate(processed_chunks, axis=0)

            output_file = os.path.join(args.output_folder, fname)
            sf.write(output_file, audio_out, sampling_rate, 'PCM_16')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', default='VB-DemandEx/noisy_test')
    parser.add_argument('--output_folder', default='results')
    parser.add_argument('--config', default='MambAttention/checkpoints/MambAttention_seed3441_VB-DemandEx.yaml')
    parser.add_argument('--checkpoint_file', default='MambAttention/checkpoints/seed3441.yaml', required=True)
    args = parser.parse_args()

    global device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        # device = torch.device('cpu')
        raise RuntimeError("Currently, CPU mode is not supported.")

    start = time.time()
    inference(args, device)
    print(time.time() - start)


if __name__ == '__main__':
    print("Initializing inference...")
    main()
