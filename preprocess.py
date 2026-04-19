import raw_dataset
from feature_extraction import *
import os
import argparse
import torch
from tqdm import tqdm
from transformers import WavLMModel, Wav2Vec2FeatureExtractor

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Base directory - repo root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description="WavLM-Large feature extraction")
parser.add_argument("--database_dir", type=str,
                    default=os.path.join(BASE_DIR, "asv2019PS", "database"),
                    help="Root directory of the raw ASVspoof2019PS database")
parser.add_argument("--protocol_dir", type=str,
                    default=os.path.join(BASE_DIR, "label"),
                    help="Directory containing protocol/label files")
parser.add_argument("--output_dir", type=str,
                    default=os.path.join(BASE_DIR, "asv2019PS", "preprocess_A1_WavLM_Large"),
                    help="Root directory where extracted .pt features will be saved")
args = parser.parse_args()

DATABASE_DIR = args.database_dir
PROTOCOL_DIR = args.protocol_dir
OUTPUT_DIR   = args.output_dir

def pad_dataset(wav):
    waveform = wav.squeeze(0)
    waveform_len = waveform.shape[0]
    cut = 64600
    if waveform_len >= cut:
        waveform = waveform[:cut]
        return waveform
    # need to pad
    num_repeats = int(cut / waveform_len) + 1
    padded_waveform = torch.tile(waveform, (1, num_repeats))[:, :cut][0]
    return padded_waveform

def normalization(orign_data):
    d_min = orign_data.min()
    if d_min < 0:
        orign_data += torch.abs(d_min)
        d_min = orign_data.min()
    d_max = orign_data.max()
    distance = d_max - d_min
    norm_data = (orign_data - d_min).true_divide(distance)
    return norm_data

for part_ in ["train", "dev", "eval"]:
    asvspoof_raw = raw_dataset.ASVspoof2019PSRaw(DATABASE_DIR,
                                           PROTOCOL_DIR, part=part_)
    target_dir = os.path.join(OUTPUT_DIR, part_,
                              "wavlm-large")
    processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-large")
    model = WavLMModel.from_pretrained("microsoft/wavlm-large", use_safetensors=True).to(device)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for idx in tqdm(range(len(asvspoof_raw))):
        waveform, filename = asvspoof_raw[idx]
        waveform = waveform.to(device)
        print(waveform.shape, 'waveform')
        waveform = waveform.squeeze(dim=0)
        input_values = processor(waveform, sampling_rate=16000,
                                 return_tensors="pt").input_values.to(device)
        with torch.no_grad():
            wavlm_out = model(input_values).last_hidden_state  # (1, T, 1024)
        print(wavlm_out.shape)
        torch.save(wavlm_out, os.path.join(target_dir, "%s.pt" % (filename)))
    print("Done!")