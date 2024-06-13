import sys
import os
import random

import librosa
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoFeatureExtractor


def load_custom_dataset(dataset_name):
    return getattr(sys.modules[__name__], dataset_name)


class VoiceDatasetWav2Vec(Dataset):
    def __init__(self, config, mode):
        self.data_path = config['data_path'][mode]
        self.max_length = config['max_input_length']
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(config['model_name'])
        self.data = []            # 전체 데이터
        self.speaker_files = []   # 총 화자
        self.speaker_datas = {}   # 각 화자에 대한 데이터
        
        with open(self.data_path, 'r') as f:
            for line in f:
                self.speaker_files.append(line.strip())
                
        for txt_file in self.speaker_files:
            speaker_id = os.path.basename(txt_file).split('.')[0]
            with open(txt_file, 'r') as f:
                self.speaker_datas[speaker_id] = [line.strip() for line in f]
        
        for files in self.speaker_datas.values():
            self.data.extend(files)


    def load_audio(self, file_path):
        audio, sampling_rate = librosa.load(file_path, sr=16000)
        inputs = self.feature_extractor(audio, sampling_rate=sampling_rate, max_length=int(self.max_length * sampling_rate), truncation=True, padding="max_length", return_tensors="pt")
        return inputs.input_values.squeeze(0)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        anchor_data = self.data[idx]
        anchor_audio = self.load_audio(anchor_data)
        
        # speaker_id 추출
        speaker_id = os.path.basename(os.path.dirname(anchor_data))
        each_speaker_datas = self.speaker_datas[speaker_id]
        
        # Positive pair는 같은 화자의 파일들 중 랜덤으로 선택
        positive_data = random.choice([x for x in each_speaker_datas if x != anchor_data])
        positive_audio = self.load_audio(positive_data)
        
        # Negative pair는 다른 화자의 파일들 중 랜덤으로 선택
        negative_speaker_ids = [other_speaker_id for other_speaker_id in self.speaker_datas.keys() if other_speaker_id != speaker_id]
        negative_speaker_id = random.choice(negative_speaker_ids)
        negative_data = random.choice(self.speaker_datas[negative_speaker_id])
        negative_audio = self.load_audio(negative_data)

        return anchor_audio, positive_audio, negative_audio

class VoiceDatasetWhisper(Dataset):
    def __init__(self, config, mode):
        self.data_path = config['data_path'][mode]
        self.max_length = config['max_input_length']
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(config['model_name'])
        self.data = []            # 전체 데이터
        self.speaker_files = []   # 총 화자
        self.speaker_datas = {}   # 각 화자에 대한 데이터
        
        with open(self.data_path, 'r') as f:
            for line in f:
                self.speaker_files.append(line.strip())
                
        for txt_file in self.speaker_files:
            speaker_id = os.path.basename(txt_file).split('.')[0]
            with open(txt_file, 'r') as f:
                self.speaker_datas[speaker_id] = [line.strip() for line in f]
        
        for files in self.speaker_datas.values():
            self.data.extend(files)

    def load_audio(self, file_path):
        audio, sampling_rate = librosa.load(file_path, sr=16000)
        audio_length_seconds = len(audio) / sampling_rate

        inputs = self.feature_extractor(audio, sampling_rate=sampling_rate, padding="max_length", return_tensors="pt")
        return inputs.input_features.squeeze(0)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        anchor_data = self.data[idx]
        anchor_audio = self.load_audio(anchor_data)
        
        # speaker_id 추출
        speaker_id = os.path.basename(os.path.dirname(anchor_data))
        each_speaker_datas = self.speaker_datas[speaker_id]
        
        # Positive pair는 같은 화자의 파일들 중 랜덤으로 선택
        positive_data = random.choice([x for x in each_speaker_datas if x != anchor_data])
        positive_audio = self.load_audio(positive_data)
        
        # Negative pair는 다른 화자의 파일들 중 랜덤으로 선택
        negative_speaker_ids = [other_speaker_id for other_speaker_id in self.speaker_datas.keys() if other_speaker_id != speaker_id]
        negative_speaker_id = random.choice(negative_speaker_ids)
        negative_data = random.choice(self.speaker_datas[negative_speaker_id])
        negative_audio = self.load_audio(negative_data)

        return anchor_audio, positive_audio, negative_audio

