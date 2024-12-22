import os
import random
import json

import torch
from trainer import Trainer, TrainerArgs
from TTS.utils.audio import AudioProcessor

from datasets.preprocess import load_wav_feat_spk_data
from configs.gpt_hifigan_config import GPTHifiganConfig
from models.gpt_gan import GPTGAN


class GPTHifiganTrainer:
    def __init__(self, config):
        self.config = config
        self.ap = AudioProcessor(**config.audio.to_dict())

        print(f"Files in data path: {os.listdir(config.data_path)}")
        print(f"Files in mel path: {os.listdir(config.mel_path)}")
        print(f"Files in speaker path: {os.listdir(config.spk_path)}")

        self.eval_samples, self.train_samples = load_wav_feat_spk_data(
            config.data_path, config.mel_path, config.spk_path, eval_split_size=config.eval_split_size
        )

        if len(self.train_samples) == 0:
            print("[!] Training set is empty. Using all samples for training.")
            self.train_samples = self.eval_samples
            self.eval_samples = []

        # Log dataset split
        print(f"Training samples: {len(self.train_samples)}")
        print(f"Evaluation samples: {len(self.eval_samples)}")

        # Initialize model
        self.model = GPTGAN(config, self.ap)

        # Load pretrained weights if provided
        if config.pretrain_path is not None:
            state_dict = torch.load(config.pretrain_path)
            hifigan_state_dict = {
                k.replace("xtts.hifigan_decoder.waveform_decoder.", "").replace("hifigan_decoder.waveform_decoder.",
                                                                                ""): v
                for k, v in state_dict["model"].items()
                if "hifigan_decoder" in k and "speaker_encoder" not in k
            }
            self.model.model_g.load_state_dict(hifigan_state_dict, strict=False)

            if config.train_spk_encoder:
                speaker_encoder_state_dict = {
                    k.replace("xtts.hifigan_decoder.speaker_encoder.", "").replace("hifigan_decoder.waveform_decoder.",
                                                                                   ""): v
                    for k, v in state_dict["model"].items()
                    if "hifigan_decoder" in k and "speaker_encoder" in k
                }
                self.model.speaker_encoder.load_state_dict(speaker_encoder_state_dict, strict=True)

    def train(self):
        # init the trainer and 🚀
        trainer = Trainer(
            TrainerArgs(), config, config.output_path, model=self.model, train_samples=self.train_samples,
            eval_samples=self.eval_samples
        )
        trainer.fit()


if __name__ == "__main__":
    with open("config_v00.json", "r") as f:
        config = json.load(f)

    # Dynamically pass the JSON keys to GPTHifiganConfig
    config = GPTHifiganConfig(**config)

    hifigan_trainer = GPTHifiganTrainer(config=config)
    hifigan_trainer.train()
