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
                k.replace("xtts.hifigan_decoder.waveform_decoder.", "").replace("hifigan_decoder.waveform_decoder.", ""): v
                for k, v in state_dict["model"].items()
                if "hifigan_decoder" in k and "speaker_encoder" not in k
            }
            self.model.model_g.load_state_dict(hifigan_state_dict, strict=False)

            if config.train_spk_encoder:
                speaker_encoder_state_dict = {
                    k.replace("xtts.hifigan_decoder.speaker_encoder.", "").replace("hifigan_decoder.waveform_decoder.", ""): v
                    for k, v in state_dict["model"].items()
                    if "hifigan_decoder" in k and "speaker_encoder" in k
                }
                self.model.speaker_encoder.load_state_dict(speaker_encoder_state_dict, strict=True)

    def train(self):
        # init the trainer and 🚀
        trainer = Trainer(
            TrainerArgs(), config, config.output_path, model=self.model, train_samples=self.train_samples, eval_samples=self.eval_samples
        )
        trainer.fit()

if __name__ == "__main__":
    with open("config_v00.json", "r") as f:
      config = json.load(f)

    # Dynamically pass the JSON keys to GPTHifiganConfig
    config = GPTHifiganConfig(**config)
    
    # config = GPTHifiganConfig(
    #     batch_size=64,  # Substantially increased for faster training and better gradient stability
    #     eval_batch_size=4,  # Increased evaluation batch size
    #     num_loader_workers=8,  # Higher number of workers to keep up with the larger batch sizes
    #     num_eval_loader_workers=8,  # Proportionally increased for efficient evaluation
    #     run_eval=True,  # Enable evaluation to monitor progress
    #     test_delay_epochs=5,  # Delay evaluation for model stabilization
    #     epochs=1000,  # Increased epochs to allow extended training if metrics are still improving
    #     seq_len=8192,  # Keep sequence length constant
    #     output_sample_rate=24000,  # Maintain desired output audio sample rate
    #     gpt_latent_dim=1024,  # Keep the same latent dimension for GPT
    #     pad_short=2000,  # Padding for short sequences remains unchanged
    #     use_noise_augment=False,  # Enable noise augmentation for better generalization
    #     eval_split_size=150,
    #     print_step=25,  # Log progress every 50 steps
    #     print_eval=True,  # Enable evaluation logging for transparency
    #     mixed_precision=True,  # FP16 for faster training on modern GPUs

    #     # Learning rates for generator and discriminator
    #     lr_gen=1e-4,  # Slightly reduced generator learning rate
    #     lr_disc=1e-4,  # Adjusteprint_step d discriminator learning rate proportionally

    #     # Loss components
    #     use_stft_loss=True,  # STFT loss for frequency alignment
    #     use_l1_spec_loss=True,  # L1 spectral loss for smoother audio
    #     #feat_match_loss_weight=90,  # Balanced weight for perceptual quality
    #     #l1_spec_loss_weight=50,  # Slightly increased for spectral accuracy

    #     # Dataset paths
    #     data_path="lapa_latents/wavs",  # Path to input WAV files
    #     mel_path="lapa_latents/gpt_latents",  # Path to mel spectrograms
    #     spk_path="lapa_latents/speaker_embeddings",  # Path to speaker embeddings
    #     output_path="outputs",  # Directory for saving outputs

    #     # Pretrained model
    #     pretrain_path="XTTS-v2/model.pth",  # Retain pretrained model path

    #     # Speaker encoder training
    #     train_spk_encoder=False,  # Keep disabled as per current requirements
    # )

  

    hifigan_trainer = GPTHifiganTrainer(config=config)
    hifigan_trainer.train()
