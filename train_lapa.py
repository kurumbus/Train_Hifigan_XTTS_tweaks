import os
import json
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from TTS.utils.audio import AudioProcessor
from notdatasets.preprocess import load_wav_feat_spk_data
from configs.gpt_hifigan_config import GPTHifiganConfig
from models.gpt_gan import GPTGAN
from trainer import Trainer, TrainerArgs

class GPTHifiganTrainer:
    def __init__(self, config, rank, world_size):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.ap = AudioProcessor(**config.audio.to_dict())

        print(f"Files in data path: {os.listdir(config.data_path)}")
        print(f"Files in mel path: {os.listdir(config.mel_path)}")
        print(f"Files in speaker path: {os.listdir(config.spk_path)}")

        # Load data
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

        # Move model to GPU and wrap it in DDP
        self.model = self.model.to(self.rank)  # Move model to the current GPU
        self.model = DDP(self.model, device_ids=[self.rank])

        # Load pretrained weights if provided
        if config.pretrain_path is not None:
            state_dict = torch.load(config.pretrain_path, map_location='cuda')
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
        # Set up Distributed Sampler
        train_sampler = DistributedSampler(self.train_samples, num_replicas=self.world_size, rank=self.rank)

        # Initialize the trainer
        trainer = Trainer(
            TrainerArgs(), self.config, self.config.output_path, model=self.model,
            train_samples=self.train_samples, eval_samples=self.eval_samples, sampler=train_sampler
        )
        trainer.fit()

def main_worker(rank, world_size, config):
    # Set environment variables for distributed training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)

    # Initialize the distributed process group
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    # Initialize and train the model
    trainer = GPTHifiganTrainer(config, rank, world_size)
    trainer.train()

    # Cleanup after training
    dist.destroy_process_group()

def main():
    world_size = torch.cuda.device_count()  # Number of GPUs
    config = GPTHifiganConfig(**json.load(open("config_v00.json", "r")))  # Load config file

    # Launch the distributed training processes
    mp.spawn(main_worker, args=(world_size, config), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
