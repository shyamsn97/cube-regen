from regen.dataset import DynamicDamageDataset
from regen.model import NCA3DDamageDetection
from regen.trainer import NCA3DTrainer
import modal
import os
import numpy as np

app = modal.App("nca-3d-trainer")
reqs =  ["torch", "tensorboard", "matplotlib", "wandb", "Pillow", "tqdm", "huggingface_hub"]
image = modal.Image.debian_slim().pip_install(reqs).add_local_python_source("regen")
env_variables = {
    "HF_TOKEN": os.environ.get("HF_TOKEN", None),
    "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", None)
}

@app.function(
    gpu="A10G",
    image=image,
    secrets=[modal.Secret.from_dict(env_variables)],
    timeout=60*60*21
)
def train(shapes, labels):
    model = NCA3DDamageDetection(use_tanh=False, clip_range=64.0, num_hidden_channels=128)
    dataset = DynamicDamageDataset(shapes, labels, damage_radius_range=(1, 3), damage_types=["sphere", "cube", "random"], random_proportion_range=(0.1, 0.2), fixed_damage=False, augment_rotations=False, return_damage_mask=True, seed=None, filter_label=3)
    trainer = NCA3DTrainer(model, dataset, batch_size=8, lr=2e-5, iterations_per_epoch=100, steps_per_sample=96, buffer_size=1000, buffer_sampling_prob=0.5, repo_id="shyamsn97/cube-big")
    trainer.train(epochs=500, save_frequency=10, visualization_frequency=10)

@app.local_entrypoint()
def main():
    labels = np.load('../data/ydata_7class.npy')
    shapes = np.load('../data/xdata_7class.npy')
    train.remote(shapes, labels)
