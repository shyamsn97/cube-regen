from regen.model import load_weights_from_huggingface
import numpy as np
from regen.dataset import DynamicDamageDataset
import torch

def inference(model_repo_id="shyamsn97/table-cube-regen-damage-detection-20"):
    loaded_model, config = load_weights_from_huggingface(
        model=None,  # Will create model from config
        repo_id=model_repo_id,
        filename="pytorch_model.pt",
        load_config=True,
        config_filename="config.json"
    )
    labels = np.load('../data/ydata_7class.npy')
    shapes = np.load('../data/xdata_7class.npy')

    dataset = DynamicDamageDataset(shapes, labels, damage_radius_range=(2, 3), damage_types=["sphere", "cube", "random"], random_proportion_range=(0.1, 0.2), fixed_damage=False, augment_rotations=False, return_damage_mask=True, seed=None, filter_label=3)
    shape = dataset.shapes[0]

    damage_mask, damage_direction, label, _ = dataset[0]
    state = loaded_model.initialize(damage_mask.unsqueeze(0))
    with torch.no_grad():
        for i in range(96):
            state = loaded_model(state, label.unsqueeze(0))
            predictions = loaded_model.classify(state)
            predictions = torch.argmax(predictions, dim=-1).detach().cpu().numpy()[0]
    print(predictions.shape)
    print(damage_mask.shape)
    print(damage_direction.shape)
    print("Accuracy: ", torch.mean((predictions.squeeze() == damage_direction.squeeze()).float()))

if __name__ == "__main__":
    inference()