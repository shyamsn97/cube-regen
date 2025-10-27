from regen.model import load_weights_from_huggingface
import numpy as np
from regen.dataset import DynamicDamageDataset
import torch
from PIL import Image, ImageDraw, ImageFont
from regen.utils import plot_voxels

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

    dataset = DynamicDamageDataset(shapes, labels, damage_radius_range=(2, 3), damage_types=["sphere", "cube"], random_proportion_range=(0.1, 0.2), fixed_damage=False, augment_rotations=False, return_damage_mask=True, seed=None, filter_label=3)
    shape = dataset.shapes[0]

    damage_mask, damage_direction, label, _ = dataset[0]
    state = loaded_model.initialize(damage_mask.unsqueeze(0))
    with torch.no_grad():
        for i in range(128):
            state = loaded_model(state, label.unsqueeze(0))
            predictions = loaded_model.classify(state)
            predictions = torch.argmax(predictions, dim=-1)
    print(predictions.shape)
    print(damage_mask.shape)
    print(damage_direction.shape)
    print("Damage Direction Accuracy: ", torch.mean((predictions.squeeze()[predictions.squeeze() != 0] == damage_direction.squeeze()[predictions.squeeze() != 0]).float()))
    print("Undamaged Accuracy: ", torch.mean((predictions.squeeze()[predictions.squeeze() == 0] == damage_direction.squeeze()[predictions.squeeze() == 0]).float()))
    ground_truth_img = plot_voxels(live_mask=damage_mask.squeeze().detach().cpu().numpy().astype(np.uint8), damage_direction=damage_direction.squeeze().detach().cpu().numpy().astype(np.uint8),)
    predicted_img = plot_voxels(live_mask=damage_mask.squeeze().detach().cpu().numpy().astype(np.uint8), damage_direction=predictions.squeeze().detach().cpu().numpy().astype(np.uint8),)

    # Calculate dimensions for the combined image with labels
    label_height = 30  # Height for text labels
    combined_width = ground_truth_img.width + predicted_img.width
    combined_height = max(ground_truth_img.height, predicted_img.height) + label_height
    
    # Create the combined image with extra space for labels
    combined_img = Image.new("RGBA", (combined_width, combined_height), (255, 255, 255, 255))
    
    # Paste the images
    combined_img.paste(predicted_img, (0, 0))
    combined_img.paste(ground_truth_img, (predicted_img.width, 0))
    
    # Add text labels
    draw = ImageDraw.Draw(combined_img)
    try:
        font = ImageFont.truetype("Arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # Add "Prediction" label under the first image
    pred_text = "Prediction"
    pred_bbox = draw.textbbox((0, 0), pred_text, font=font)
    pred_text_width = pred_bbox[2] - pred_bbox[0]
    pred_x = (predicted_img.width - pred_text_width) // 2
    draw.text((pred_x, predicted_img.height + 5), pred_text, fill=(0, 0, 0, 255), font=font)
    
    # Add "Ground Truth" label under the second image
    gt_text = "Ground Truth"
    gt_bbox = draw.textbbox((0, 0), gt_text, font=font)
    gt_text_width = gt_bbox[2] - gt_bbox[0]
    gt_x = predicted_img.width + (ground_truth_img.width - gt_text_width) // 2
    draw.text((gt_x, ground_truth_img.height + 5), gt_text, fill=(0, 0, 0, 255), font=font)
    
    combined_img.save("combined_img.png")

if __name__ == "__main__":
    inference()