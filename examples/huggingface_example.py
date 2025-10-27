#!/usr/bin/env python3
"""
Example script demonstrating how to save and load model weights and config to/from Hugging Face Hub.
"""

from regen.model import (
    NCA3DDamageDetection, 
    save_weights_to_huggingface, 
    load_weights_from_huggingface,
    save_config_to_json,
    load_config_from_json,
    create_model_from_config
)

def main():
    # Initialize model
    model = NCA3DDamageDetection(
        use_class_embeddings=True,
        num_hidden_channels=128,
        num_classes=7,
        num_damage_directions=7
    )
    
    print("Model initialized successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Example: Save model to Hugging Face Hub
    # Replace 'your-username' with your actual Hugging Face username
    repo_id = "shyamsn97/cube-regen-damage-detection"
    
    try:
        # Save to Hugging Face Hub (with config)
        print(f"\nSaving model and config to {repo_id}...")
        uploaded_files = save_weights_to_huggingface(
            model=model,
            repo_id=repo_id,
            commit_message="Initial NCA3D damage detection model with config",
            filename="pytorch_model.pt",  # You can use .pt, .pth, .bin, or any other extension
            save_config=True,  # Also save the model configuration
            config_filename="config.json"
        )
        print(f"Model and config saved successfully!")
        print(f"Weights URL: {uploaded_files['weights']}")
        print(f"Config URL: {uploaded_files['config']}")
        
    except Exception as e:
        print(f"Error saving model: {e}")
        print("Make sure you're logged in to Hugging Face Hub:")
        print("Run: huggingface-cli login")
        return
    
    # Example 1: Load model and config from Hugging Face Hub
    try:
        print(f"\nLoading model and config from {repo_id}...")
        loaded_model, config = load_weights_from_huggingface(
            model=None,  # Will create model from config
            repo_id=repo_id,
            filename="pytorch_model.pt",
            load_config=True,
            config_filename="config.json"
        )
        print("Model and config loaded successfully!")
        print(f"Loaded model parameters: {sum(p.numel() for p in loaded_model.parameters())}")
        print(f"Config: {config}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
    
    # Example 2: Load only weights (without config)
    try:
        print(f"\nLoading only weights from {repo_id}...")
        model_weights_only = NCA3DDamageDetection()  # Create with default config
        loaded_model_weights = load_weights_from_huggingface(
            model=model_weights_only,
            repo_id=repo_id,
            filename="pytorch_model.pt",
            load_config=False  # Don't load config
        )
        print("Weights loaded successfully!")
        
    except Exception as e:
        print(f"Error loading weights: {e}")
    
    # Example 3: Local config save/load
    try:
        print(f"\nDemonstrating local config save/load...")
        
        # Save config locally
        config = save_config_to_json(model, "local_config.json")
        print(f"Config saved locally: {config}")
        
        # Load config locally
        loaded_config = load_config_from_json("local_config.json")
        print(f"Config loaded locally: {loaded_config}")
        
        # Create model from config
        model_from_config = create_model_from_config(loaded_config)
        print(f"Model created from config with {sum(p.numel() for p in model_from_config.parameters())} parameters")
        
    except Exception as e:
        print(f"Error with local config: {e}")

if __name__ == "__main__":
    main()
