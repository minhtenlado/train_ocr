"""
Configuration file for Square CRNN OCR training and inference.

This file centralizes all hyperparameters and configuration settings
to make it easy to experiment with different model configurations.
"""

# ============================================================
# Dataset Configuration
# ============================================================
DATASET_CONFIG = {
    'csv_path': '/content/drive/MyDrive/AIOT/train.csv',  # Update this path
    'image_dir': '/content/drive/MyDrive/AIOT/images',    # Update this path
    'image_size': 128,
    'train_split': 0.8,
}

# ============================================================
# Model Configuration
# ============================================================
MODEL_CONFIG = {
    'num_classes': 36,  # 10 digits + 26 letters
    'hidden_size': 256,
    'dropout_rate': 0.3,
}

# ============================================================
# Training Configuration
# ============================================================
TRAINING_CONFIG = {
    'batch_size': 32,
    'num_epochs': 150,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'gradient_clip_max_norm': 5.0,
    'num_workers': 2,
    'pin_memory': True,
}

# ============================================================
# Optimizer Configuration
# ============================================================
OPTIMIZER_CONFIG = {
    'optimizer': 'adamw',  # 'adam' or 'adamw'
    'lr': 0.001,
    'weight_decay': 1e-4,
    'betas': (0.9, 0.999),
}

# ============================================================
# Scheduler Configuration
# ============================================================
SCHEDULER_CONFIG = {
    'scheduler': 'reducing',  # 'reducing' or 'step'
    'patience': 4,
    'factor': 0.5,
    'min_lr': 1e-6,
    'step_size': 10,  # For step scheduler
    'gamma': 0.1,     # For step scheduler
}

# ============================================================
# Loss Configuration
# ============================================================
LOSS_CONFIG = {
    'loss': 'ctc',
    'blank_index': 0,
    'reduction': 'mean',
}

# ============================================================
# Character Mapping
# ============================================================
CHARACTER_SET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-."

# ============================================================
# Paths Configuration
# ============================================================
PATHS_CONFIG = {
    'model_save_dir': '/content/drive/MyDrive/AIOT/',
    'model_name': 'best_square_ocr_pro.pth',
    'checkpoint_dir': './checkpoints/',
    'log_dir': './logs/',
}

# ============================================================
# Device Configuration
# ============================================================
DEVICE_CONFIG = {
    'use_cuda': True,
    'device': 'cuda',  # Will be automatically set to 'cpu' if CUDA unavailable
}

# ============================================================
# Data Augmentation Configuration
# ============================================================
AUGMENTATION_CONFIG = {
    'brightness': {
        'alpha_range': (0.7, 1.3),
        'beta_range': (-30, 30),
        'enabled': True,
    },
    'rotation': {
        'angle_range': (-5, 5),
        'enabled': True,
        'probability': 0.5,
    },
    'noise': {
        'std': 15,
        'enabled': True,
        'probability': 0.3,
    },
}

# ============================================================
# Inference Configuration
# ============================================================
INFERENCE_CONFIG = {
    'confidence_threshold': 0.5,
    'beam_width': 10,  # For beam search decoding
    'max_length': 50,
}


def get_config(section: str = None):
    """
    Get configuration dictionary.
    
    Args:
        section: Specific configuration section to retrieve.
                If None, returns all configurations.
    
    Returns:
        Configuration dictionary
    """
    all_config = {
        'dataset': DATASET_CONFIG,
        'model': MODEL_CONFIG,
        'training': TRAINING_CONFIG,
        'optimizer': OPTIMIZER_CONFIG,
        'scheduler': SCHEDULER_CONFIG,
        'loss': LOSS_CONFIG,
        'character_set': CHARACTER_SET,
        'paths': PATHS_CONFIG,
        'device': DEVICE_CONFIG,
        'augmentation': AUGMENTATION_CONFIG,
        'inference': INFERENCE_CONFIG,
    }
    
    if section:
        return all_config.get(section, {})
    return all_config


if __name__ == "__main__":
    # Print all configurations
    import json
    config = get_config()
    for section, values in config.items():
        print(f"\n{section.upper()}:")
        if isinstance(values, dict):
            for key, value in values.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {values}")
