from omegaconf import OmegaConf
import os

# change dir to the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
# Load the configuration file
config = OmegaConf.load('config.yaml')

# Override the factor values dynamically
overrides = OmegaConf.create({
    'data_augmentation': {
        'layers': {
            'RandomTranslation': {
                'factor': 0.2
            },
            'RandomZoom': {
                'factor': 0.2
            }
        }
    }
})

# Merge the overrides with the original configuration
config = OmegaConf.merge(config, overrides)

# Access the updated configuration parameters
translation_height_factor = config.data_augmentation.layers.RandomTranslation.kwargs.height_factor
translation_width_factor = config.data_augmentation.layers.RandomTranslation.kwargs.width_factor
zoom_height_factor = config.data_augmentation.layers.RandomZoom.kwargs.height_factor
zoom_width_factor = config.data_augmentation.layers.RandomZoom.kwargs.width_factor

# Print the factors to verify
print(f"Translation height factor: {translation_height_factor}")
print(f"Translation width factor: {translation_width_factor}")
print(f"Zoom height factor: {zoom_height_factor}")
print(f"Zoom width factor: {zoom_width_factor}")
print(f"layers: {config.data_augmentation.layers}")

# Proceed with your data augmentation and training logic using the updated config parameters
