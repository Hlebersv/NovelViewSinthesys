from dataclasses import dataclass


@dataclass
class TrainingConfig:
    image_size = 512  # the generated image resolution
    train_batch_size = 6
    eval_batch_size = 6
    num_epochs = 100
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 200
    save_image_epochs = 1
    save_model_epochs = 1
    mixed_precision = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = 'ddpm-ffhq-512'

    push_to_hub = False
    hub_private_repo = False
    overwrite_output_dir = True
    seed = 0

