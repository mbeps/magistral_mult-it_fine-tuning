from enum import Enum


class ThinkingMode(Enum):
    """Enumeration for thinking mode options."""

    DISABLED = "disabled"  # No thinking tokens (default, backwards compatible)
    ENABLED = "enabled"  # All examples use thinking tokens
    MIXED = "mixed"  # Mix of thinking and non-thinking examples


class MagistralFineTuningConfig:
    """
    Configuration for Magistral Small fine-tuning with QLoRA.
    Optimised for 24B parameter model on consumer hardware.
    """

    def __init__(
        self,
        model_name: str = "mistralai/Magistral-Small-2506",
        train_file: str = "data/train.jsonl",
        output_dir: str = "./results",
        batch_size: int = 2,  # Conservative for 24B model with QLoRA
        gradient_accumulation_steps: int = 4,  # Effective batch size = 8
        learning_rate: float = 2e-4,  # Higher for QLoRA
        warmup_ratio: float = 0.1,
        lr_scheduler_type: str = "cosine",
        num_epochs: int = 1,  # Single epoch as requested
        max_length: int = 512,
        # QLoRA-specific parameters
        lora_r: int = 16,  # Rank
        lora_alpha: int = 32,  # 2*r scaling
        lora_dropout: float = 0.05,  # Light regularisation
        target_modules: list | None = None,
        # 4-bit quantization settings
        load_in_4bit: bool = True,
        bnb_4bit_quant_type: str = "nf4",  # NormalFloat 4-bit
        bnb_4bit_compute_dtype: str = "bfloat16",  # Compute in BF16
        bnb_4bit_use_double_quant: bool = True,  # Double quantization
        # Training optimizations
        dataloader_num_workers: int = 4,
        gradient_checkpointing: bool = True,  # Essential for 24B model
        thinking_mode: ThinkingMode = ThinkingMode.DISABLED,
    ) -> None:
        self.model_name: str = model_name
        self.train_file: str = train_file
        self.output_dir: str = output_dir
        self.gradient_accumulation_steps: int = gradient_accumulation_steps
        self.warmup_ratio: float = warmup_ratio
        self.lr_scheduler_type: str = lr_scheduler_type
        self.num_epochs: int = num_epochs
        self.lora_r: int = lora_r
        self.lora_alpha: int = lora_alpha
        self.lora_dropout: float = lora_dropout
        self.load_in_4bit: bool = load_in_4bit
        self.bnb_4bit_quant_type: str = bnb_4bit_quant_type
        self.bnb_4bit_compute_dtype: str = bnb_4bit_compute_dtype
        self.bnb_4bit_use_double_quant: bool = bnb_4bit_use_double_quant
        self.dataloader_num_workers: int = dataloader_num_workers
        self.gradient_checkpointing: bool = gradient_checkpointing
        self.thinking_mode: ThinkingMode = thinking_mode

        # Adjust hyperparameters based on thinking mode
        if thinking_mode in [ThinkingMode.ENABLED, ThinkingMode.MIXED]:
            # Longer sequences for thinking mode
            self.batch_size: int = max(1, batch_size // 2)  # Reduce batch size
            self.learning_rate: float = learning_rate * 0.8  # Lower LR for stability
            if max_length == 512:
                self.max_length: int = 2048  # Increase for thinking
            else:
                self.max_length: int = max_length
        else:
            self.batch_size: int = batch_size
            self.learning_rate: float = learning_rate
            self.max_length: int = max_length

        # Mistral/Magistral target modules (based on research)
        if target_modules is None:
            self.target_modules: list[str] = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        else:
            self.target_modules = target_modules

    @property
    def effective_batch_size(self) -> int:
        """Calculate effective batch size."""
        return self.batch_size * self.gradient_accumulation_steps

    @property
    def enable_thinking(self) -> bool:
        """Backwards compatibility property."""
        return self.thinking_mode != ThinkingMode.DISABLED

    def print_config(self) -> None:
        """Print configuration summary."""
        print(f"Model: {self.model_name}")
        print(f"Learning rate: {self.learning_rate}, Epochs: {self.num_epochs}")
        print(f"Batch size: {self.batch_size} (effective: {self.effective_batch_size})")
        print(
            f"QLoRA: r={self.lora_r}, alpha={self.lora_alpha}, dropout={self.lora_dropout}"
        )
        print(
            f"4-bit: {self.bnb_4bit_quant_type}, double_quant={self.bnb_4bit_use_double_quant}"
        )
        print(f"Thinking mode: {self.thinking_mode.value}")
        print(f"Max length: {self.max_length}")
        print(f"Target modules: {self.target_modules}")
