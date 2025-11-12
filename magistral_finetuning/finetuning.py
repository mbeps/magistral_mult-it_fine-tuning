import json
import torch
import re
import os
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
from tqdm import tqdm
from dotenv import load_dotenv
from .config import MagistralFineTuningConfig, ThinkingMode


class MagistralFineTuning:
    """
    QLoRA fine-tuning class for Magistral Small.
    Simplified implementation focused on efficiency and reliability.
    """

    def __init__(self, config: MagistralFineTuningConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self._setup_environment()

    def _setup_environment(self):
        """Load environment variables."""
        load_dotenv()
        self.hf_token = os.getenv("HF_TOKEN")

        if not self.hf_token:
            raise ValueError("HF_TOKEN not found in .env file")

    @staticmethod
    def load_jsonl(file_path: str) -> list:
        """Load data from JSONL file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def _get_quantization_config(self):
        """Create BitsAndBytesConfig for 4-bit quantization."""
        return BitsAndBytesConfig(
            load_in_4bit=self.config.load_in_4bit,
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
        )

    def format_prompt(self, example: dict, force_thinking: bool = None) -> dict:
        """
        Format a single example using chat template.
        Supports both standard QA and Chain-of-Thought formats.

        Args:
            example: Dict with 'question', 'options', 'answer' and optionally 'thinking'
            force_thinking: Override thinking mode for this specific example
        Returns:
            Dict with formatted 'text' field
        """
        question = example["question"]
        options = example["options"]
        answer = example["answer"]
        thinking = example.get("thinking", None)

        # Determine if this example should use thinking
        use_thinking = (
            force_thinking
            if force_thinking is not None
            else (
                self.config.thinking_mode == ThinkingMode.ENABLED
                or (
                    self.config.thinking_mode == ThinkingMode.MIXED
                    and thinking is not None
                )
            )
        )

        # Format options
        options_text = "\n".join(
            [f"{list(opt.keys())[0]}) {list(opt.values())[0]}" for opt in options]
        )

        # Create user message
        user_content = f"Domanda: {question}\n\n{options_text}"

        # Create assistant response based on thinking mode and data availability
        if use_thinking and thinking:
            # Thinking mode: include reasoning in <think> tags
            assistant_content = f"<think>\n{thinking}\n</think>\n\n{answer}"
        else:
            # Standard mode: direct answer only
            assistant_content = answer

        # Create messages for chat template
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]

        # Apply chat template (Magistral tokenizer has different interface)
        try:
            # Try with standard parameters first
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            )
        except Exception as e:
            # Fallback to manual formatting if tokenizer doesn't support chat template
            if use_thinking and thinking:
                text = f"<s>[INST]{user_content}[/INST]<think>\n{thinking}\n</think>\n\n{answer}</s>"
            else:
                text = f"<s>[INST]{user_content}[/INST]{answer}</s>"

        return {"text": text}

    def prepare_dataset(self, data: list) -> Dataset:
        """
        Prepare dataset for training.
        Simple, single-process approach that works.
        """

        formatted_data = []
        thinking_count = 0
        mixed_thinking_count = 0

        for example in tqdm(data, desc="Formatting"):
            formatted = self.format_prompt(example)
            formatted_data.append(formatted)

            # Count examples with thinking content in original data
            if example.get("thinking"):
                thinking_count += 1
                # In mixed mode, check if this example actually used thinking
                if self.config.thinking_mode == ThinkingMode.MIXED:
                    if "<think>" in formatted["text"]:
                        mixed_thinking_count += 1

        # Print dataset statistics
        print(f"Dataset prepared: {len(formatted_data)} examples")
        if self.config.thinking_mode == ThinkingMode.ENABLED:
            print(f"All examples using thinking mode")
        elif self.config.thinking_mode == ThinkingMode.MIXED:
            print(
                f"Mixed training: {mixed_thinking_count} thinking, {len(formatted_data) - mixed_thinking_count} non-thinking"
            )
            print(f"Original data had {thinking_count} examples with thinking content")
        else:
            print(f"All examples using non-thinking mode")

        dataset = Dataset.from_list(formatted_data)
        return dataset

    @staticmethod
    def analyze_data(data: list, name: str):
        """Analyze dataset distribution."""
        categories = {}
        answers = {}
        thinking_examples = 0

        for item in data:
            cat = item.get("category", "unknown")
            ans = item.get("answer", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
            answers[ans] = answers.get(ans, 0) + 1

            if item.get("thinking"):
                thinking_examples += 1

        print(f"{name}: {len(data)} examples, {len(categories)} categories")
        print(
            f"Answer distribution: {', '.join(f'{k}:{v}' for k, v in sorted(answers.items()))}"
        )
        if thinking_examples > 0:
            print(
                f"Examples with thinking: {thinking_examples} ({thinking_examples / len(data) * 100:.1f}%)"
            )

    def setup_model(self):
        """
        Load model and tokenizer with QLoRA quantization.
        """
        # Get quantization config
        quantization_config = self._get_quantization_config()

        # Load tokenizer first with proper configuration
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
                token=self.hf_token,
                use_fast=False,  # Force slow tokenizer to avoid conversion issues
            )
        except Exception as e:
            print(f"Failed to load tokenizer with use_fast=False, trying default: {e}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name, trust_remote_code=True, token=self.hf_token
            )

        # Set padding token (handle Magistral tokenizer specifics)
        if self.tokenizer.pad_token is None or self.tokenizer.pad_token_id is None:
            # For Magistral, we need to be more careful with pad token setup
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                # Try to set pad_token string carefully
                try:
                    # Test if we can decode the eos token properly
                    decoded = self.tokenizer.decode([self.tokenizer.eos_token_id])
                    if len(decoded.split()) == 1:  # Should be a single token
                        self.tokenizer.pad_token = decoded
                    else:
                        # If it decodes to multiple pieces, use the eos_token string
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                except:
                    # Fallback to eos_token string
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                # Last resort fallback
                self.tokenizer.pad_token_id = 0
                self.tokenizer.pad_token = "</s>"

            self.tokenizer.padding_side = "right"

        # Load model with 4-bit quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            trust_remote_code=True,
            token=self.hf_token,
        )

        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)

        # Configure QLoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def setup_trainer(self, train_data: list):
        """
        Set up trainer with optimized configuration for QLoRA.
        """
        # Prepare dataset
        train_dataset = self.prepare_dataset(train_data)

        # Training arguments optimized for QLoRA and 24B model
        training_args = SFTConfig(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            gradient_checkpointing=self.config.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False}
            if self.config.gradient_checkpointing
            else None,
            learning_rate=self.config.learning_rate,
            lr_scheduler_type=self.config.lr_scheduler_type,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=0.01,
            logging_steps=50,
            save_strategy="epoch",
            seed=42,
            bf16=True,  # Use BF16 for training
            tf32=True,  # Enable tensor float 32
            optim="adamw_bnb_8bit",  # 8-bit optimizer for memory efficiency
            max_length=self.config.max_length,
            packing=True,  # Pack sequences for efficiency
            dataset_text_field="text",
            dataloader_num_workers=self.config.dataloader_num_workers,
            dataloader_pin_memory=True,
            torch_empty_cache_steps=4,  # Clear cache periodically
        )

        # Create trainer with explicit data collator to avoid pad token issues
        from transformers import DataCollatorForLanguageModeling

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
        )

        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_dataset,
            args=training_args,
            data_collator=data_collator,  # Explicit data collator
        )

        print(
            f"✓ Trainer ready ({len(train_dataset)} samples, {len(train_dataset) // self.config.effective_batch_size * self.config.num_epochs} steps)"
        )
        print(f"✓ QLoRA mode: 4-bit quantization enabled")
        print(f"✓ Thinking mode: {self.config.thinking_mode.value}")
        print(f"✓ Effective batch size: {self.config.effective_batch_size}")

    def train(self):
        """Start training."""
        if self.trainer is None:
            raise ValueError("Trainer not set up. Call setup_trainer() first.")

        print(
            f"\nStarting QLoRA training with {self.config.thinking_mode.value} mode..."
        )
        self.trainer.train()
        print("✓ Training completed")

    def save_model(self):
        """Save the trained model."""
        if self.trainer is None:
            raise ValueError("No trainer available. Train the model first.")

        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        print(f"✓ Model saved to {self.config.output_dir}")

    @staticmethod
    def extract_answer(output: str) -> str:
        """Extract answer letter from model output."""
        if not output:
            return ""
        match = re.search(r"\b([ABCDE])\b", output.upper())
        return match.group(1) if match else ""

    def evaluate_model(self, test_data: list) -> float:
        """
        Evaluate model on test data.

        Args:
            test_data: List of test examples
        Returns:
            Accuracy score
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call setup_model() first.")

        correct = 0
        total = len(test_data)
        failed_extractions = 0

        # Put model in eval mode
        self.model.eval()

        # Set generation parameters based on thinking mode
        if self.config.thinking_mode in [ThinkingMode.ENABLED, ThinkingMode.MIXED]:
            gen_params = {
                "max_new_tokens": 2048,  # More tokens for thinking
                "do_sample": True,
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
            }
            enable_thinking = True
        else:
            gen_params = {
                "max_new_tokens": 256,  # Fewer tokens for direct answers
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
            }
            enable_thinking = False

        for example in tqdm(test_data, desc="Evaluating"):
            # Format prompt for inference
            options_text = "\n".join(
                [
                    f"{list(opt.keys())[0]}) {list(opt.values())[0]}"
                    for opt in example["options"]
                ]
            )

            messages = [
                {
                    "role": "user",
                    "content": f"Domanda: {example['question']}\n\n{options_text}",
                }
            ]

            # For mixed mode, use thinking if the example has thinking content
            if self.config.thinking_mode == ThinkingMode.MIXED:
                enable_thinking = bool(example.get("thinking"))

            # Apply chat template for inference (handle Magistral tokenizer differences)
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                )
                # Add generation prompt manually if needed
                if not prompt.endswith("[/INST]"):
                    prompt += "[/INST]"
            except Exception:
                # Manual formatting fallback
                prompt = f"<s>[INST]{messages[0]['content']}[/INST]"

            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length,
            )

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs.to(self.model.device),
                    pad_token_id=self.tokenizer.eos_token_id,
                    **gen_params,
                )

            # Decode and extract answer
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )

            predicted = self.extract_answer(response)

            if not predicted:
                failed_extractions += 1
            elif predicted == example["answer"]:
                correct += 1

        accuracy = correct / total

        print(f"\nEvaluation Results:")
        print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print(f"Correct: {correct}/{total}")
        if failed_extractions > 0:
            print(f"Failed extractions: {failed_extractions}")

        return accuracy

    def run_complete_pipeline(self, train_data: list, test_data: list = None):
        """
        Run the complete training pipeline.

        Args:
            train_data: Training examples
            test_data: Optional test examples for evaluation
        """
        # Analyze data
        self.analyze_data(train_data, "Training")
        if test_data:
            self.analyze_data(test_data, "Test")

        # Setup and train
        self.setup_model()
        self.setup_trainer(train_data)
        self.train()
        self.save_model()

        # Evaluate if test data provided
        if test_data:
            accuracy = self.evaluate_model(test_data)
            return accuracy

        return None
