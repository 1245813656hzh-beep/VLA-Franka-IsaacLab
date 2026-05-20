.PHONY: help install install-dev clean test format lint

PYTHON := python3
PIP := pip

help:
	@echo "VLA Franka IsaacLab - Available commands:"
	@echo ""
	@echo "  make install      - Install package dependencies"
	@echo "  make install-dev  - Install with dev dependencies"
	@echo "  make clean        - Clean generated files"
	@echo "  make format       - Format code with ruff"
	@echo "  make lint         - Lint code with ruff"
	@echo "  make test         - Run tests"
	@echo ""
	@echo "Data Collection:"
	@echo "  make collect-single  - Collect single cube pick-place demos"
	@echo "  make collect-stack   - Collect multi-cube stacking demos"
	@echo ""
	@echo "Inference:"
	@echo "  make infer-gr00t     - Run GR00T inference"
	@echo "  make infer-act       - Run ACT inference"
	@echo ""

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/

format:
	ruff format scripts/ tasks/ configs/

lint:
	ruff check scripts/ tasks/ configs/

test:
	pytest tests/ -v

# Data collection shortcuts
collect-single:
	@echo "Collecting single cube pick-place demos..."
	@echo "Run: python scripts/data_collection/auto_collect_single.py \\"
	@echo "      --num_episodes 100 \\"
	@echo "      --output_dir datasets/lerobot/franka_place_bin \\"
	@echo "      --repo_id local/franka_place_bin \\"
	@echo "      --use_videos \\"
	@echo "      --target_color green"

collect-stack:
	@echo "Collecting multi-cube stacking demos..."
	@echo "Run: python scripts/data_collection/auto_collect_stack.py \\"
	@echo "      --num_episodes 100 \\"
	@echo "      --output_dir datasets/lerobot/franka_stack \\"
	@echo "      --repo_id local/franka_stack \\"
	@echo "      --use_videos"

# Inference shortcuts
infer-gr00t:
	@echo "Running GR00T inference..."
	@echo "Run: python scripts/inference/inference_gr00t_isaaclab.py \\"
	@echo "      --model_path ./pretrained_models/gr00t_place_bin \\"
	@echo "      --num_episodes 5 \\"
	@echo "      --save_video"

infer-act:
	@echo "Running ACT inference..."
	@echo "Run: python scripts/inference/inference_act_isaaclab.py \\"
	@echo "      --model_path ./pretrained_models/act_place_bin \\"
	@echo "      --dataset_path ./datasets/lerobot/auto_collected \\"
	@echo "      --num_episodes 5 \\"
	@echo "      --save_video"
