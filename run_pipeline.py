#!/usr/bin/env python3
"""
Main pipeline script for RegLLM project.
Runs the complete workflow from scraping to training.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from scraper.regulation_scraper import RegulationScraper
from preprocessing.data_processor import DataProcessor
from training.model_setup import ModelSetup
from training.train import RegulationTrainer

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def print_banner():
    """Print welcome banner."""
    print("=" * 70)
    print(" " * 10 + "Spanish Banking Regulation LLM (RegLLM)")
    print("=" * 70)
    print()


def run_scraping():
    """Run the web scraping step."""
    logger.info("STEP 1: Web Scraping")
    logger.info("-" * 50)

    scraper = RegulationScraper()
    documents = scraper.run()

    logger.info(f"✓ Scraping complete! Collected {len(documents)} documents\n")
    return documents


def run_preprocessing():
    """Run the data preprocessing step."""
    logger.info("STEP 2: Data Preprocessing")
    logger.info("-" * 50)

    processor = DataProcessor()
    train_data, val_data = processor.process()

    logger.info(f"✓ Preprocessing complete!")
    logger.info(f"  Training examples: {len(train_data)}")
    logger.info(f"  Validation examples: {len(val_data)}\n")

    return train_data, val_data


def run_model_setup(model_name: str = 'phi-2'):
    """Test model setup."""
    logger.info("STEP 3: Model Setup")
    logger.info("-" * 50)

    setup = ModelSetup(model_name=model_name)

    logger.info("Loading model for verification...")
    model, tokenizer = setup.load_model_and_tokenizer(use_4bit=True)

    logger.info("✓ Model loaded successfully!")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Parameters: {model.num_parameters() / 1e9:.2f}B\n")

    return True


def run_overfitting_test(model_name: str = 'phi-2', epochs: int = 10):
    """Run overfitting test on small subset."""
    logger.info("STEP 4: Overfitting Test (Small Subset)")
    logger.info("-" * 50)
    logger.info("This verifies the training pipeline works correctly.")
    logger.info("The model should overfit on the small dataset (loss → 0)\n")

    trainer = RegulationTrainer(
        model_name=model_name,
        use_small_subset=True
    )

    trainer.train(
        num_epochs=epochs,
        batch_size=2,
        learning_rate=3e-4
    )

    logger.info("✓ Overfitting test complete! Check logs/training_plot_*.png\n")

    return trainer


def run_full_training(model_name: str = 'phi-2', epochs: int = 3,
                     batch_size: int = 4, learning_rate: float = 2e-4):
    """Run full training on complete dataset."""
    logger.info("STEP 5: Full Dataset Training")
    logger.info("-" * 50)
    logger.info("Training on complete dataset. This may take a while...\n")

    trainer = RegulationTrainer(
        model_name=model_name,
        use_small_subset=False
    )

    trainer.train(
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )

    logger.info("✓ Training complete!\n")

    return trainer


def main():
    """Main pipeline."""
    parser = argparse.ArgumentParser(
        description='Run the RegLLM pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python run_pipeline.py --all

  # Run specific steps
  python run_pipeline.py --scrape --preprocess
  python run_pipeline.py --train-small
  python run_pipeline.py --train-full

  # Customize model and training
  python run_pipeline.py --train-full --model phi-3-mini --epochs 5
        """
    )

    # Pipeline steps
    parser.add_argument('--all', action='store_true',
                       help='Run all steps (scrape, preprocess, train)')
    parser.add_argument('--scrape', action='store_true',
                       help='Run web scraping')
    parser.add_argument('--preprocess', action='store_true',
                       help='Run data preprocessing')
    parser.add_argument('--setup', action='store_true',
                       help='Test model setup')
    parser.add_argument('--train-small', action='store_true',
                       help='Run overfitting test on small subset')
    parser.add_argument('--train-full', action='store_true',
                       help='Run full training')

    # Model options
    parser.add_argument('--model', type=str, default='phi-2',
                       help='Model to use (default: phi-2)')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs (default: 3)')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size (default: 4)')
    parser.add_argument('--lr', type=float, default=2e-4,
                       help='Learning rate (default: 2e-4)')

    args = parser.parse_args()

    # If no arguments, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return

    print_banner()

    try:
        # Run scraping
        if args.all or args.scrape:
            run_scraping()

        # Run preprocessing
        if args.all or args.preprocess:
            run_preprocessing()

        # Test model setup
        if args.all or args.setup:
            run_model_setup(args.model)

        # Run overfitting test
        if args.all or args.train_small:
            run_overfitting_test(args.model, epochs=10)

        # Run full training
        if args.train_full or (args.all and not args.train_small):
            run_full_training(
                args.model,
                args.epochs,
                args.batch_size,
                args.lr
            )

        logger.info("=" * 70)
        logger.info("Pipeline complete! 🎉")
        logger.info("=" * 70)
        logger.info("\nNext steps:")
        logger.info("1. Check training plots in logs/")
        logger.info("2. Find your trained model in models/finetuned/")
        logger.info("3. Launch the UI with:")
        logger.info("   python src/ui/chat_interface.py --model-path models/finetuned/run_XXXXXX/final_model")
        logger.info()

    except KeyboardInterrupt:
        logger.info("\n\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\nPipeline failed: {e}")
        logger.exception("Full error:")
        sys.exit(1)


if __name__ == "__main__":
    main()
