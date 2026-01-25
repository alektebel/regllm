"""
Telegram bot interface for the finetuned banking regulation model.
Allows querying the model remotely via Telegram.
"""

import asyncio
import logging
from pathlib import Path
from collections import defaultdict
import time
import sys
import argparse

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config import TELEGRAM, INFERENCE, MODEL
from src.ui.chat_interface import RegulationChatbot

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class TelegramBot:
    """Telegram bot for banking regulation queries."""

    # Available models (same as RegulationChatbot)
    AVAILABLE_MODELS = {
        'qwen2.5-7b': 'Qwen/Qwen2.5-7B-Instruct',
        'qwen2.5-3b': 'Qwen/Qwen2.5-3B-Instruct',
        'phi-3-mini': 'microsoft/Phi-3-mini-4k-instruct',
        'stablelm-3b': 'stabilityai/stablelm-3b-4e1t',
        'phi-2': 'microsoft/phi-2',
        'gemma-2b': 'google/gemma-2b-it',
        'qwen-1.8b': 'Qwen/Qwen2-1.8B-Instruct',
    }

    def __init__(self, base_model: str = None, model_path: str = None):
        """
        Initialize the Telegram bot.

        Args:
            base_model: Base model name (overrides config if provided)
            model_path: Path to finetuned model (overrides config if provided)
        """
        self.bot_token = TELEGRAM['bot_token']
        self.allowed_users = TELEGRAM.get('allowed_users', [])
        self.model_path = model_path or TELEGRAM.get('model_path')
        self.max_response_length = TELEGRAM.get('max_response_length', 500)

        # Base model: CLI arg > TELEGRAM config > MODEL config
        self.base_model = base_model or TELEGRAM.get('base_model') or MODEL.get('base_model')

        # Rate limiting
        self.rate_limit_messages = TELEGRAM.get('rate_limit_messages', 10)
        self.rate_limit_window = TELEGRAM.get('rate_limit_window', 60)
        self.user_message_times = defaultdict(list)

        # Model will be loaded on first message
        self.chatbot = None
        self.model_loaded = False

    def is_user_allowed(self, user_id: int) -> bool:
        """Check if user is allowed to use the bot."""
        if not self.allowed_users:
            return True
        return user_id in self.allowed_users

    def is_rate_limited(self, user_id: int) -> bool:
        """Check if user has exceeded rate limit."""
        current_time = time.time()
        # Clean old messages
        self.user_message_times[user_id] = [
            t for t in self.user_message_times[user_id]
            if current_time - t < self.rate_limit_window
        ]
        # Check limit
        if len(self.user_message_times[user_id]) >= self.rate_limit_messages:
            return True
        # Record this message
        self.user_message_times[user_id].append(current_time)
        return False

    def load_model(self):
        """Load the model (lazy loading on first request)."""
        if self.model_loaded:
            return

        logger.info("Loading model for Telegram bot...")
        logger.info(f"Base model: {self.base_model}")

        # Find the latest model if path is a directory
        model_path = Path(self.model_path)
        if model_path.is_dir():
            # Look for the latest run directory with final_model
            run_dirs = sorted(model_path.glob("run_*/final_model"), reverse=True)
            if run_dirs:
                model_path = run_dirs[0]
            else:
                # Try direct final_model path
                if (model_path / "final_model").exists():
                    model_path = model_path / "final_model"

        logger.info(f"Using model path: {model_path}")
        self.chatbot = RegulationChatbot(
            model_path=str(model_path),
            base_model=self.base_model
        )
        self.model_loaded = True
        logger.info("Model loaded successfully!")

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        user_id = update.effective_user.id

        if not self.is_user_allowed(user_id):
            await update.message.reply_text(
                "Lo siento, no tienes permiso para usar este bot."
            )
            return

        welcome_message = f"""
Hola! Soy un asistente experto en regulacion bancaria espanola.

Modelo actual: {self.base_model}

Puedes preguntarme sobre:
- Probabilidad de Default (PD)
- Loss Given Default (LGD)
- Metodo IRB
- Regulacion de riesgo de credito
- Normativa del Banco de Espana, BCE, etc.

Comandos disponibles:
/start - Mostrar este mensaje
/help - Ayuda
/status - Estado del modelo
/model - Ver modelos disponibles

Simplemente escribe tu pregunta y te respondere!
        """
        await update.message.reply_text(welcome_message)

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        help_text = """
Como usar este bot:

1. Escribe tu pregunta sobre regulacion bancaria
2. Espera la respuesta (puede tardar unos segundos)

Ejemplos de preguntas:
- Que es la probabilidad de default?
- Como se calcula el LGD?
- Que es el metodo IRB?
- Cuales son los requisitos de capital para riesgo de credito?

Nota: Las respuestas se basan en documentos oficiales del Banco de Espana, BCE, BOE y Comite de Basilea.
        """
        await update.message.reply_text(help_text)

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command."""
        user_id = update.effective_user.id

        if not self.is_user_allowed(user_id):
            await update.message.reply_text("No autorizado.")
            return

        status = f"""
Estado del bot:

Modelo cargado: {'Si' if self.model_loaded else 'No (se cargara con el primer mensaje)'}
Modelo base: {self.base_model}
Path del modelo: {self.model_path}
Usuario autorizado: {'Si' if self.is_user_allowed(user_id) else 'No'}
        """
        await update.message.reply_text(status)

    async def model_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /model command - show current model and available models."""
        user_id = update.effective_user.id

        if not self.is_user_allowed(user_id):
            await update.message.reply_text("No autorizado.")
            return

        current_marker = lambda m: " (actual)" if m == self.base_model else ""

        models_text = f"""
Modelo actual: {self.base_model}

Modelos disponibles:
{chr(10).join(f"  - {name}{current_marker(name)}" for name in self.AVAILABLE_MODELS.keys())}

Nota: Para cambiar el modelo, reinicia el bot con --base-model <modelo>
        """
        await update.message.reply_text(models_text)

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle incoming messages."""
        user_id = update.effective_user.id
        username = update.effective_user.username or "Unknown"

        # Check authorization
        if not self.is_user_allowed(user_id):
            logger.warning(f"Unauthorized access attempt from user {user_id} ({username})")
            await update.message.reply_text(
                "Lo siento, no tienes permiso para usar este bot."
            )
            return

        # Check rate limit
        if self.is_rate_limited(user_id):
            await update.message.reply_text(
                f"Has enviado demasiados mensajes. Espera {self.rate_limit_window} segundos."
            )
            return

        question = update.message.text
        logger.info(f"Question from {username} ({user_id}): {question}")

        # Send typing indicator
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

        try:
            # Load model if needed
            if not self.model_loaded:
                await update.message.reply_text("Cargando modelo... (esto puede tardar un momento)")
                self.load_model()

            # Generate response
            response = self.chatbot.generate_response(
                question,
                max_length=self.max_response_length
            )

            # Telegram has a 4096 character limit
            if len(response) > 4000:
                response = response[:4000] + "...\n\n(Respuesta truncada)"

            await update.message.reply_text(response)
            logger.info(f"Response sent to {username}")

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            await update.message.reply_text(
                "Lo siento, hubo un error al procesar tu pregunta. Por favor, intenta de nuevo."
            )

    def run(self):
        """Run the Telegram bot."""
        if self.bot_token == 'YOUR_TELEGRAM_BOT_TOKEN_HERE':
            logger.error("Please set your Telegram bot token in config.py")
            logger.info("To get a token:")
            logger.info("1. Message @BotFather on Telegram")
            logger.info("2. Send /newbot and follow instructions")
            logger.info("3. Copy the token to config.py TELEGRAM['bot_token']")
            return

        logger.info("Starting Telegram bot...")
        logger.info(f"Using base model: {self.base_model}")
        logger.info(f"Model path: {self.model_path}")

        # Create application
        application = Application.builder().token(self.bot_token).build()

        # Add handlers
        application.add_handler(CommandHandler("start", self.start_command))
        application.add_handler(CommandHandler("help", self.help_command))
        application.add_handler(CommandHandler("status", self.status_command))
        application.add_handler(CommandHandler("model", self.model_command))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

        # Run the bot
        logger.info("Bot is running! Press Ctrl+C to stop.")
        application.run_polling(allowed_updates=Update.ALL_TYPES)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Telegram Bot for Banking Regulation Queries')
    parser.add_argument('--base-model', type=str, default=None,
                       choices=list(TelegramBot.AVAILABLE_MODELS.keys()),
                       help='Base model to use (default from config.py)')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to finetuned model (default from config.py)')

    args = parser.parse_args()

    bot = TelegramBot(
        base_model=args.base_model,
        model_path=args.model_path
    )
    bot.run()


if __name__ == "__main__":
    main()
