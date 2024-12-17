import torch
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from utils import load_models, recommend_books

# Загружаем модели
collab_model, cbf_model, merged_df, index_to_title = load_models()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обрабатывает команду /start."""
    await update.message.reply_text("Привет! Я рекомендую книги. Напишите название книги или автора.")

async def recommend(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обрабатывает команду /recommend."""
    user_input = " ".join(context.args)
    if not user_input:
        await update.message.reply_text("Пожалуйста, введите название книги.")
        return

    try:
        # Получение рекомендаций
        recommendations = recommend_books(user_input, collab_model, cbf_model, merged_df, index_to_title)
        if recommendations:
            response = "Рекомендации:\n" + "\n".join(recommendations)
        else:
            response = "Книги не найдены."
    except Exception as e:
        response = f"Произошла ошибка: {e}"

    await update.message.reply_text(response)

def main() -> None:
    """Основная функция."""
    # Создаём приложение
    application = Application.builder().token("Ваш_Токен_Бота").build()

    # Регистрируем обработчики команд
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("recommend", recommend))

    # Запуск бота
    application.run_polling()

if __name__ == "__main__":
    main()
