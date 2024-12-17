import torch
import pandas as pd

# Глобальная переменная для DataFrame с данными о книгах
merged = pd.read_csv("./merged_dataframe_with_sentiment_labels.csv")
title_sentiment_aggregated = merged.groupby(
    ['Title', 'authors', 'categories']
)['sentiment_score'].mean().reset_index()

def get_collaborative_recommendations(model, title, num_recommendations=10):
    """Получение рекомендаций с использованием коллаборативной фильтрации."""
    # Словарь для преобразования названия книги в индекс
    item_to_index = {title: idx for idx, title in enumerate(merged['Title'].unique())}

    if title not in item_to_index:
        return []

    input_title_index = item_to_index[title]

    # Получение рекомендаций
    model.eval()
    with torch.no_grad():
        similar_titles = model.get_similar_titles(input_title_index, top_k=num_recommendations)

    return similar_titles

def get_content_based_recommendations(content_based_model, collaborative_recommendations):
    """Получение рекомендаций с использованием контентной фильтрации."""
    title_details = title_sentiment_aggregated.set_index('Title')[['categories', 'authors', 'sentiment_score']].to_dict(orient='index')

    details = [title_details[title] for title in collaborative_recommendations if title in title_details]
    if not details:
        return []

    # Преобразование данных в тензоры
    unique_categories = merged['categories'].unique()
    unique_authors = merged['authors'].unique()
    category_to_index = {category: idx for idx, category in enumerate(unique_categories)}
    author_to_index = {author: idx for idx, author in enumerate(unique_authors)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    category_indices = torch.tensor([category_to_index[detail['categories']] for detail in details], dtype=torch.long).to(device)
    author_indices = torch.tensor([author_to_index[detail['authors']] for detail in details], dtype=torch.long).to(device)
    sentiment_scores = torch.tensor([detail['sentiment_score'] for detail in details], dtype=torch.float32).to(device)

    # Получение предсказаний от модели контентной фильтрации
    content_based_model.eval()
    with torch.no_grad():
        predictions = content_based_model(category_indices, author_indices, sentiment_scores)

    # Сортировка заголовков на основе предсказаний
    sorted_titles = [
        title for _, title in sorted(zip(predictions.cpu().numpy(), collaborative_recommendations), reverse=True)
    ]

    return sorted_titles

def load_models():
    """Загрузка моделей и данных."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загрузка коллаборативной модели
    collab_model_path = "./collaborative_model.pth"
    collab_model = torch.load(collab_model_path, map_location=device)

    # Загрузка контентной модели
    cbf_model_path = "./content_based_model.pth"
    cbf_model = torch.load(cbf_model_path, map_location=device)

    # Подготовка данных
    index_to_title = {idx: title for idx, title in enumerate(merged['Title'].unique())}

    return collab_model, cbf_model, merged, index_to_title

def recommend_books(user_input, collab_model, cbf_model, merged_df, index_to_title):
    """Рекомендация книг на основе пользовательского ввода."""
    matching_titles = merged_df[merged_df['Title'].str.contains(user_input, case=False, na=False)]
    if matching_titles.empty:
        return []

    input_title = matching_titles.iloc[0]['Title']

    collab_recommendations = get_collaborative_recommendations(collab_model, input_title)
    if not collab_recommendations:
        return []

    recommendations = get_content_based_recommendations(cbf_model, collab_recommendations)
    return recommendations
