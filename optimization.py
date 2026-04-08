import streamlit as st
import pandas as pd
import numpy as np
import optuna
import matplotlib.pyplot as plt
import shap
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

# Модели
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge


# --- 1. Функции предобработки (копируем из вашего основного скрипта) ---
# Лучше вынести это в отдельный utils.py, но для простого проекта продублируем здесь
@st.cache_data
def load_and_process_data():
    data = fetch_openml(data_id=42876, as_frame=True, parser='auto')
    df = data.frame.copy()

    # Dates
    df['DateTimeOfAccident'] = pd.to_datetime(df['DateTimeOfAccident'])
    df['DateReported'] = pd.to_datetime(df['DateReported'])
    df['AccidentMonth'] = df['DateTimeOfAccident'].dt.month
    df['AccidentDayOfWeek'] = df['DateTimeOfAccident'].dt.dayofweek
    df['ReportingDelay'] = (df['DateReported'] - df['DateTimeOfAccident']).dt.days

    # Feature Engineering
    df['Age_WeeklyPay_Interaction'] = df['Age'] * df['WeeklyPay']
    df['Estimate_to_Pay_Ratio'] = df['InitialCaseEstimate'] / (df['WeeklyPay'] + 1)
    df['Has_Dependents'] = (df['DependentChildren'] > 0).astype(int)

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=50)
    tfidf_matrix = tfidf.fit_transform(df['ClaimDescription'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())
    df = pd.concat([df.reset_index(drop=True), tfidf_df], axis=1)

    # Drop unused
    df = df.drop(columns=['DateTimeOfAccident', 'DateReported', 'ClaimDescription'])

    # Label Encoding for other cats
    for col in ['Gender', 'MaritalStatus', 'PartTimeFullTime']:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Scale Numerics
    num_cols = ['Age', 'DependentChildren', 'DependentsOther', 'WeeklyPay',
                'HoursWorkedPerWeek', 'DaysWorkedPerWeek', 'InitialCaseEstimate',
                'AccidentMonth', 'AccidentDayOfWeek', 'ReportingDelay',
                'Age_WeeklyPay_Interaction', 'Estimate_to_Pay_Ratio']
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df

def objective_xgb(trial, X, y):
    param = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': 1000,
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42,
        'n_jobs': -1
    }

    model = XGBRegressor(**param)
    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    return scores.mean()


def objective_cat(trial, X, y):
    param = {
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "iterations": 1000,
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
        "rsm": trial.suggest_float("rsm", 0.5, 1.0),
        "random_state": 42, "verbose": False
    }
    model = CatBoostRegressor(**param)
    scores = cross_val_score(model, X, y,)
    return scores.mean()


def objective_ridge(trial, X, y):
    alpha = trial.suggest_float('alpha', 1e-3, 100.0, log=True)
    model = Ridge(alpha=alpha)
    scores = cross_val_score(model, X, y,)
    return scores.mean()


# --- 3. Основная функция страницы ---
def optimization_page():
    st.title("Оптимизация гиперпараметров (Optuna)")

    st.write("""
    На этой странице происходит автоматический подбор лучших параметров для моделей.
    Это может занять несколько минут.
    """)

    # Загрузка данных
    with st.spinner("Подготовка данных..."):
        df = load_and_process_data()
        X = df.drop(columns=['UltimateIncurredClaimCost'])
        y = np.log1p(df['UltimateIncurredClaimCost'])  # Логарифмируем целевую

        # Делим на Train/Opt_Val
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


    # Сайдбар с настройками
    model_name = st.selectbox("Выберите модель для оптимизации", ["XGBoost", "CatBoost", "Ridge"])
    n_trials = st.slider("Количество попыток (Trials)", 10, 100, 20)

    # Кнопка запуска
    if st.button("Запустить оптимизацию", type="primary"):

        # Создаем прогресс бар и текстовое поле для логов
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Колбэк для обновления прогресса
        def callback(study, trial):
            progress = (trial.number + 1) / n_trials
            progress_bar.progress(progress)
            status_text.text(f"Попыток {trial.number + 1}/{n_trials}. Лучший R2: {study.best_value:.4f}")

        # Запуск Optuna
        study = optuna.create_study(direction='maximize')

        # Выбор функции цели
        if model_name == "XGBoost":
            func = lambda trial: objective_xgb(trial, X, y)
        elif model_name == "CatBoost":
            func = lambda trial: objective_cat(trial, X, y)
        else:
            func = lambda trial: objective_ridge(trial, X, y)

        study.optimize(func, n_trials=n_trials, callbacks=[callback])

        # --- Результаты ---
        st.subheader("Результаты оптимизации")
        st.metric("Лучший R2 Score (на валидации)", f"{study.best_value:.5f}")

        st.write("**Лучшие гиперпараметры:**")
        st.json(study.best_params)

        # --- ИСПРАВЛЕННАЯ Визуализация истории оптимизации ---
        st.write("История оптимизации:")
        # Не создаем fig, ax вручную! Функция возвращает Axes
        ax_history = optuna.visualization.matplotlib.plot_optimization_history(study)
        st.pyplot(ax_history.figure)  # Берем figure из возвращенного Axes

        # --- ГРАФИК ПРЕДСКАЗАНИЙ VS РЕАЛЬНОСТЬ ---
        st.subheader("Предсказания vs Реальность (Validation Set)")

        # 1. Обучаем финальную модель с лучшими параметрами
        best_params = study.best_params

        if model_name == "XGBoost":
            final_model = XGBRegressor(
                **best_params,
                n_estimators=1000,
                early_stopping_rounds=50,
                random_state=42,
                n_jobs=-1
            )
            # Обучаем на train, валидируемся на val
            final_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        elif model_name == "CatBoost":
            final_model = CatBoostRegressor(
                **best_params,
                iterations=1000,
                random_state=42,
                verbose=False
            )
            final_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)

        else:  # Ridge
            final_model = Ridge(**best_params, random_state=42)
            final_model.fit(X_train, y_train)

        # 2. Предсказываем на ВАЛИДАЦИИ (X_val)
        y_pred = final_model.predict(X_val)

        # 3. Рисуем график
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_val, y_pred, alpha=0.3, c='red')

        # Линия идеала (x=y)
        # Берем границы по валидации
        lims = [y_val.min(), y_val.max()]
        ax.plot(lims, lims, 'b--', lw=2, label='Идеальное предсказание')

        ax.set_xlabel('Реальные значения (Log Scale)')
        ax.set_ylabel('Предсказанные значения (Log Scale)')
        ax.set_title(f'Предсказания на Validation для {model_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # --- ИНТЕРПРЕТИРУЕМОСТЬ (SHAP) ---
        st.subheader("Интерпретируемость модели (SHAP)")
        st.write("""
        SHAP показывает вклад каждого признака в предсказание. 
        Красный цвет = увеличивает выплату. Синий = уменьшает.
        """)

        try:
            # 1. Берем выборку для быстрого расчета (например, 1000 строк)
            X_sample = X_val.sample(n=min(1000, len(X_val)), random_state=42)

            # 2. Создаем объяснитель (Explainer)
            # Для деревьев используем TreeExplainer (быстро), для линейных - LinearExplainer
            if model_name in ["XGBoost", "CatBoost"]:
                explainer = shap.TreeExplainer(final_model)
            else:  # Ridge
                # Для линейных моделей нужно передать маску (данные), но для простоты можно Universal
                explainer = shap.Explainer(final_model, X_sample)

            # Считаем SHAP values
            with st.spinner("Расчет SHAP значений..."):
                shap_values = explainer(X_sample)

            # --- График 1: Summary Plot (Общая важность) ---
            st.write("**Общая важность признаков (Summary Plot):**")

            # shap.plots возвращает объект Figure/Axes, нам нужно его захватить для Streamlit
            # Используем matplotlib backend
            fig_shap, ax_shap = plt.subplots(figsize=(10, 8))
            shap.plots.beeswarm(shap_values, show=False)  # show=False важно, чтобы не вылезло в отдельном окне
            st.pyplot(fig_shap)
            plt.clf()  # Очищаем память

            # --- График 2: Waterfall Plot (Для конкретного примера) ---
            st.write("**Анализ конкретного случая (Waterfall Plot):**")
            st.write("Как модель приняла решение для одного случайного клиента.")

            # Берем индекс первой строки из выборки
            sample_ind = 0

            # Рисуем waterfall. Это сложнее, так как shap иногда требует объект Explanation
            # shap_values уже является объектом Explanation

            fig_waterfall, ax_waterfall = plt.subplots()
            # Для waterfall plot лучше использовать встроенную функцию shap
            shap.plots.waterfall(shap_values[sample_ind], show=False)
            st.pyplot(fig_waterfall)
            plt.clf()

        except ImportError:
            st.error("Библиотека shap не установлена. Выполните: pip install shap")
        except Exception as e:
            st.error(f"Ошибка при построении SHAP графиков: {e}")


if __name__ == "__main__":
    optimization_page()
