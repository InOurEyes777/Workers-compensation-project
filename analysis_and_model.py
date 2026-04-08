import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
from xgboost import XGBRegressor
from sklearn.feature_extraction.text import TfidfVectorizer


def analysis_and_model_page():
    st.title("Прогнозирование стоимости страховых выплат")

    if st.button("Загрузить данные"):
        with st.spinner("Загрузка данных..."):
            data = fetch_openml(data_id=42876, as_frame=True, parser='auto')
            df = data.frame
            st.session_state['df'] = df
            st.success("Данные успешно загружены!")

    # Если данные уже загружены ранее
    if 'df' in st.session_state:
        df = st.session_state['df']

        st.subheader("Просмотр данных")
        st.write(df.head())

        st.subheader("Статистика")
        st.write(df.describe())

        # --- 2. ПРЕДОБРАБОТКА И ОБУЧЕНИЕ ---
        if st.button("Обучить модели"):
            with st.spinner("Предобработка и обучение... Это может занять время"):
                # 2.1 Копия данных
                data = df.copy()

                # 2.2 Работа с датами
                data['DateTimeOfAccident'] = pd.to_datetime(data['DateTimeOfAccident'])
                data['DateReported'] = pd.to_datetime(data['DateReported'])
                data['AccidentMonth'] = data['DateTimeOfAccident'].dt.month
                data['AccidentDayOfWeek'] = data['DateTimeOfAccident'].dt.dayofweek
                data['ReportingDelay'] = (data['DateReported'] - data['DateTimeOfAccident']).dt.days

                # --- Feature Engineering ---
                # 1. Взаимодействия
                data['Age_WeeklyPay_Interaction'] = data['Age'] * data['WeeklyPay']
                # 2. Отношения
                data['Estimate_to_Pay_Ratio'] = data['InitialCaseEstimate'] / (data['WeeklyPay'] + 1)
                # 3. Бинарные признаки
                data['Has_Dependents'] = (data['DependentChildren'] > 0).astype(int)
                # Трансформация текста в векторы
                tfidf = TfidfVectorizer(max_features=50)
                tfidf_matrix = tfidf.fit_transform(data['ClaimDescription'])
                tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())
                data = pd.concat([data, tfidf_df], axis=1)
                data = data.drop(columns=['DateTimeOfAccident', 'DateReported', 'ClaimDescription'])

                # 2.4 Кодирование категорий
                label_encoders = {}
                categorical_columns = ['Gender', 'MaritalStatus', 'PartTimeFullTime']
                for col in categorical_columns:
                    le = LabelEncoder()
                    data[col] = le.fit_transform(data[col])
                    label_encoders[col] = le
                # Сохраняем энкодеры в сессию (понадобятся для предсказания)
                st.session_state['encoders'] = label_encoders
                numerical_features = ['Age', 'DependentChildren', 'DependentsOther', 'WeeklyPay',
                                      'HoursWorkedPerWeek', 'DaysWorkedPerWeek', 'InitialCaseEstimate',
                                      'AccidentMonth', 'AccidentDayOfWeek', 'ReportingDelay',
                                      'Age_WeeklyPay_Interaction', 'Estimate_to_Pay_Ratio']
                scaler = StandardScaler()
                data[numerical_features] = scaler.fit_transform(data[numerical_features])
                st.session_state['scaler'] = scaler

                # 2.5 Разделение данных
                X = data.drop(columns=['UltimateIncurredClaimCost'])
                y = data['UltimateIncurredClaimCost']

                # Сохраняем список признаков (важно для порядка колонок!)
                st.session_state['feature_names'] = X.columns.tolist()

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Преобразуем целевую переменную
                y_train = np.log1p(y_train)
                y_test = np.log1p(y_test)

                # 2.7 Обучение
                # Linear Regression
                lin_reg = LinearRegression()
                lin_reg.fit(X_train, y_train)
                st.session_state['model_lin'] = lin_reg

                rf_reg = RandomForestRegressor(n_estimators=50, random_state=42)
                rf_reg.fit(X_train, y_train)
                st.session_state['model_rf'] = rf_reg

                xgb_reg = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
                xgb_reg.fit(X_train, y_train)
                st.session_state['model_xgb'] = xgb_reg

                ridge_reg = Ridge(alpha=1, random_state=42)
                ridge_reg.fit(X_train, y_train)
                st.session_state['model_ridge'] = ridge_reg

                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test

            st.success("Модели обучены!")

        # --- 3. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ ---
        if 'model_rf' in st.session_state:
            st.header("Результаты обучения")

            model_choice = st.selectbox("Выберите модель для оценки", ["Random Forest", "Linear Regression",
                                                                       "Ridge Regression", "XGBoost"])
            if model_choice == "Random Forest":
                model = st.session_state['model_rf']
                X_test_use = st.session_state['X_test']
            elif model_choice == "Ridge Regression":
                model = st.session_state['model_ridge']
                X_test_use = st.session_state['X_test']
            elif model_choice == "XGBoost":
                model = st.session_state['model_xgb']
                X_test_use = st.session_state['X_test']
            else:
                model = st.session_state['model_lin']
                X_test_use = st.session_state['X_test']

            y_test = st.session_state['y_test']
            y_pred = model.predict(X_test_use)

            # Метрики
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            col1, col2, col3 = st.columns(3)
            col1.metric("MAE", f"${mae:,.2f}")
            col2.metric("RMSE", f"${rmse:,.2f}")
            col3.metric("R² Score", f"{r2:.4f}")

            # График
            st.subheader("Предсказания vs Реальность")
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, alpha=0.3, c='red')
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'b--', lw=2)
            ax.set_xlabel('Реальные значения')
            ax.set_ylabel('Предсказанные значения')
            st.pyplot(fig)

            if model_choice == "Random Forest":
                st.subheader("Важность признаков")
                feature_names = st.session_state['feature_names']
                importance = model.feature_importances_
                feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance}).sort_values(
                    'Importance', ascending=False)

                fig2, ax2 = plt.subplots()
                sns.barplot(x='Importance', y='Feature', data=feature_df.head(10), ax=ax2)
                st.pyplot(fig2)
            elif model_choice == "XGBoost":
                st.subheader("Важность признаков")
                feature_names = st.session_state['feature_names']
                importance = model.feature_importances_
                feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance}).sort_values(
                    'Importance', ascending=False)

                fig2, ax2 = plt.subplots()
                sns.barplot(x='Importance', y='Feature', data=feature_df.head(10), ax=ax2)
                st.pyplot(fig2)

        # --- 4. ИНТЕРФЕЙС ПРЕДСКАЗАНИЯ ---
        st.header("Предсказание стоимости возмещения для нового случая")
        st.info("Введите параметры ниже. Остальные параметры будут заполнены средними значениями из датасета.")

        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Возраст", min_value=13, max_value=76, value=35)
                gender = st.selectbox("Пол", ["M", "F"])
                weekly_pay = st.number_input("Еженедельная зарплата ($)", min_value=0, value=500)

            with col2:
                initial_estimate = st.number_input("Начальная оценка ($)", min_value=0, value=5000)
                dependents = st.number_input("Кол-во детей", min_value=0, value=0)
                hours_worked = st.number_input("Часов в неделю", min_value=0, value=40)

            submit_button = st.form_submit_button("Предсказать")

            if submit_button:

                # 1. Создаем словарь с базовыми введенными данными
                input_data = {
                    'Age': age,
                    'DependentChildren': dependents,
                    'DependentsOther': 0,
                    'WeeklyPay': weekly_pay,
                    'HoursWorkedPerWeek': hours_worked,
                    'DaysWorkedPerWeek': 5,
                    'InitialCaseEstimate': initial_estimate,
                    'AccidentMonth': 6,
                    'AccidentDayOfWeek': 2,
                    'ReportingDelay': 5
                }

                # 2. Создаем DataFrame
                input_df = pd.DataFrame([input_data])

                # 3. Кодируем категориальные признаки
                # Используем энкодеры из сессии
                le_gender = st.session_state['encoders']['Gender']
                input_df['Gender'] = le_gender.transform([gender])[0]

                input_df['MaritalStatus'] = 0  # Значение по умолчанию
                input_df['PartTimeFullTime'] = 0  # Значение по умолчанию

                # --- ИСПРАВЛЕНИЕ: Feature Engineering ---
                # Мы должны пересчитать эти признаки, так как модель их ждет!
                input_df['Age_WeeklyPay_Interaction'] = input_df['Age'] * input_df['WeeklyPay']
                input_df['Estimate_to_Pay_Ratio'] = input_df['InitialCaseEstimate'] / (input_df['WeeklyPay'] + 1)
                input_df['Has_Dependents'] = (input_df['DependentChildren'] > 0).astype(int)

                # 4. Масштабирование числовых признаков
                # ВАЖНО: Список должен совпадать с тем, что был при обучении
                numerical_features = ['Age', 'DependentChildren', 'DependentsOther', 'WeeklyPay',
                                      'HoursWorkedPerWeek', 'DaysWorkedPerWeek', 'InitialCaseEstimate',
                                      'AccidentMonth', 'AccidentDayOfWeek', 'ReportingDelay',
                                      'Age_WeeklyPay_Interaction', 'Estimate_to_Pay_Ratio']

                scaler = st.session_state['scaler']

                # Scale (преобразуем массив обратно в DataFrame для удобства)
                input_numerical = scaler.transform(input_df[numerical_features])
                input_numerical_df = pd.DataFrame(input_numerical, columns=numerical_features)

                # 5. Объединение данных
                # Берем категориальные и бинарные признаки из input_df, числовые - из масштабированных
                categorical_features = ['Gender', 'MaritalStatus', 'PartTimeFullTime', 'Has_Dependents']

                # Объединяем (axis=1 склеивает колонки)
                input_processed = pd.concat([input_numerical_df, input_df[categorical_features]], axis=1)

                # 6. Выравнивание колонок (Alignment)
                # Модель ждет ровно тот же набор колонок, что и при обучении, включая TF-IDF слова
                feature_names = st.session_state['feature_names']

                # Добавляем недостающие колонки (например, слова из TF-IDF) и заполняем их нулями
                missing_cols = set(feature_names) - set(input_processed.columns)
                for c in missing_cols:
                    input_processed[c] = 0.0

                # Удаляем лишние, если есть, и упорядочиваем колонки в точном порядке
                input_processed = input_processed[feature_names]

                # 7. Предсказание
                model = st.session_state['model_xgb']
                prediction_log = model.predict(input_processed)[0]

                # Обратное преобразование логарифма
                prediction_original = np.expm1(prediction_log)

                st.success(f"Примерная итоговая стоимость выплаты: **${prediction_original:,.2f}**")

analysis_and_model_page()
