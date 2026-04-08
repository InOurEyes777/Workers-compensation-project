import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml


def deph_analysis():
    st.title("Детальный анализ исходных данных")

    # 1. Блок загрузки данных
    if st.button("Загрузить данные"):
        with st.spinner("Загрузка данных..."):
            data = fetch_openml(data_id=42876, as_frame=True, parser='auto')
            df = data.frame

            # ВАЖНО: Сразу преобразуем дату и сохраняем УЖЕ обработанный датафрейм
            df['DateTimeOfAccident'] = pd.to_datetime(df['DateTimeOfAccident'])

            st.session_state['df'] = df
            st.success("Данные успешно загружены!")

    # 2. Блок анализа (выполняется, если данные есть в сессии)
    if 'df' in st.session_state:
        df = st.session_state['df']  # Достаем сохраненные данные

        # --- ЧАСТЬ 1: Ваш старый график по годам (немного улучшенный) ---
        st.subheader("Общая динамика по годам")

        # Создаем копию для манипуляций, чтобы не менять исходный df в сессии
        data = df.copy()
        data['Year'] = data['DateTimeOfAccident'].dt.year

        mean_cost_by_year = data.groupby('Year')['UltimateIncurredClaimCost'].mean().reset_index()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(mean_cost_by_year['Year'], mean_cost_by_year['UltimateIncurredClaimCost'],
                'r', marker='o', lw=2, markersize=6)

        ax.set_xlabel('Год')
        ax.set_ylabel('Средний размер выплаты в $')
        ax.set_title('Динамика средних страховых выплат по годам')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        st.subheader("Средние выплаты по годам")
        st.dataframe(mean_cost_by_year.style.format({'UltimateIncurredClaimCost': '{:,.0f}'}))

        # --- ЧАСТЬ 2: Новый функционал (Детализация по месяцам) ---
        st.divider()  # Разделительная линия
        st.subheader("Детализация по месяцам")

        # Шаг 1: Получаем список уникальных лет, сортируем их
        available_years = sorted(data['Year'].unique())

        # Шаг 2: Создаем selectbox
        # st.selectbox возвращает выбранное значение, которое мы сохраняем в переменную
        selected_year = st.selectbox(
            "Выберите год для просмотра помесячной динамики:",
            available_years
        )

        # Шаг 3: Фильтруем данные по выбранному году
        # Мы берем исходный df и оставляем только строки с нужным годом
        df_filtered = df[df['DateTimeOfAccident'].dt.year == selected_year].copy()

        # Шаг 4: Извлекаем месяц для группировки
        # dt.month возвращает число от 1 до 12
        df_filtered['Month'] = df_filtered['DateTimeOfAccident'].dt.month

        # Шаг 5: Считаем среднюю выплату по месяцам внутри отфильтрованного года
        monthly_stats = df_filtered.groupby('Month')['UltimateIncurredClaimCost'].mean().reset_index()

        # Шаг 6: Строим график
        fig2, ax2 = plt.subplots(figsize=(10, 5))

        # Столбчатая диаграмма (bar chart) лучше подходит для месяцев, чем линия
        sns.barplot(x='Month', y='UltimateIncurredClaimCost', data=monthly_stats,
                    ax=ax2, palette='viridis', hue='Month', legend=False)

        # Можно также наложить линию тренда для наглядности
        sns.lineplot(x=range(len(monthly_stats)), y='UltimateIncurredClaimCost', data=monthly_stats,
                     ax=ax2, color='red', marker='o', linewidth=2, label='Тренд')

        ax2.set_title(f'Средняя стоимость выплат в {selected_year} году')
        ax2.set_xlabel('Месяц')
        ax2.set_ylabel('Средняя сумма ($)')
        ax2.grid(True, axis='y', alpha=0.3)  # Сетка только по Y

        # Добавляем подписи значений над столбцами
        for index, row in monthly_stats.iterrows():
            ax2.text(row['Month'] - 1, row['UltimateIncurredClaimCost'] + 50,
                     f'{row["UltimateIncurredClaimCost"]:,.0f}',
                     color='black', ha="center", fontsize=9)

        st.pyplot(fig2)

        # Вывод таблицы с цифрами для точности
        st.write(f"**Таблица средних выплат за {selected_year} год:**")
        st.dataframe(monthly_stats.style.format({'UltimateIncurredClaimCost': '{:,.0f}'}))

    else:
        st.info("Нажмите кнопку выше, чтобы загрузить данные.")

    # --- ЧАСТЬ 3: Анализ распределения и выбросов ---
    st.divider()
    st.subheader("Анализ распределения целевой переменной")

    target_col = 'UltimateIncurredClaimCost'

    # В страховании данные часто имеют тяжелый хвост, поэтому дадим возможность посмотреть логарифм
    use_log = st.checkbox("Применить логарифм к данным", value=False,
                          help="Полезно для данных с длинным хвостом, чтобы разглядеть структуру распределения.")

    # Подготовка данных для графика
    plot_data = df[target_col].copy()
    if use_log:
        plot_data = np.log1p(plot_data)  # log1p(x) = log(1+x), чтобы избежать ошибок с нулями

    # 1. Гистограмма
    fig3, ax3 = plt.subplots(figsize=(10, 6))

    # Строим гистограмму с кривой плотности (KDE)
    sns.histplot(plot_data, kde=True, ax=ax3, bins=50, color='skyblue')

    title_text = f'Распределение {"log(" + target_col + ")" if use_log else target_col}'
    ax3.set_title(title_text, fontsize=14)
    ax3.set_xlabel('Значение')
    ax3.set_ylabel('Частота')

    st.pyplot(fig3)

    # 2. Проверка на нормальность
    st.write("**Проверка гипотезы нормальности:**")

    # Для теста берем случайную выборку, так как на 100к строк тест будет слишком строгим/медленным
    sample_size = min(5000, len(df))
    sample_data = df[target_col].sample(n=sample_size, random_state=42)
    if use_log:
        sample_data = np.log1p(sample_data)

    from scipy.stats import normaltest, zscore
    stat, p_value = normaltest(sample_data)

    st.write(f"Использована выборка размером {sample_size} записей.")
    st.write(f"P-value: {p_value:.5e}")
    if p_value > 0.05:
        st.success("Распределение похоже на нормальное (p > 0.05, нулевая гипотеза не отвергается).")
    else:
        st.warning("Распределение НЕ является нормальным (p < 0.05). Это типично для финансовых данных.")


    # 3. Анализ выбросов (Boxplot и статистика)
    st.subheader("Анализ выбросов")
    st.subheader("Расчет границ выбросов по методу Тьюки (1.5 * IQR)")

    col1, col2 = st.columns(2)

    with col1:
        # Boxplot
        fig4, ax4 = plt.subplots(figsize=(6, 6))
        sns.boxplot(y=df[target_col], ax=ax4, color='lightgreen')
        ax4.set_title('Boxplot стоимости выплат')
        st.pyplot(fig4)

    with col2:
        # Расчет границ выбросов по методу Тьюки (1.5 * IQR)
        Q1 = df[target_col].quantile(0.25)
        Q3 = df[target_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Подсчет выбросов
        outliers_count = ((df[target_col] < lower_bound) | (df[target_col] > upper_bound)).sum()
        percent = (outliers_count / len(df)) * 100

        st.metric("Нижняя граница", f"${lower_bound:,.0f}")
        st.metric("Верхняя граница", f"${upper_bound:,.0f}")
        st.metric("Количество выбросов", f"{outliers_count:,}")
        st.metric("Процент выбросов", f"{percent:.2f}%")

        st.info("Выбросы определены методом межквартильного размаха (IQR). "
                "Значения за пределами 1.5 * IQR считаются аномальными.")

    st.divider()
    st.subheader("Поиск выбросов с помощью Z-оценки")

    st.write("""
    Z-оценка показывает, на сколько стандартных отклонений значение отклоняется от среднего.
    Обычно порог равен 3 (значение отличается от среднего более чем на 3 сигмы).
    """)

    # Добавим слайдер для интерактивного выбора порога
    threshold = st.slider("Выберите порог Z-оценки:", min_value=2.0, max_value=5.0, value=3.0, step=0.1)

    # 1. Считаем Z-оценку
    # zscore = (x - mean) / std
    # Мы используем copy(), чтобы не менять исходный df в сессии (избежать предупреждений)
    df_z = df.copy()
    df_z['z_score'] = zscore(df_z[target_col])

    # 2. Определяем выбросы
    # np.abs берет модуль числа. Мы ищем значения > threshold (обычно 3)
    # Но так как стоимость не бывает отрицательной, нас интересуют только положительные выбросы (справа)
    df_z['is_outlier'] = df_z['z_score'] > threshold

    outliers = df_z[df_z['is_outlier']]
    clean_data = df_z[~df_z['is_outlier']]

    # Вывод метрик
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Всего выбросов", f"{len(outliers):,}")
    with col2:
        percent = (len(outliers) / len(df_z)) * 100
        st.metric("Процент выбросов", f"{percent:.2f}%")
    with col3:
        # Граница отсечения (среднее + 3 * стд)
        cutoff_value = df_z[target_col].mean() + threshold * df_z[target_col].std()
        st.metric("Граница отсечения ($)", f"{cutoff_value:,.0f}")

    # 3. Визуализация
    fig5, ax5 = plt.subplots(figsize=(10, 6))

    # Гистограмма всех данных (серый)
    sns.histplot(np.log1p(df_z[target_col]), bins=100, ax=ax5, color='gray', label='Все данные', alpha=0.3)

    # Гистограмма выбросов (красный) - рисуем поверх, чтобы выделить их
    sns.histplot(outliers[target_col], bins=100, ax=ax5, color='red', label='Выбросы')

    # Рисуем вертикальную линию границы отсечения
    ax5.axvline(cutoff_value, color='blue', linestyle='--', lw=2, label=f'Граница Z={threshold}')

    ax5.set_title('Распределение с выделением выбросов')
    ax5.legend()
    st.pyplot(fig5)

    # Вывод примеров выбросов
    st.write("**Примеры выявленных выбросов:**")
    st.dataframe(outliers[[target_col, 'z_score']].sort_values(by=target_col, ascending=False).head(10).style.format(
        {target_col: '{:,.0f}', 'z_score': '{:.2f}'}))

    # --- ЧАСТЬ 5: Корреляционный анализ ---
    st.divider()
    st.subheader("Корреляционный анализ")

    # 1. Подготовка данных: оставляем только числовые колонки
    # Correlation matrix работает только с числами
    numeric_df = df.select_dtypes(include=[np.number])

    # Удаляем колонки, которые не несут смысла для корреляции (если есть ID или коды)
    # В этом датасете обычно нет ID, но на всякий случай проверим размерность
    st.write(f"Размер матрицы корреляций: {numeric_df.shape[1]} признаков.")

    # 2. Построение Heatmap
    st.write("**Тепловая карта корреляций (Heatmap):**")

    fig6, ax6 = plt.subplots(figsize=(12, 8))

    # Вычисляем матрицу корреляций Пирсона
    corr_matrix = numeric_df.corr()

    # Рисуем тепловую карту
    sns.heatmap(corr_matrix,
                annot=True,  # Показывать числа
                fmt=".2f",  # Формат чисел (2 знака после точки)
                cmap='coolwarm',  # Цветовая схема (синий - низкая, красный - высокая)
                ax=ax6,
                linewidths=0.5)  # Линии между клетками

    ax6.set_title('Матрица корреляций признаков', fontsize=14)
    st.pyplot(fig6)

    # Вывод топ-корреляций с целевой переменной (чтобы не искать глазами)
    st.write("**Топ-5 признаков, коррелирующих с целевой переменной:**")
    # Сортируем по модулю корреляции, берем топ-5 (исключая саму целевую переменную)
    target_corr = corr_matrix[target_col].drop(target_col).abs().sort_values(ascending=False).head(5)
    st.dataframe(target_corr.to_frame().style.format({target_col: "{:.3f}"}))

    # 3. Scatter Plots для важных пар признаков
    st.divider()
    st.subheader("Анализ зависимостей")
    st.write("Выберите два признака для просмотра диаграммы рассеяния.")

    # Получаем список числовых колонок для выбора
    cols_for_scatter = numeric_df.columns.tolist()

    # Находим индекс целевой переменной, чтобы выбрать её по умолчанию для оси Y
    default_y_index = cols_for_scatter.index(target_col) if target_col in cols_for_scatter else 0

    col_x, col_y = st.columns(2)
    with col_x:
        # Выбор признака для оси X
        x_feature = st.selectbox("Признак X:", cols_for_scatter, index=0)
    with col_y:
        # Выбор признака для оси Y
        y_feature = st.selectbox("Признак Y:", cols_for_scatter, index=default_y_index)

    # Построение графика
    fig7, ax7 = plt.subplots(figsize=(10, 6))

    # Используем scatterplot
    # alpha=0.3 делает точки полупрозрачными, чтобы видеть плотность (важно для больших данных)
    sns.scatterplot(data=df, x=x_feature, y=y_feature, alpha=0.3, ax=ax7, color='teal')

    # Добавляем линию тренда (линию регрессии)
    # ci=None убирает доверительный интервал (заливку вокруг линии) для чистоты
    sns.regplot(data=df, x=x_feature, y=y_feature, scatter=False, ax=ax7, color='red', line_kws={'linewidth': 2})

    ax7.set_title(f'Зависимость {y_feature} от {x_feature}', fontsize=14)
    ax7.grid(True, alpha=0.3)
    st.pyplot(fig7)

    # Дополнительный инсайт: коэффициент корреляции Пирсона для выбранной пары
    corr_val = df[[x_feature, y_feature]].corr().iloc[0, 1]
    st.info(f"Коэффициент корреляции Пирсона для выбранных признаков: **{corr_val:.3f}**")



    # --- ЧАСТЬ 6: Анализ категориальных переменных ---
    st.divider()
    st.subheader("Анализ категориальных переменных")

    # Для анализа нам нужны категориальные колонки. Проверим, что они есть
    cat_cols = ['Gender', 'MaritalStatus', 'PartTimeFullTime']

    # Проверка, есть ли эти колонки в загруженных данных
    existing_cat_cols = [col for col in cat_cols if col in df.columns]

    if existing_cat_cols:
        # 1. Стандартные категориальные признаки (Пол, Статус, Занятость)
        st.write("**Распределение стоимости по категориям:**")

        # Создаем вкладки для удобства, чтобы не скроллить страницу вниз
        tabs = st.tabs(existing_cat_cols)

        for i, col in enumerate(existing_cat_cols):
            with tabs[i]:
                fig8, ax8 = plt.subplots(figsize=(10, 6))

                # Строим боксплот
                sns.boxplot(data=df, x=col, y=target_col, ax=ax8, palette='Set2', hue=col, legend=False)

                # Ограничим ось Y для наглядности (убираем сверхвыбросы, чтобы увидеть "ящики")
                # Берем 95 перцентиль как лимит, чтобы выбросы не сжимали график
                ylim = df[target_col].quantile(0.95)
                ax8.set_ylim(0, ylim * 1.1)

                ax8.set_title(f'Стоимость выплат в зависимости от {col}')
                ax8.set_xlabel(col)
                ax8.set_ylabel('Стоимость выплаты ($)')
                ax8.grid(True, axis='y', alpha=0.3)
                st.pyplot(fig8)

                # Выведем средние значения по группам для точности
                st.write(f"**Средние значения по группам {col}:**")
                st.dataframe(df.groupby(col)[target_col].agg(['mean', 'count']).style.format({'mean': '${:,.0f}'}))

    # 2. Анализ по типам травм (используем ClaimDescription)
    if 'ClaimDescription' in df.columns:
        st.divider()
        st.subheader("Анализ стоимости по типам травм (ClaimDescription)")

        st.write("""
        В датасете нет отдельной колонки "Тип травмы", но есть текстовое описание.
        Ниже представлены боксплоты для **10 самых частых типов описаний**.
        """)

        # Находим топ-10 самых частых описаний
        top_10_claims = df['ClaimDescription'].value_counts().head(10).index.tolist()

        # Фильтруем датасет, оставляя только эти 10 типов
        df_top_claims = df[df['ClaimDescription'].isin(top_10_claims)]

        fig9, ax9 = plt.subplots(figsize=(14, 7))

        # Строим боксплот. Поворачиваем подписи оси X, чтобы они не накладывались
        sns.boxplot(data=df_top_claims, x='ClaimDescription', y=target_col, ax=ax9, palette='viridis',
                    hue='ClaimDescription', legend=False)

        # Настройка внешнего вида
        ax9.set_title('Стоимость выплат для ТОП-10 частых описаний травм', fontsize=14)
        ax9.set_xlabel('Описание травмы')
        ax9.set_ylabel('Стоимость выплаты ($)')

        # Поворот подписей на 45 градусов
        plt.xticks(rotation=45, ha='right')

        # Снова ограничиваем ось Y для наглядности
        ylim = df[target_col].quantile(0.95)
        ax9.set_ylim(0, ylim * 1.1)

        ax9.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()  # Чтобы подписи не обрезались при сохранении
        st.pyplot(fig9)

deph_analysis()
