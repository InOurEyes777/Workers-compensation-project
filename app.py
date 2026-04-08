import streamlit as st

pages = [
    st.Page("analysis_and_model.py", title="Анализ и модель"),
    st.Page("presentation.py", title="Презентация"),
    st.Page("deph_analysis.py", title="Детальный анализ"),
    st.Page("optimization.py", title="Оптимизация моделей")]

# Отображение навигации
current_page = st.navigation(pages, position="sidebar", expanded=True)
current_page.run()
