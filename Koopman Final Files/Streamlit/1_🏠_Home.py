import streamlit as st

st.set_page_config(
    page_title="Learning Koopman Operator with Deep Neural Networks",
    page_icon="📚"
)

st.title("Page Principale")

st.header("Introduction", divider= True)

st.write("""
Ce projet a été développé dans le cadre du cours "Projet de Deep Learning" par Caudard Joris, Zheng Vicky, Andriarimanana Sylviane.

L'objectif de ce projet est une implémentation deréseaux de neurones capables d'identifier l'opérateur de Koopman dans le cadre des systèmes dynamiques.
""")

