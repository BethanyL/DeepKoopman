import streamlit as st
import preprocessingData

df, params = preprocessingData.preprocessDiscreteSpectrum(dataPath="data/DiscreteSpectrumExample_train1_x.csv")

bestParams = params

st.title("Théorie de l'opérateur de Koopman et Implémentation en Réseau de neurones")

col1, col2 = st.columns(2)

with col1:
    optimal = st.radio("Select parameters to use",
             ["Optimal", "Custom"],
             index=0,
             captions=["Use Optimal parameters & saved network", "Train a new network with custom parameters"])

with col2:
    if optimal == 'Custom':  
        params["numShifts"] = st.slider("numShifts", 0, 50, bestParams["numShifts"])
        params["reconLam"] = st.slider("reconLam", 0, 10, 1)
        params["L1Lam"] = st.slider("L1Lam", 0, 10, 1)
        params["L2Lam"] = st.slider("L2Lam", 0, 10, 1)
        params["hiddenSize"] = st.slider("hiddenSize", 0, 10, 1)
        params["hiddenLayer"] = st.slider("hiddenLayer", 0, 10, 1)
        params["latentSize"] = st.slider("latentSize", 0, 10, 1)
    else:
        params = bestParams


st.write("Here are the parameters of the model : ")

st.write(params)