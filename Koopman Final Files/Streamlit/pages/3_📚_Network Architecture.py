import streamlit as st

st.title("Network architecture")

tab1, tab2, tab3 = st.tabs(["Auto-Encoder", "Architecture Simple", "Architecture Générale"])

with tab1:
    st.write("L'architecture générale du réseau proposé pour l'identification de l'opérateur de Koopman repose sur l'érchitecture de l'auto-encoder simple.")

    #st.image() ##TODO: Import Auto-encoder picture

    st.write("A cette structure simple, on rajoute un avancement linéaire dans le temps permettant l'identification de l'opérateur de Koopman")

with tab2:
    st.header("Structure Basique")

    #st.image() ##TODO: Import Simple Koopman Network picture

    st.write("L'implémentation de ce réseau simple de Koopman a été faite en utilisant les focntionnalités du module python Pytorch.")

    st.code("""class SimpleKoopmanNeuralNetwork(nn.Module):
    def __init__(self, params):
        super(SimpleKoopmanNeuralNetwork, self).__init__()
        
        self.params = params

        inputSize, hiddenSize, hiddenLayer, latentSize = self.params['inputSize'], self.params['hiddenSize'], self.params['hiddenLayer'], self.params['latentSize']

        #Input layer of Encoder
        encoderLayers = [nn.Linear(inputSize, hiddenSize), nn.ReLU()]

        ###Define Encoder Layer
        for _ in range(hiddenLayer):
            encoderLayers.append(nn.Linear(hiddenSize, hiddenSize))
            encoderLayers.append(nn.ReLU())

        #Output layer of Encoder
        encoderLayers.append(nn.Linear(hiddenSize, latentSize))

        #Creating the Encoder Network
        self.Encoder = nn.Sequential(*encoderLayers)

        #Input layer of Decoder
        decoderLayers = [nn.Linear(latentSize, hiddenSize), nn.ReLU()]

        ###Define Decoder Layer
        for _ in range(hiddenLayer):
            decoderLayers.append(nn.Linear(hiddenSize, hiddenSize))
            decoderLayers.append(nn.ReLU())

        #Output layer of Decoder
        decoderLayers.append(nn.Linear(hiddenSize, inputSize))

        #Creating the Decoder Network
        self.Decoder = nn.Sequential(*decoderLayers)

        #Simple Koopman Auto-Encoder (Without Auxiliary Network)
        self.K = nn.Linear(latentSize, latentSize, bias=False)

    def forward(self, initialInput):
        #Take as input a 2 dimension tensor (initial State)

        #get the first encoded state (y1)
        encodedInitialInput = self.Encoder(initialInput)

        #First element of the latent trajectory is encoded input
        latentTrajectoryPrediction = [encodedInitialInput]

        #Alongside the trajectory, we multiply by the Koopman operator
        for m in range(49):
            latentTrajectoryPrediction.append(self.K(latentTrajectoryPrediction[-1]))

        #Decoding the trajectory
        trajectoryPrediction = [self.Decoder(latentState) for latentState in latentTrajectoryPrediction]
        
        #We get both the encoded trajectory and the decoded trajectory
        return trajectoryPrediction, latentTrajectoryPrediction""")
    
    st.write("Ce réseau prend en input une condition initiale du système dynamique étudié, et permet de prédire en sortie deux objets distincts : D'une part, la trajectoire à partir de cet état initial dans l'espace latent, d'autre part la trajectoire prédite dans l'espace initial.")

#with tab3:
    #TODO: Add description of the general architecture