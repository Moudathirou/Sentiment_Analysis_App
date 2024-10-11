# Application d'Analyse de Sentiments

![Capture d’écran 2024-10-11 154524](https://github.com/user-attachments/assets/c605c9c3-003e-4e49-83ff-4ac2c8fabcf4)

## Description
Ce dépôt contient une application d'analyse de sentiments qui utilise des modèles hybrides combinant LSTM, CNN et GRU, ainsi qu'un modèle de langage (LLM) fourni par Groq pour l'analyse de sentiments. L'application permet de sélectionner différents modèles pour faire des prédictions. Elle utilise des techniques d'ensemble comme le stacking et le voting, atteignant des performances impressionnantes, proches de celles des transformers sur le jeu de données utilisé.

## Fonctionnalités
- **Modèles hybrides** : Combinaison de LSTM, CNN et GRU pour une analyse de sentiments robuste.
- **LLM Groq** : Utilisation d'un modèle de langage large intégré à Groq pour l'analyse de sentiments.
- **Sélection de modèle** : Choix parmi plusieurs modèles pour effectuer des prédictions, y compris le modèle Groq.
- **Techniques d'ensemble** : Utilisation du stacking et du voting pour améliorer les performances.
- **Performances comparables aux transformers** : Stacking et voting atteignent des résultats proches de ceux des transformers.
- **Optimisé avec Groq** : Utilisation de Groq pour accélérer les inférences et les performances du LLM.

## Prérequis
- Python 3.12.4
- Groq SDK (pour l'accélération matérielle et l'utilisation du LLM)
- TensorFlow/PyTorch (selon les implémentations des modèles)
- Autres dépendances (listées dans `requirements.txt`)

## Installation

```bash
# Clonez le dépôt
git clone https://github.com/Moudathirou/Sentiment_Analysis_App.git
cd Sentiment_Analysis_App

# Créez un enviromment virtuel
python -m venv myenv

# Installez les dépendances
pip install -r requirements.txt
