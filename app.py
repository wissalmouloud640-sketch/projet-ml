import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

st.set_page_config(
    page_title="Prédiction de réussite",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Prédiction de réussite d'un étudiant - Version Améliorée")
st.write("Application avancée de Machine Learning avec Streamlit")

# 1. Création des données d'exemple (avec ajout des absences justifiées)
data = pd.DataFrame({
    "heures_revision": [
        1, 2, 1, 3, 2, 4, 5, 6, 4, 7,
        8, 6, 9, 3, 5, 8, 2, 7, 9, 10
    ],
    "absences": [
        10, 8, 12, 6, 9, 5, 3, 2, 5, 1,
        0, 3, 1, 8, 4, 0, 11, 2, 0, 1
    ],
    "absences_justifiees": [
        2, 1, 3, 2, 4, 1, 0, 1, 2, 0,
        0, 1, 0, 3, 1, 0, 4, 1, 0, 0
    ],
    "note_precedente": [
        8, 9, 7, 10, 9, 12, 14, 15, 11, 16,
        17, 14, 18, 8, 13, 17, 7, 15, 19, 18
    ],
    "reussite": [
        0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
        1, 1, 1, 0, 1, 1, 0, 1, 1, 1
    ]
})

# 2. Affichage du dataset dans Streamlit (Amélioration 2)
st.subheader("📊 Aperçu du Dataset")
st.dataframe(data, use_container_width=True)

# 3. Séparation des variables (avec la nouvelle variable)
X = data[[
    "heures_revision",
    "absences",
    "absences_justifiees",
    "note_precedente"
]]
y = data["reussite"]

# 4. Séparation entraînement / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Choix du modèle dans la barre latérale (Amélioration 4)
st.sidebar.header("Paramètres du Modèle")
type_modele = st.sidebar.selectbox(
    "Choisir l'algorithme",
    ["Decision Tree", "K-Nearest Neighbors (KNN)"]
)

if type_modele == "Decision Tree":
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
else:
    model = KNeighborsClassifier(n_neighbors=3)

# Entraînement du modèle
model.fit(X_train, y_train)

# 5. Évaluation du modèle
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.sidebar.metric("Exactitude (Accuracy)", f"{accuracy:.0%}")

st.divider()

# 6. Graphique montrant la relation entre heures de révision et réussite (Amélioration 3)
st.subheader("📈 Visualisation des données")
fig, ax = plt.subplots(figsize=(8, 4))
colors = ['red' if r == 0 else 'green' for r in data["reussite"]]
ax.scatter(data["heures_revision"], data["note_precedente"], c=colors, s=100, alpha=0.7)
ax.set_xlabel("Heures de révision")
ax.set_ylabel("Note précédente")
ax.set_title("Relation entre Heures de révision, Note précédente et Réussite (Vert: Réussite, Rouge: Échec)")
st.pyplot(fig)

st.divider()

# 7. Matrice de confusion (Amélioration 5)
st.subheader("📉 Matrice de confusion")
cm = confusion_matrix(y_test, y_pred)
fig_cm, ax_cm = plt.subplots(figsize=(5, 3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax_cm)
ax_cm.set_xlabel("Prédit")
ax_cm.set_ylabel("Réel")
ax_cm.set_title("Matrice de Confusion")
st.pyplot(fig_cm)

st.divider()
st.subheader("🧪 Tester un nouvel étudiant")

col1, col2 = st.columns(2)
with col1:
    heures = st.slider("Heures de révision par jour", 0, 12, 4)
    absences = st.slider("Nombre d'absences non justifiées", 0, 20, 3)

with col2:
    abs_justifiees = st.slider("Nombre d'absences justifiées", 0, 10, 1)
    note = st.slider("Note précédente sur 20", 0, 20, 12)

if st.button("Lancer la prédiction"):
    nouvel_etudiant = pd.DataFrame([{
        "heures_revision": heures,
        "absences": absences,
        "absences_justifiees": abs_justifiees,
        "note_precedente": note
    }])

    prediction = model.predict(nouvel_etudiant)[0]
    
    if hasattr(model, "predict_proba"):
        probabilites = model.predict_proba(nouvel_etudiant)[0]
        st.write(f"Probabilité estimée de réussite : {probabilites[1]:.0%}")

    if prediction == 1:
        st.success("🎉 Le modèle prédit une **réussite**.")
    else:
        st.warning("⚠️ Le modèle prédit un **risque d'échec**.")

st.caption(
    "Projet pédagogique : les données sont fictives. "
    "Les prédictions ne sont pas fiables pour une décision réelle sans données représentatives."
)
