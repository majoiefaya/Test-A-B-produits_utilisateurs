import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# Titre de l'application
st.title("Test A/B - Analyse des Revenus")

# Introduction au projet
st.markdown("""
### Introduction au Test A/B
L'objectif de cette étude était de mener un test A/B pour comparer les revenus générés par deux groupes d'utilisateurs : un groupe **contrôle** et un groupe **variant**. Nous avons souhaité analyser si le variant testé avait un impact significatif sur les revenus générés par les utilisateurs, en utilisant des tests statistiques pour valider ou invalider cette hypothèse.

### Méthodologie
Le test a été réalisé en plusieurs étapes, comprenant la préparation et l'exploration des données, l'analyse des statistiques descriptives, la vérification de la normalité des données, et l'application d'un test statistique adapté pour comparer les deux groupes.
""")

# Charger les données
file_path = os.path.join(os.path.dirname(__file__), '../AB_Test_Results.csv')
data = pd.read_csv(file_path)

# Afficher les premières lignes
st.write("Voici les premières lignes des données chargées :")
st.write(data.head())

# Vérifier s'il y a des valeurs manquantes
missing_ratio = data.isnull().mean()
st.write("Proportions de valeurs manquantes par colonne :")
st.write(missing_ratio)

# Supprimer les colonnes avec plus de 40% de valeurs manquantes
seuil_colonnes = 0.4
data = data.drop(columns=missing_ratio[missing_ratio > seuil_colonnes].index)

# Supprimer les lignes avec des valeurs manquantes
data = data.dropna()

# Remplir les valeurs manquantes de 'REVENUE' par sa moyenne
data['REVENUE'].fillna(data['REVENUE'].mean(), inplace=True)

# Vérifier la répartition des utilisateurs dans chaque groupe
group_counts = data['VARIANT_NAME'].value_counts()
st.write("Répartition des utilisateurs dans chaque groupe :")
st.write(group_counts)

# Séparer les groupes
control_group = data[data['VARIANT_NAME'] == 'control']
variant_group = data[data['VARIANT_NAME'] == 'variant']

# Statistiques descriptives
control_stats = control_group['REVENUE'].describe()
variant_stats = variant_group['REVENUE'].describe()
st.write("### Statistiques descriptives pour le groupe contrôle :")
st.write(control_stats)
st.write("### Statistiques descriptives pour le groupe variant :")
st.write(variant_stats)

# Créer un histogramme pour chaque groupe
st.write("### Distribution des Revenus")
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Histogramme pour le groupe contrôle
ax[0].hist(control_group['REVENUE'], bins=20, color='blue', alpha=0.7, label='Contrôle')
ax[0].set_title('Distribution des Revenus - Groupe de Contrôle')
ax[0].set_xlabel('Revenu')
ax[0].set_ylabel('Fréquence')

# Histogramme pour le groupe variant
ax[1].hist(variant_group['REVENUE'], bins=20, color='green', alpha=0.7, label='Variant')
ax[1].set_title('Distribution des Revenus - Groupe Variant')
ax[1].set_xlabel('Revenu')
ax[1].set_ylabel('Fréquence')

st.pyplot(fig)

# Boxplot pour chaque groupe
st.write("### Boxplot des Revenus")
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Boxplot pour le groupe contrôle
ax[0].boxplot(control_group['REVENUE'], vert=False, patch_artist=True, boxprops=dict(facecolor="blue", color="blue"), medianprops=dict(color="yellow"))
ax[0].set_title('Boxplot des Revenus - Groupe de Contrôle')
ax[0].set_xlabel('Revenu')

# Boxplot pour le groupe variant
ax[1].boxplot(variant_group['REVENUE'], vert=False, patch_artist=True, boxprops=dict(facecolor="green", color="green"), medianprops=dict(color="yellow"))
ax[1].set_title('Boxplot des Revenus - Groupe Variant')
ax[1].set_xlabel('Revenu')

st.pyplot(fig)

# Test de normalité (Shapiro-Wilk)
shapiro_control = stats.shapiro(control_group['REVENUE'])
shapiro_variant = stats.shapiro(variant_group['REVENUE'])

st.write(f"Test de normalité pour le groupe contrôle : {shapiro_control}")
st.write(f"Test de normalité pour le groupe variant : {shapiro_variant}")

# Si les données sont normales, effectuer un t-test
if shapiro_control.pvalue > 0.05 and shapiro_variant.pvalue > 0.05:
    st.write("Les données suivent une distribution normale. Effectuons un t-test.")
    t_test_result = stats.ttest_ind(control_group['REVENUE'], variant_group['REVENUE'])
    st.write(f"Résultat du t-test : {t_test_result}")
else:
    st.write("Les données ne suivent pas une distribution normale. Effectuons un test de Mann-Whitney.")
    mann_whitney_result = stats.mannwhitneyu(control_group['REVENUE'], variant_group['REVENUE'])
    st.write(f"Résultat du test de Mann-Whitney : {mann_whitney_result}")

# Conclusion et recommandations
st.markdown("""
### Conclusion et Recommandations
Le test A/B n'a pas montré de différence significative entre les groupes contrôle et variant. Cela suggère que le variant testé n'a pas eu un impact mesurable sur les revenus des utilisateurs. Cependant, plusieurs éléments peuvent expliquer ces résultats :

1. **Revoir l'hypothèse** : L'élément testé pourrait ne pas avoir été suffisamment significatif.
2. **Augmenter l'échantillon** : Une augmentation de la taille de l'échantillon pourrait rendre le test plus fiable.
3. **Optimiser l'engagement des utilisateurs** : Si la majorité des utilisateurs génèrent peu ou pas de revenus, il peut être nécessaire de mieux cibler l'audience ou d'améliorer la visibilité du changement testé.

### Prochaines étapes :
- Tester d'autres éléments significatifs du produit.
- Prolonger l'expérience ou augmenter la taille de l'échantillon.
""")
