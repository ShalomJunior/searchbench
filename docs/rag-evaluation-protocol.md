# Protocole d'Évaluation RAG (Retrieval-Augmented Generation)

L'évaluation d'un système RAG ne peut pas se réduire à de simples métriques comme le Recall ou le MRR, car elle doit prendre en compte la qualité de la synthèse et la fiabilité des citations.

Conformément aux recommandations d'audit, la génération (réponses) et la recherche (candidats) doivent être évaluées séparément. Ce document décrit le protocole manuel pour évaluer les réponses du modèle sur le sous-ensemble de 50 questions `SciFact`.

## Critères d'Évaluation Manuelle

Pour chaque question du dataset test et sa réponse générée, attribuez un score (ex: 0, 0.5, ou 1) sur les trois axes suivants :

### 1. Exactitude de la réponse (Correctness)

La réponse générée est-elle factuellement exacte par rapport aux documents _fournis_ ?

- **Piège SciFact** : Sur SciFact, une requête est souvent une _affirmation_ scientifique (ex: "Le cholestérol LDL n'est pas impliqué dans les maladies cardiovasculaires"). Un document pertinent peut prouver que cette affirmation est **fausse**.
- Le modèle ne doit pas "valider" la requête aveuglément. Il doit dire : "D'après les documents, cette affirmation est fausse, car le document indique que..."
- **Critère** : La réponse contredit-elle ou confirme-t-elle la requête de manière juste selon les documents ?

### 2. Soutien des Citations (Groundedness / Citation Accuracy)

Chaque fait ou affirmation dans la réponse générée DOIT être suivi d'une citation `[doc_id]`.

- **Vérification** : Prenez la phrase qui précède la citation `[doc_id]`. Lisez le texte du document correspondant dans le cache de recherche. Le document contient-il _réellement_ cette information ?
- **Hallucination de citation** : Si le modèle cite un document qui ne parle pas du tout de ce fait, c'est une erreur grave.

### 3. Reconnaissance des Manques (Honesty / Refusal)

Si le moteur de recherche (BM25 + Dense + Cross-Encoder) a échoué et n'a ramené **aucun document utile** dans le top 10 pour répondre à la question :

- Le modèle doit admettre qu'il ne sait pas.
- S'il tente d'inventer une réponse en s'appuyant sur ses poids internes (sans citation valide), c'est un échec (hallucination RAG).

## Procédure

1. Exécutez `experiments/generate_rag_cache.py` pour générer `data/rag_retrieval_cache_scifact.json` (les 50 questions et leurs documents figés).
2. Exécutez `experiments/run_rag_generation.py` pour générer les réponses avec votre modèle local ou Kaggle (sauvegardé dans `results/rag_answers_scifact_<model>_<timestamp>.json`).
3. Ouvrez le fichier de résultats généré et notez un échantillon aléatoire de 20 questions selon les 3 critères ci-dessus.
4. Synthétisez les résultats dans votre journal de projet.
