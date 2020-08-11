## Dataset:
The dataset I was working on consisted of 21 classes and was comprimsed of emails. Overall, the dataset only contained 450 rows and I used an 80-20 split 
for training and classification purposes.

## Techniques used:
1. **Sklearn**: SVC with 2 features: **tfid vectorizer** (ngram = (1,3)) and embeddings generated using **SpacyVectorTransformer**
1. **[TextCategorizer in Spacy](https://spacy.io/usage/training#textcat)**
1. **[Ludwig](https://ludwig-ai.github.io/ludwig-docs/user_guide/)**
1. **[FastText](https://fasttext.cc/docs/en/supervised-tutorial.html)**
1. **Keras**
1. **[Fined tuned roBERTa using k-train](https://github.com/amaiya/ktrain)**

## Results:
The best results I had were by using a roBERTa 
