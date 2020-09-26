## Dataset:
The dataset I was working with consisted of emails with a total of 21 classes. The dataset contained 450 rows and I used an 80-20 split 
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

## **Text Classification Techniques**



1. Categorizing emails into 21 distinct categories is difficult due to two reasons: 

   -  there is simply not enough data to properly train the model
   - the data in the classes is imbalanced e.g. some classes have 40 data rows and others only have 5.

   One technique I tried was to lump together similar classes into categories. The idea was to train a classifier that can place an email into one of these categories and then another classifier would further classifiy that email into one of the classes belonging to that category. With this method, I ended up with four categories and a total of five classifiers: one for classifying an email to a category (from four categories) and the other four that classifiy an email to a particular class of that category. This technique however did not produce relatively decent results.

2. [**SVC (Support Vector Classifier) from Sklearn**](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html):

   I used the SVC from from Sklearn and intialized it with the following paramters:

   ```python
   SVC(C=150, gamma=0.02, probability=True)
   ```

   Please have a look at [this](https://medium.com/all-things-ai/in-depth-parameter-tuning-for-svc-758215394769) articles to get a better understanding of the SVC paramters.

   No machine learning classifier or model can directly operate on text. It only understands numbers, whether the input is text, images etc and so first, we must convert our text data into a numeric form. To do this I have used two techniques in conjuction with the SVC classifier:

   - The Sklearn library provides three vectorizer methods for converting text to numeric form: **TfidfVectorizer**, **CountVectorizer** and **HashingVectorizer**. After testing with all of these methods, [**TfIDVectorizer**](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer) with the following parameters gave the best results: 

     ```python
     TfidfVectorizer(ngram_range=(1, 3))
     ```

     The `ngram_range` defines the number of tokens to use from the text e.g. for a text `he ate the apple`, a value of 1 will produce: `[he, ate, the, apple]` which are called unigrams. A a value of 2 will produce: `[he ate, he the, he apple]` and is called bigrams and then trigrams and so on. By giving a range `(1,3)` we are essentially using unigrams, bigrams and trigrams altogether.

   - A vectorizer utilizing word vectors from [sPacy](https://spacy.io/usage/vectors-similarity/).

   These two vectorizerers are using together in the following way:

   ```python
   outer_pipe = Pipeline(
       steps=[
           ("combined_features", FeatureUnion(
                   transformer_list=[
                       ("tfid", TfidfVectorizer(ngram_range=(1,3))),
                       ("embed", SpacyVectorTransformer(nlp)),
                   ]
           )),
           ("classifier", SVC(C=150, gamma=0.02, probability=True))
       ]
   )
   ```

   The output of both vectorizerers are then fed to our earlier SVC model.

3. [**Ludwig**](https://github.com/uber/ludwig):

   Ludwig is a ML library developed by Uber that allows for the creation of models without expending effort into coding them. 

   The following model was used:

   ```python
   model_definition = 
   {
     'input_features': 
                  [{'name': 'Text', 'type': 'text', "level": "word", "dropout": True}], 
    
      'output_features': 
       		   [{'name': 'Label', 'type': 'category'}],        
    
       'training': 
                 {'epochs':50}
   }
   
   model = LudwigModel(model_definition)
   training_stats = model.train(training_dataframe, logging_level=logging.INFO)
   ```

    The results from this model were not particuarly good, with an accuracy around 50% at the best.

4. [**fastText**](https://fasttext.cc/):

   fastText is a text classification library from Facebook. fastText does not provide much facilities regarding finetuning the model and requires input in a different format. It's result were better then Ludwig but not particuarly decent.

   1. **Preparing data for fastText**:

      ```python
      data = pd.read_csv("Train.csv")
      data = data.dropna()
      training_df, validation_df = train_test_split(data, test_size=0.20, random_state=45)
      
      def create_file(df, filename):
          file = open(filename +'.txt', "w")
      
          for index, row in df.iterrows():
              label = row['Label'].replace("_", "-")
              text = row['Text'].replace('\r', '').replace('\n', '')
              text = tokenizer(text)
      
              line = '\n__label__' + label + ' ' + text
              file.write(line)
      
          file.close()
          
      create_file(training_df, 'fasttext-train')
      create_file(validation_df, 'fasttext-val')
      ```

   2. **Training model**:

      ```python
      model = fasttext.train_supervised(input="fasttext-train.txt", autotuneValidationFile='fasttext-val.txt')                            
      ```

      The `autotuneValidationFile` allows the model to automatically set its parameters. 

5. [**kTrain**](https://github.com/amaiya/ktrain):

   kTrain is a library designed to make use of deep learning and other ML models easy. We are particularly interested in its [text classification faciltiies](https://nbviewer.jupyter.org/github/amaiya/ktrain/blob/master/tutorials/tutorial-04-text-classification.ipynb). Overall, using the **roberta-large** model gave the best overall results in terms of prediction accuracy. 

   ```python
   MODEL_NAME = 'roberta-large' 
   t = text.Transformer(MODEL_NAME, classes=my_classes)
   trn = t.preprocess_train(x_train.values, y_train.values)
   val = t.preprocess_test(x_test.values, y_test.values)
   model = t.get_classifier()
   # Batch size determines the number of data rows using during training. The
   # larger the batch size, the more resource-intensize the training will be.
   learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=8)
   
   # 5e-5 = 0.00005 and specifies the learning rate.
   # 6 is the number of epochs i.e. training cycles.
   learner.fit_onecycle(5e-5, 6)
   
   predictor = ktrain.get_predictor(learner.model, preproc=t)
   
   # Testing the models accuracy
   predictions = predictor.predict(x_test.values)
   print("Accuracy:", metrics.accuracy_score(y_true=y_test.values, y_pred=predictions, normalize=False))
   print(metrics.classification_report(y_true=y_test.values, y_pred=predictions))
   ```

6. **Creating an Ensemble of SVC and Roberta**:

   An ensemble is simply a collection of multiple ML models trained on the same input and which combine their ouputs into a single prediction. Since SVC and Roberta were giving me the best results, I tried creating an ensemble of the two to see if I could overcome their individual weaknesses and combine their strengths:

   ```python
   from mlxtend.classifier import EnsembleVoteClassifier
   ensemble = EnsembleVoteClassifier(clfs=[pipe, predictor], weights=[1, 1], voting='soft', refit=False)
   ensemble.fit(x_train, y_train)
   ensmbl_preds = ensemble.predict(x_test.values)
   ```

   Here `pipe` and `predictor` in the `clfs (classifiers)` parameter of `EnsembleVoteClassifier` are the **roberta** and **SVC** models. The issue with this approach was that the probability against any single prediction of **roberta** was very high so it would almost always dominate the overall prediction. Hence, the ensemble gave slighlty worse results than either the individual predictors. 

7. **Techniques employed for performance tuning**:

   1. **Parameter optimization**:

      Parameter optimization basically invovles trying out all sorts of different parameter values for a ML model to try to finds the values that give the best results e.g. the following are ranges for different parameters of a RandomForest:

      ```python
      params = {
          "combined_features__bow__tfidf__use_idf": [True, False],
          "combined_features__bow__tfidf__ngram_range": [(1, 1), (1, 2)],
          "classifier__bootstrap": [True, False],
          "classifier__class_weight": ["balanced", None],
          "classifier__n_estimators": [100, 300, 500, 800, 1200],
          "classifier__max_depth": [5, 8, 15, 25, 30],
          "classifier__min_samples_split": [2, 5, 10, 15, 100],
          "classifier__min_samples_leaf": [1, 2, 5, 10]
      }
      search = RandomizedSearchCV(pipe, params)
      ```

      [`RandomizedSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV) tries random variations of the parameters to find the best fit. There is also a [`GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV) which is much more exhaustive and tries all possible combinations. However, this can take a very long time which is why the randomized search is sometimes preferred. 

   2. **Data Augmentation**:

      This technique is used when the training data is limited. The idea is to use different strategies to augment the training data availabe e.g for text classification we can replace words with their synonyms. As an example, for the sentence `amer enjoyed eating the apple`, we could generate another similar sentence: `amer liked eating the apple`.










