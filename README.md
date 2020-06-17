## Sentiment Analysis

1. Install Spacy
   - Download **en_core_web_sm** library for english language
     ```sh
         python3 -m spacy download en_core_web_sm
     ```

### About Spacy

---

Released around 2015, lightweight compared to popular library NLTK.
It boats many useful functionalities out of box, that NLTK is capable of doing with help of other plugins. But, in SpaCy things are very straight forward, and it usually implies the best algorith to use a task, rather than like NLTK that provides tons of ways to do a single task.

SpaCy Dependency Parser Engine is based on [**Stanford Typed Dependencies Manual**](https://nlp.stanford.edu/software/dependencies_manual.pdf)

## PreProcessing Reviews

**Initial Structure of CodeBase**
```py
import spacy
import pickle
import string
from spacy.lang.en import STOP_WORDS

class PreProcessor:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.stop_words = set(STOP_WORDS)
        self.stop_words.update(string.punctuation)
        self.stop_words.remove('not')

```

Reviews would usually consist of many sentences, joined with conjuctions, and prepositions.

My attempt here is to _split_ **Review Document** into **Custom Sentences** such that each individual sentence would contain **singular polarity**, either **+ve** or **-ve**.

```py
    review = "I liked the food, but service was awful. Ambience was damn poor."
```

will be converted into 3 sentences, like

```py
    [
        "I liked the food",
        "service was awful",
        "Ambience was damn poor"
    ]
```

**How it has been done ?**

```py
## split_into_sents belongs to class PreProcessor

def split_into_sents(self, review):
    if not isinstance(review, spacy.tokens.doc.Doc):
        review = self.nlp(review)

    sents = []
    for sentence in review.sents:
        start = 0
        counter = 0
        print("Sentence: ", sentence)
        for token in sentence:
            # 89 -> Conjunctions,
            # 97 -> Punctuations
            if token.pos in [89, 97] or token.text.strip() == ',':
                if counter > start:
                    sents.append(sentence[start: counter])
                start = counter + 1
            counter += 1
        if counter > start:
            sents.append(sentence[start: counter])
    return sents
```

#### Feature Extraction
***

**Feature Extraction** is pretty important in my use case, cause i am doing **ASPECT BASED SENTIMENT ANALYSIS**.

Aspect Based Sentiment Analysis means extracting Entities and Features from statements, such that we could map features to entities, but could still calculate the polarity of that sentence.

>>
    > It is important in case of **Restaurant Reviews based Sentiment Analysis Model**, cause then we would be able to identify what **FEATURES** people **LIKED** or **DISLIKED** in which **ENTITIES** 
>>


**How it's been done ?**
***

It starts with an assumption, that kind of works in most cases,
>>
    > **ENTITIES** are **nouns** in the sentences, irrespective of whether they are used as a _subject_, _object_ or a _complement_.
    >
    > **FEATURES** are **adjectives** or sometimes **verbs** in sentences as these words would reflect what message is being coveyed about those **ENTITIES**
>>

Following the assumption, i just extracted entities and features

```py
    def feature_extraction(self, custom_sent):
        features = {}
        
        nouns = []
        verbs = []
        adj = []
        
        # 92 -> NOUN, 96 -> Proper Noun
        # 95 -> PRONOUN
        # 86 -> AdVerb
        # 84 -> Adjective
        # 100 -> VERB
        # 87 -> AUX. VERB
        # 94 -> Partition (mostly used alongside AUX. VERB)
        for token in custom_sent:
            if token.pos in [92, 96]:
                nouns.append(token.lemma_)
            elif token.pos in [84, 86]:
                adj.append(token.lemma_)
            elif token.pos in [100, 87, 94]:
                verbs.append(token.lemma_)
        return { 
            "entity": ', '.join(nouns),
            "features": ' '.join(adj) if len(adj) > 0 else ' '.join(verbs)
        }              
```

### Trying out with Various ML Models

#### Some Common Terms

**Term Frequency (TF):** It refers to how often a term is present in a document. 
` TF = Number of times a term is present / Total No. of TERMS in that document `

**Inverse Document Frequency (IDF):** It measures how important a term is in a collection or corpus 
` IDF = log_e(Total No. of Documents / No. of Documents that contain the term) `

**Bias** are ERRORs from WRONG Assumptions in the learning algorithm
>>
    > High Bias means Underfitting, as more BIAS would mean more WRONG algorithms
    > Low Bias means Overfitting, as low BIAS would mean algorithm too fit on training data
>>

**Variance** are algorithm's sensitivity to Noise in Training Data
>>
    > High Variance means it modelled more NOISE from training data too, so OVERFITTING
    > Lower the variance better the algorithm
>>


**Bayes Theorem:** says presence of a particular feature is unrelated to the presence of any other feature in an entity

**We used MultinomialNB in our use case, cause it works better with TF-IDF**

**Random Forest Model** uses a collection of Decision Trees, which follow a series of YES/NO Questions to come to a conclusion

**K-Nearest Neighbors** is a statistical modelling of classification that groups K features together to form heaps of classifications

**Support Vector Machines** works amazing on two-group classification problems, it uses a HyperPlanes to separate one group from another