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
    return sents
```
