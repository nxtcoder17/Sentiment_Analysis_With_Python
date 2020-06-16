## Sentiment Analysis

1. Install Spacy
    + Download **en_core_web_sm** library for english language
        ```sh
            python3 -m spacy download en_core_web_sm
        ```

### About Spacy
***
Released around 2015, lightweight compared to popular library NLTK.
It boats many useful functionalities out of box, that NLTK is capable of doing with help of other plugins. But, in SpaCy things are very straight forward, and it usually implies the best algorith to use a task, rather than like NLTK that provides tons of ways to do a single task.

SpaCy Dependency Parser Engine is based on [**Stanford Typed Dependencies Manual**](https://nlp.stanford.edu/software/dependencies_manual.pdf)


## PreProcessing Reviews
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

