# Sentiment Analysis

Even though in most statistical classification methods, the neutral class is
ignored under the assumption that neutral texts lie near the boundary of the
binary classifier, several researchers suggest that, as in every polarity
problem, three categories must be identified. Moreover, it can be proven that
specific classifiers such as the Max Entropy[9] and SVMs[10] can benefit from
the introduction of a neutral class and improve the overall accuracy of the
classification. There are in principle two ways for operating with a neutral
class. Either, the algorithm proceeds by first identifying the neutral language,
filtering it out and then assessing the rest in terms of positive and negative
sentiments, or it builds a three-way classification in one step.[11] This second
approach often involves estimating a probability distribution over all
categories (e.g. naive Bayes classifiers as implemented by the NLTK). Whether
and how to use a neutral class depends on the nature of the data: if the data is
clearly clustered into neutral, negative and positive language, it makes sense
to filter the neutral language out and focus on the polarity between positive
and negative sentiments. If, in contrast, the data are mostly neutral with small
deviations towards positive and negative affect, this strategy would make it
harder to clearly distinguish between the two poles.
>>

#### Feature/aspect-based Sentiment Analysis
* * * It refers to determining the opinions or sentiments expressed on different
  features or aspects of entities, e.g., of a cell phone, a digital camera, or a
  bank.[19] A feature or aspect is an attribute or component of an entity, e.g.
  the screen of a cell phone, the service for a restaurant, or the picture
  quality of a camera. The advantage of feature-based sentiment analysis is the
  possibility to capture nuances about objects of interest. Different features
  can generate different sentiment responses, for example a hotel can have a
  convenient location, but mediocre food.[20] This problem involves several
  sub-problems, e.g., identifying relevant entities, extracting their
  features/aspects, and determining whether an opinion expressed on each
  feature/aspect is positive, negative or neutral.[21] The automatic
  identification of features can be performed with syntactic methods, with topic
  modeling,[22][23] or with deep learning.[24] More detailed discussions about
  this level of sentiment analysis can be found in Liu's work.[25]

#### Aspect Based Sentiment Analysis [Monkey Learn](https://monkeylearn.com/blog/aspect-based-sentiment-analysis/)

+ **Cue Based Sentiment Analysis (CBSA)**: A text analysis technique that breaks
  down text into aspects (attributes or components of a product or service) and
  allocates each one a sentiment level. This technique can help businesses
  become customer-centric and place their customers at the heart of everything
  they do. It’s about listening to their customers, understanding their voice,
  analyzing their feedback and learning more about customer experiences, as well
  as their expectations for products or services.

##### How To
1. **Creating the Models**
    - PreProcessing the Data into different **OPINION Units** e.g. I like the
      food but the service was not that good.  OPINION 1: I like the food
      OPINION 2: the service was not that good.
    - Create a Sentiment Analysis Model
    - Create an Aspect Model We need to create an ASPECT Model that suits our
      needs, for different scenarios it would need to be different, as we don't 
      require the same feature set in different scenarios.

##### Example explanation
For example, a software company might want to understand the specific
sentiments towards different aspects of its product. A review might say:
“support were great but UI is confusing”, which contains a positive sentiment
towards ‘aspect customer support’ but a negative sentiment towards ‘aspect user
experience’. A sentiment analysis model might classify the overall sentiment as
negative, and ignore the fact that the staff did a good job, or vice versa.
Whereas an aspect-based analysis model would differentiate between aspects and
allocate a sentiment to each one.

Once data has been imported, either from internal or external sources,
aspect-based analysis tools are able to classify sentiments towards specific
product features or services. And this is where it gets interesting for
organizations. Customers want to feel like they’re being listened to, and by
using deeper machine learning models like aspect-based sentiment analysis,
businesses can send quick, efficient and personalized responses. And for
customer support teams it means streamlining processes and gaining more
valuable insights. 


+ Sentiment Analysis References:
  [Tutorial](https://www.cs.uic.edu/~liub/FBS/Sentiment-Analysis-tutorial-AAAI-2011.pdf)

### Opinion
+ **Defintion**: An opinion is a quintuple 
(e<sub>j</sub>, a<sub>jk</sub>, so<sub>ijkl</sub>, h<sub>i</sub>, t<sub>l</sub>)
        (ej, ajk, soijkl, hi, tl),
  is a target entity.

    - a<sub>jk</sub> is an aspect/feature of the entity ej. 
    - so<sub>ijkl</sub> is the sentiment value of the opinion from the opinion 
    holder hi on feature a<sub>jk</sub> on
    entity e<sub>j</sub> at time t<sub>l</sub>.
    - so<sub>ijkl</sub> is +ve, -ve, or neu, or more granular ratings.
    - h<sub>i</sub> is an opinion holder
    - t<sub>l</sub> is the time when the opinion is expressed.

+ [**Aspect Based Opinion Mining - Peter Min**](https://medium.com/@pmin91/aspect-based-opinion-mining-nlp-with-python-a53eb4752800)

+ Negative Words Lexicon : [Download](http://ptrckprry.com/course/ssd/data/negative-words.txt)
    [Source](https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html)


### Stuffs 
- "That is not how you make dahi kababs" - should be -ve

### Need to take care of 
- Things like, Non Veg, Non-Veg, non-veg, non veg 
    + Model should consider Non veg as a whole not as 2 separate entities

- When i am splitting on '.', I better make sure that that period (.) is not part of a web address or an email address

- You need to deal with Capitalisation issues too, you can't consider everything in small case, as 
    'US' is completely different from 'us'


