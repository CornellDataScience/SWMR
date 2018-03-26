# In-Domain Cross-Writer Sentiment Classification on Movie Reviews

## Research Context
With the rise of the information age, sentiment analysis on texts has been a crucial Natural Language Processing task that many researchers have put effort on. Recent researches on sentiment analysis have obtained great results using various methods such as Naive Bayes, SVM, Recurrent Neural Networks and etc.

More recently, cross-domain sentiment classification that applies the model trained on texts in one specific domain, for instance, reviews to the texts in another domain, for instance, tweets has been a topic of interest because datasets without standard labels can be classified using a pretrained model, where normal sentiment classification models give significantly worse results. Popular methods for cross-doamin sentiment classification include Sentiment Sensitive Thesaurus, Stacked Denoising Auto-Encoders, Spectral Feature Alignment and etc.

## Research Objective: 
We are going to investigate how current models and our model perform on the In-Domain Cross-Writer sentiment classification task on movie review dataset by Pang and Lee \cite{Pang+Lee:05a}.

First we need to show that models that excel at normal sentiment classification tasks will end up with worse performance on this task. Then we start by presenting an empirical study of current methods of normal sentiment classification tasks and hopefully propose a new method that specifically addresses the this task.

If our method turns out that no significantly improved performance on experiments is observed, then we will need to account for the failure and gain insights on why some current methods outperform the others.  

## Data: 
### movie reviews on Rotten Tomato 
![Data](../master/present/data.png)

### Word Vector
![Data](../master/present/wordvector.png)

### Two principal components

## Model

