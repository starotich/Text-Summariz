#!/usr/bin/env python
# coding: utf-8

# In[1]:


#install some libraries
get_ipython().system('pip install spacy')
get_ipython().system('pip install pytextrank')


# In[ ]:


#Import neccessary libraries


# In[2]:


import spacy
import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import pytextrank


# In[ ]:


#Define typical spacy pipeline


# In[4]:


get_ipython().system('python -m spacy download en_core_web_lg')
# Load the large English language model
nlp = spacy.load("en_core_web_lg")
# Add TextRank to the spaCy pipeline
nlp.add_pipe('textrank')


# In[7]:


#take a look at an example
example_text="""Deep learning is a subset of machine learning that involves neural networks with multiple 
layers (hence the term "deep"). These neural networks are capable of automatically learning representations of data through 
the process of feature learning. Deep learning algorithms have achieved remarkable success in various tasks such as image 
recognition, speech recognition, natural language processing, and many others."""


# In[8]:


example_text


# In[9]:


#Run spacy pipeline
doc=nlp(example_text)


# In[11]:


#show text summary
for sent in doc._.textrank.summary(limit_sentences=1):
    print(sent)


# In[12]:


for sent in doc._.textrank.summary(limit_phrases=1,limit_sentences=1):
    print(sent)


# In[ ]:




