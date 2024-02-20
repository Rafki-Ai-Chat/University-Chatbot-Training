import pandas as pd
import random
import json
import nltk
import re  #regular expressions

from nltk.stem import wordnet  # for lemmatization
from sklearn.feature_extraction.text import CountVectorizer  # for bag of words (bow)
from nltk import pos_tag  # for parts of speech
from sklearn.metrics import pairwise_distances  # cosine similarity
from nltk import word_tokenize
from nltk.corpus import stopwords

#deeeeeeeeeeeeeeeerby
#from nltk.corpus import wordnet
from nltk.stem import ISRIStemmer
from langdetect import detect




with open('Data_sets/intents.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

df = pd.DataFrame(data['intents'])  #the 'intents' data is a list of dictionaries, the DataFrame will have columns corresponding to the keys of these dictionaries.

def format_menu(menu):
    formatted_menu = ""
    for category in menu:
        formatted_menu += f"{category['category']}:\n"
        for item in category['items']:
            formatted_menu += f"{item['name']} - ${item['price']}\n"
    return formatted_menu.strip()

df.loc[14:15, 'responses'] = df.loc[14:15, 'responses'].apply(format_menu)


dic = {"tag":[], "patterns":[], "responses":[]} #initialize a dictionary
for i in range(len(df)): #iterates over each row of the DataFrame df , it extracts the values of the 'patterns', 'responses', and 'tag' columns for the current row
    ptrns = df[df.index == i]['patterns'].values[0]
    rspns = df[df.index == i]['responses'].values[0]
    tag = df[df.index == i]['tag'].values[0]
    for j in range(len(ptrns)): #it appends the value of current row to the dictionary
        dic['tag'].append(tag)
        dic['patterns'].append(ptrns[j])
        dic['responses'].append(rspns)

df = pd.DataFrame.from_dict(dic) #convert the dicionary into a dataframe to organize the data into a tabular format , that makes the data more easier in further analysis


stop = stopwords.words('english')

#Derby
def detlang(text):
    l=detect(text)
    if l in ['ar', 'fa']:
        stop = stopwords.words('arabic')
        return 'arabic'
    else:
        return 'english'

def cleaning(x):
    cleaned_array = list()
    for i in x:
        a = str(i).lower()  # convert to all lower letters
        #p = re.sub(r'[^a-z0-9]', ' ', a)  # remove any special characters as [# ,& , $ , ...] but keep numbers
        p = re.sub(r'[^\u0600-\u06FFa-z0-9]', ' ',a)  # Derby: remove any characters that are not Arabic or English letters or numbers
        cleaned_array.append(p)  # add variable p to our array names cleaned_array
    return cleaned_array


df.insert(1, 'Cleaned Context', cleaning(df['patterns']), True)


def text_normalization(text):
    if detlang(text) =='english':
        text = str(text).lower()  # convert to all lower letters
        spl_char_text = re.sub(r'[^a-z0-9]', ' ', text)  # remove any special characters including numbers
        tokens = nltk.word_tokenize(spl_char_text)  # tokenize words
        lema = wordnet.WordNetLemmatizer()  # lemmatizer initiation
        tags_list = pos_tag(tokens, tagset = None)  # parts of speech
        lema_words = []
        for token, pos_token in tags_list:
            if pos_token.startswith('V'):  # if the tag from tag_list is a verb, assign 'v' to it's pos_val
                pos_val = 'v'
            elif pos_token.startswith('J'):  # adjective
                pos_val = 'a'
            elif pos_token.startswith('R'):  # adverb
                pos_val = 'r'
            else:  # otherwise it must be a noun
                pos_val = 'n'
            lema_token = lema.lemmatize(token, pos_val)  # performing lemmatization
            lema_words.append(lema_token)  # added the lemamtized words into our list
        return " ".join(lema_words)
    else:#Derby

         spl_char_text = re.sub(r'[^\u0621-\u064A0-9\s]', ' ', text)
         tokens = word_tokenize(spl_char_text)
         stemmer = ISRIStemmer()
         lema_words = [stemmer.stem(token) for token in tokens]
         return " ".join(lema_words)
normalized = df['patterns'].apply(text_normalization)  # calling the function
df.insert(2, 'Normalized patterns', normalized, True)


##lambda x: text_normalization(x) if   in x else x
stop = stopwords.words('english')

def removeStopWords(text):

    Q = []
    s = text.split()  # create an array of words from our text sentence, cut it into words
    q = ''
    for w in s:  # for every word in the given sentence if the word is a stop word ignore it
        if w in stop:
            continue
        else:  # otherwise add it to the end of our array
            Q.append(w)
        q = " ".join(Q)  # create a sentence out of our array of non stop words
    return q
normalized_non_stopwords = df['Normalized patterns'].apply(removeStopWords)
df.insert(3, 'Normalized and StopWords Removed', normalized_non_stopwords, True)

cv = CountVectorizer()  # initializing count vectorizer
x_bow = cv.fit_transform(df['Normalized patterns']).toarray()  # badly speaking this converts words to vectors

features_bow = cv.get_feature_names_out()  # use function to get all the normalized words
df_bow = pd.DataFrame(x_bow, columns = features_bow)  # create dataframe to show the 0, 1 value for each word

def chat_bow(question):
    # if detlang(question) =='english':
    tidy_question = text_normalization(removeStopWords(question))  # clean & lemmatize the question
    # else:#Derby
    #     tidy_question = text_normalization_arabic(removeStopWords(question))
    cv_ = cv.transform([tidy_question]).toarray()  # convert the question into a vector
    cos = 1- pairwise_distances(df_bow, cv_, metric = 'cosine')  # calculate the cosine value
    index_value = cos.argmax()  # find the index of the maximum cosine value , getting the most similar index value to the cosine value of the question
    response_value = df['responses'].loc[index_value]  # get the response value

    if isinstance(response_value, list): #if the reterived response is a list of responses , we choose one of them randomly
        selected_response = random.choice(response_value)  # select a random response from the list
    else:
        selected_response = response_value
    return selected_response

x=True
while x:
    user_input = input("user: ")
    if user_input.lower() == 'exit':
        x=False
    else:
        stop = stopwords.words(detlang(user_input))#Derby

        y=chat_bow(user_input)
        print("chatbot: ",y)

