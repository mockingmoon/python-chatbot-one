# imports
import json
import pickle
import random
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

#tkinter
import tkinter
from tkinter import *

mymodel = load_model('./chatbot_mymodel.h5')
intents = json.loads(open('./intents.json').read())
words = pickle.load(open('./words_stored.pkl','rb'))
classes = pickle.load(open('./classes_stored.pkl','rb'))

wnl = WordNetLemmatizer()
ignore_words = ['?','!','.']

# text preprocessing

def text_cleanup(sent):
    dupsent = word_tokenize(sent)
    clsent = [ wnl.lemmatize(wo.lower()) for wo in dupsent if wo not in ignore_words ]
    return clsent

def text_bag_of_words(sent, words):
    clsent = text_cleanup(sent)
    bag = [ int(wo in clsent) for wo in words ]
    return bag

def predict_class(sent, model):
    bow = text_bag_of_words(sent, words)
    res = model.predict([bow])[0]
    error_threshold = 0.25
    results = [ [i,r] for i,r in enumerate(res) ]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent":classes[r[0]], "prob":str(r[1])})
    return return_list

def get_response(ints, intents_json):
    tag = ints[0]['intent']
    allintents = intents_json['intents']
    myintent = list(filter(lambda x: x['tag']==tag, allintents))
    myreply = random.choice(myintent[0]['responses'])
    return myreply

def chatbot_response(text):
    pred_intents = predict_class(text, mymodel)
    replymsg = get_response(pred_intents, intents)
    return replymsg

#Creating GUI with tkinter

def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)
    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))

        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)

base = Tk()
base.title("Hello")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

#Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)

ChatLog.config(state=DISABLED)

#Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

#Create Button to send message
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command= send )

#Create the box to enter message
EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")
#EntryBox.bind("<Return>", send)


#Place all components on the screen
scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

base.mainloop()
