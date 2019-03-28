#
import pandas
import os
import pickle
import re
import sys
import string
import numpy
import nltk
import csv
from sklearn.externals import joblib
from sklearn import cross_validation
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
stemmer=SnowballStemmer('english')
###############################################################################################################################################
#Stems passed string and returns stemmed string without special characters and punctuations
def review_to_words(raw_review):
      letters_only=re.sub("[^a-zA-Z]"," ",raw_review)
      words=letters_only.split()
      list=[]
      for k in words:
      	s=stemmer.stem(k)
        list.append(s)
      return(" ".join(list))
################################################################################################################################################
clf=joblib.load("tr4bfclf.pkl")
from sklearn.feature_selection  import SelectPercentile,f_classif
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=joblib.load("tr4bfvec.pkl")
selector=joblib.load("tr4bfsel.pkl")
print("Hello,I am intended to talk to you and to solve your difficulties.You could ask me anything and I'll try to solve your problems.Could you please tell your name?")
with open('newtree.txt','rb') as fp:
       list=pickle.load(fp)
curr=list
raw_input()
print('Where do you come from ?')
raw_input()
print('I would like to know more about you.What are your likings.Are you fond of something?')
raw_input()
print('Alright!So how to feel about your life.Do you believe that your past was blissful.')
raw_input()
print('Tell me,is there something that you would like to accomplish in your future?I would be glad to know about it.')
raw_input()


print('Our current memories are shaped by our past experiences.What do you think is one such expereince that has had a major impact on your personality?')
pred=clf.predict(selector.transform(vectorizer.transform([review_to_words(raw_input())]).toarray()))
if pred==0:
     print('Several experiences can bring about a very negative impact on our mind .these memories might flash back from time to time.Sharing these with others helps get some relief.')
if pred==4:
    print('You do not seem to be that affected by your previous bad experiences that you might have had.It is really a good thing to overcome thoughts yourself.')
if pred==2:
    print('It seems you do not want to talk to me about it.Well that is fine as well but I should tell you that having sad memories about a past experience is pretty normal all you have to try to overcome it!')




print('What do you think have been the changes in your behaviour and interests recently?')
pred=clf.predict(selector.transform(vectorizer.transform([review_to_words(raw_input())]).toarray()))
if pred==0:
     print('You seem to have control over your behavour and interests.Often changes in behaviour occur when you stick to some sad memories or overthink about your past.These changes in behaviour can have a negative impact on your family and friends.')
if pred==4:
    print('It seems you do are not in much control over your behaviour and your interests are swirling.Yet it is commendable that you are atleast aware of this fact.Often changes in behaviour occur when you stick to some sad memories or overthink about your past.These changes in behaviour can have a negative impact on your family and friends.')
if pred==2:
    print('You seem to be a bit confused about your behaviour and interests.first of all you need to accept that there have been changes in you actoins due to your thoughts as these thoughts may have a negative impact on your friends and family.')

print('Consider this,you are able to travel back in time!What according to you would you like to change?')
raw_input()
print('That response was quick! such thoughts do come to mind ,changing the past seems so amazing and a source of relief.The lesser the thoughts you have about changing your past the better.live your present to the fullest.')


#####2 categories
clf=joblib.load("tr2bfclf.pkl")
vectorizer=joblib.load("tr2bfvec.pkl")
selector=joblib.load("tr2bfsel.pkl")
print('Coming back to your memories, such incidents do happen with people sometimes.its not your fault but sometimes we need to change our thinking process.many get affected by it but at some stage it is our thinking process that puts us into greater trouble or deep thoughts?tell me more about your daily schedule,has it changed recently?are you becoming lazy?')
pred=clf.predict(selector.transform(vectorizer.transform([review_to_words(raw_input())]).toarray()))

if pred=='0':
    print('well it happens.you need to understand that you are thinking a quite lot.you need to get out of past and live the present.you should try observing your thoughts.remember thoughts are not facts.thinking excessively makes one feel that what we feel is the reality hence one develops a negative side towards certain events or people disturbing oneunnecessarily.thats why you are loosing interest in certain things.by the way,are you an emotional person,do you often feel different negative emotions say in a difficult situation?')
    pred=clf.predict(selector.transform(vectorizer.transform([review_to_words(raw_input())]).toarray()))
    if pred=='0':
        print('you must remember that emotions are information not facts.it maybe difficult but notice and practice emotions that you want to feel more often.do not make emotions larger than what they are.this can help reduce a lot of problems.be able to notife emotions without pushing them or making them larger than useful.do you feel anxiety often?i mean it can happen in such situations.')
        pred=clf.predict(selector.transform(vectorizer.transform([review_to_words(raw_input())]).toarray()))
        if pred=='1':
           print('nice.i think you are going in the right direction.just remember that your thinking influences everything.the way you percieve things actually makes you thing in that direction.controlling your emotions,your behaviour,self-introspection can always help in ugly situations that you may come across in life.maybe you percieve yourself as someone bad or worthless?')
           pred=clf.predict(selector.transform(vectorizer.transform([review_to_words(raw_input())]).toarray()))
           if pred=='1':
               print('nice.i think you are going in the right direction.just remember that your thinking influences everything.the way you percieve things actually makes you thing in that direction.controlling your emotions,your behaviour,self-introspection can always help in ugly situations that you may come across in life.')
        

           if pred=='0':
              print('you know you should choose behaviour consistent with the person you want to be and ideally it should bemore towards your ideal self.behaviour can trigger emotions that maybe can lead in a negative direction which could be harmful for yourself only.you should be able to choose your behaviour,create space between impulse and action so that then you can make an effective choice.')
        

        if pred=='0':
           print('one major step that you should seek is that you should sit and accept anxiety.it is pretty a normal thingand can happen in tense situations.unnecessarily making it a big issue is worthless as well it consumes a lot of your mind.as i told you before a free mind is a devils mind.so keep yourself engaged in certain things.maybe you percieve yourself as someone bad or worthless?')
           pred=clf.predict(selector.transform(vectorizer.transform([review_to_words(raw_input())]).toarray()))
           if pred=='1':
              print('nice.i think you are going in the right direction.just remember that your thinking influences everything.the way you percieve things actually makes you thing in that direction.controlling your emotions,your behaviour,self-introspection can always help in ugly situations that you may come across in life.')
        

           if pred=='0':
             print('you know you should choose behaviour consistent with the person you want to be and ideally it should bemore towards your ideal self.behaviour can trigger emotions that maybe can lead in a negative direction which could be harmful for yourself only.you should be able to choose your behaviour,create space between impulse and action so that then you can make an effective choice.')
    if pred=='1':
        print('that is a good quality in you,you dont hype emotions.so recently or after the incident has your behaviour or attitude towards self or towards others changed negatively?maybe you percieve yourself as someone bad or worthless?')
        pred=clf.predict(selector.transform(vectorizer.transform([review_to_words(raw_input())]).toarray()))
        if pred=='1':
           print('nice.i think you are going in the right direction.just remember that your thinking influences everything.the way you percieve things actually makes you thing in that direction.controlling your emotions,your behaviour,self-introspection can always help in ugly situations that you may come across in life.do you feel anxiety often?i mean it can happen in such situations.')
           pred=clf.predict(selector.transform(vectorizer.transform([review_to_words(raw_input())]).toarray()))
           if pred=='1':
            print('nice.i think you are going in the right direction.just remember that your thinking influences everything.the way you percieve things actually makes you thing in that direction.controlling your emotions,your behaviour,self-introspection can always help in ugly situations that you may come across in life.maybe you percieve yourself as someone bad or worthless?')
           pred=clf.predict(selector.transform(vectorizer.transform([review_to_words(raw_input())]).toarray()))
           if pred=='1':
               print('nice.i think you are going in the right direction.just remember that your thinking influences everything.the way you percieve things actually makes you thing in that direction.controlling your emotions,your behaviour,self-introspection can always help in ugly situations that you may come across in life.')

        if pred=='0':
           print('you know you should choose behaviour consistent with the person you want to be and ideally it should bemore towards your ideal self.behaviour can trigger emotions that maybe can lead in a negative direction which could be harmful for yourself only.you should be able to choose your behaviour,create space between impulse and action so that then you can make an effective choice.do you feel anxiety often?i mean it can happen in such situations.')
           pred=clf.predict(selector.transform(vectorizer.transform([review_to_words(raw_input())]).toarray()))
           if pred=='1':
              print('nice.i think you are going in the right direction.just remember that your thinking influences everything.the way you percieve things actually makes you thing in that direction.controlling your emotions,your behaviour,self-introspection can always help in ugly situations that you may come across in life.maybe you percieve yourself as someone bad or worthless?')
           pred=clf.predict(selector.transform(vectorizer.transform([review_to_words(raw_input())]).toarray()))
           if pred=='1':
               print('nice.i think you are going in the right direction.just remember that your thinking influences everything.the way you percieve things actually makes you thing in that direction.controlling your emotions,your behaviour,self-introspection can always help in ugly situations that you may come across in life.') 


if pred=='1':
    print('so you do have control over yourself.often people stop doing thier daily work and spend lot of time visiting the past.however,do you think you are loosing friends or maybe feeling alone.someone who was a friend before,you want to get in touch with?')
    pred=clf.predict(selector.transform(vectorizer.transform([review_to_words(raw_input())]).toarray()))
    if pred=='0':
       print('hmm.it happens.you should be aware and shift negative/critical/emotionally abusive self talk.maybe you talkwith others the same way and you know they may not like it and then people say start ignoring and all that stuffthat can make you sad.so sometimes if you think that you are stupid or lazy or worthless or anything justtry to say opposite of that that i am strong,i am fit.it helps in conversations.believe me itreally does help.coming to the point,is your mind easily distracted into different things?are you finding it difficult to concentrate on one particular thing that you are doing?')
       pred=clf.predict(selector.transform(vectorizer.transform([review_to_words(raw_input())]).toarray()))
       if pred=='0':
          print('it is a difficult task but you should start noticing where your mind goes and bring it back to the present moment.maybe you go to the past and thing of lot of unnecessary things which will continue making you feel sad.maybe you go to the future and worry about certain things based on your present.say for example i am not able to perform well,ow will i get a job.what will be my parents reaction and lots of other stuff.you are responding well.thats nice to see.in difficult past situations whenever you visit them in the present do you think that maybe changing your actions could have yielded a better result?')
       
       if pred=='1':
          print('nice.i think you are going in the right direction.just remember that your thinking influences everything.the way you percieve things actually makes you thing in that direction.controlling your emotions,your behaviour,self-introspection can always help in ugly situations that you may come across in life.you are responding well.thats nice to see.in difficult past situations whenever you visit them in the present do you think that maybe changing your actions could have yielded a better result?')
    
    if pred=='1':
       print('you are responding well.thats nice to see.in difficult past situations whenever you visit them in the present do you think that maybe changing your actions could have yielded a better result?')
       pred=clf.predict(selector.transform(vectorizer.transform([review_to_words(raw_input())]).toarray()))
       if pred=='0':
          print('you know you need to be more effective in the moment.you need to decrease should and what-ifs and accept reality.thinking much about what you did wrong or what if this would have happened is actually hurting you.you are actually blaming yourself for something that is inevitable.you need to decrease habits of shame,denial,blame which inturn will help you a lot.is your mind easily distracted into different things?are you finding it difficult to concentrate on one particular thing that you are doing?')
          pred=clf.predict(selector.transform(vectorizer.transform([review_to_words(raw_input())]).toarray()))
          if pred=='0':
            print('it is a difficult task but you should start noticing where your mind goes and bring it back to the present moment.maybe you go to the past and thing of lot of unnecessary things which will continue making you feel sad.maybe you go to the future and worry about certain things based on your present.say for example i am not able to perform well,ow will i get a job.what will be my parents reaction and lots of other stuff.')
       
          if pred=='1':
             print('nice.i think you are going in the right direction.just remember that your thinking influences everything.the way you percieve things actually makes you thing in that direction.controlling your emotions,your behaviour,self-introspection can always help in ugly situations that you may come across in life.') 
       if pred=='1':
          print('nice.i think you are going in the right direction.just remember that your thinking influences everything.the way you percieve things actually makes you thing in that direction.controlling your emotions,your behaviour,self-introspection can always help in ugly situations that you may come across in life.is your mind easily distracted into different things?are you finding it difficult to concentrate on one particular thing that you are doing?')
          pred=clf.predict(selector.transform(vectorizer.transform([review_to_words(raw_input())]).toarray()))
          if pred=='0':
             print('it is a difficult task but you should start noticing where your mind goes and bring it back to the present moment.maybe you go to the past and thing of lot of unnecessary things which will continue making you feel sad.maybe you go to the future and worry about certain things based on your present.say for example i am not able to perform well,ow will i get a job.what will be my parents reaction and lots of other stuff.')
       
          if pred=='1':
            print('nice.i think you are going in the right direction.just remember that your thinking influences everything.the way you percieve things actually makes you thing in that direction.controlling your emotions,your behaviour,self-introspection can always help in ugly situations that you may come across in life.')

###3
clf=joblib.load("tr4bfclf.pkl")
vectorizer=joblib.load("tr4bfvec.pkl")
selector=joblib.load("tr4bfsel.pkl")
print('A lot of tips have now been shared .Tell me something about your workplace.How is work going on there?')
pred=clf.predict(selector.transform(vectorizer.transform([review_to_words(raw_input())]).toarray()))
if pred==0:
     print('You have to asses your work.your mindset might have a negative impact on your work.This can bring up unnecessary pressure and further worsen your condition.')
if pred==4:
    print('You are doing fine!Your mindset plays a very important part in you work,negativity in your mind can bring up unnecessary pressure and this affect performace which induces more negativity.')
if pred==2:
    print('You don not seem to be doing that well.Your mindset plays a very important part in you work,negativity in your mind can bring up unnecessary pressure and this affect performace which induces more negativity.')


clf=joblib.load("tr2bfclf.pkl")
vectorizer=joblib.load("tr2bfvec.pkl")
selector=joblib.load("tr2bfsel.pkl")
print('well you know there is a homework part to follow self intorospecting is a big part of CBT theory.Go back and look back into our conversation the areas you think you need to change.the idea is that the client identifies their own unhelpful beliefs and prooves them wrong as a result their beliefs begin to change.For example someone is anxious in social situation maybe setup to a homework assignment to meet a friend at a pub and have a drink.Are you okay with this?')
pred=clf.predict(selector.transform(vectorizer.transform([review_to_words(raw_input())]).toarray()))
if pred=='1':
   print('so we have discussed a lot.it was nice talking to you.i had a great experience.so how are feeling now?better?')
   raw_input()
   print('i hope you continue feeling well.you should emphasize on the points that i told you and it will certainly helpyou in the future.you should stop thinking and going in the past too often.life is all about the present,i think.am i right or am i right?haa.good bye and come back over whenever you want to')
if pred=='0':
   print('Well at least give it a try.so we have discussed a lot.it was nice talking to you.i had a great experience.so how are feeling now?better?')
   raw_input()
   print('i hope you continue feeling well.you should emphasize on the points that i told you and it will certainly helpyou in the future.you should stop thinking and going in the past too often.life is all about the present,i think.am i right or am i right?haa.good bye and come back over whenever you want to')



