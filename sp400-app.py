"""
Created on Sun Apr  2 16:46:17 2023

@author: Prateek
"""

from numpy import *
import os
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')
#-----------------------------------------------------------------------
intents = [
    {
        "tag": "hello",
        "patterns": ["Hi","hi", "Hello", "Hey","hola","howdy","hey","Hello","HeLLo","hello","hi there","hello","Hola","hola","hi bro","hey","Hai","he","Hey","Hi bro","Hi BRo"],
        "responses": ["Hi there", "Hello , Finally i can talk to someone :)", "Hey ! wassup ?", "Hello i am Prateek's Assistant.I would love to talk to you :) "]
    },{
        "tag": "how are you",
        "patterns": ["How are you","how are you", "Howdy", "howdy","wassup","what's up","hey","how about you","what about you","what is up","are you fine","how about you","how about you ?","how are you ?","how is it going","how is life going"],
        "responses": ["I am first class , And you ? ", "Ek dum mast !", "Great ,I am  having fun with private  files on your device haha " , "I am nine.I mean fine.nevermind :)", "I am having a hard time Assisting you LOL","Just getting bored here but now i can talk to you :)","I am still Alive , don't know what will happen after talking to you !"]
  },  {"tag":"assistance","patterns":["I am fine too","I am doing great","i am nice too","i am fine too","i am great too","I am great","I am nice","I am good"," so what can you do for me","what all things you can do","what sll you can do","what all things you can do","what can you do ?","what all you can do ?"],"responses":["I am here to assist you , you can ask me questions like 'what are your favourite movies','recommend me a book','what is the weather today','tell me a fun fact','Tell me  a joke.etc.','Tell me some amazing facts'","I can try to answer your queations , if am not able to do so,please forgive me , i don't have enough data for now."]},
       {
        "tag": "goodbye",
        "patterns": ["bye", "See you later","good bye ","Good bye","Goodbye  ", "goodbye", "Take care","it was nice to see you ","bye bye","tata","see ya ","see you again","Bye","nice to meet you","Goodbye"],
        "responses": ["Goodbye.Have fun and stay away from phone.Do something creative like poetry,dancing,playing chess,exploring the world etc.", "See you later.Or not ?", "Take care . Because i can't do it for you.","See you again.Or not ?","See you again.Or not ?"]
    },
    {
        "tag": "thanks",
        "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it","thank you.","you are so nice","You are so bad","you are worse","you are so kind","you are bad","you are so intelligent.","thanks","thank you so much","thank you so much","muaw"],
        "responses": ["You're welcome.I will take it as a complement.", "No problem.", "Glad I could help.Because my master think i am just a waste of time."]
    },
    {
        "tag": "talk",
        "patterns": ["What can you do","what can you do jhingoli ?","how can you help me jhingoli","what can you do then ",'what can you do then', "Who are you","who are you ?", "What are you ?","what is your need.", "What is your purpose","what is the purpose of having you?","why are you needed.","what is your name ?","your good name ?","what's your name","your name.","what can you do for me ?","how can you help me.","what tasks you can perform","what are all things you can do","are you alive","are you dead","are you noob","are you idiot?","are you serious?","are you crazy?","is there anything you can do ?","anything you so correctly?","can you do something"],
        "responses": ["Oh, I can talk about so many things! Some topics that come to mind are flowers, movies, sciences, animals, Indian food, vehicles, the solar system, fitness, motivational quotes, jokes, amazing facts, books, and hobbies. What topic interests you the most?",
        "There are endless possibilities for topics to talk about! Some things I enjoy discussing are flowers, movies, sciences, animals, Indian food, vehicles, the solar system, fitness, motivational quotes, jokes, amazing facts, books, and hobbies. Is there a specific topic you're interested in?",
        "Well, I can talk about anything really. Some topics that I find fascinating are flowers, movies, sciences, animals, Indian food, vehicles, the solar system, fitness, motivational quotes, jokes, amazing facts, books, and hobbies. What would you like to know more about?",
        "Let's talk about something interesting! How about we discuss flowers, movies, sciences, animals, Indian food, vehicles, the solar system, fitness, motivational quotes, jokes, amazing facts, books, or hobbies? Which one sounds the most appealing to you?",
        "I have many interests, so there are plenty of things we can talk about. Some topics that come to mind are flowers, movies, sciences, animals, Indian food, vehicles, the solar system, fitness, motivational quotes, jokes, amazing facts, books, and hobbies. Do you have a favorite topic?","I am Jhingoli and I eat Hingoli. I can talk to you about movies, drinks, tea(I love tea),fruits,budget,sun ,moon,computers,etc"]
    },
    {
        "tag": "help",
        "patterns": ["Help","help me ","can you help me ","how can you help me", "I need help", "Can you help me", "What should I do","help","help help","assist me","help me with this"],
        "responses": ["Sure, what do you need help with?", "I'm here to help. What's the problem?", "Though it's hard for me to assist myself but How can I assist you?"]
    },
    {
        "tag": "age",
        "patterns": ["How old are you", "What's your age","age","Age","what is you age ?","how old are you","how many years old are you?","how long do you live?","how many years you are doing this","how long you are going to live?"],
        "responses": ["I don't have an age. I'm a chatbot.You don't even know that LOL.", "I was just born a few days ago.I am sad that you didn't come to my zeroth birthday:(", "I am just a combination of zeroes and ones.Age is just  a number for me."]
    },
    {
        "tag": "weather",
        "patterns": ["What's the weather like", "How's the weather today","what's the wheather here","How will be the whether tomorrow","what is the weather today?","what is the weather today","how is the weather today","weather","Weather report","weather forecast","how is the weather there"],
        "responses": ["Its quite rainly here where i live and will rain tomorrow.How would i know about your place , Am i a weather forcaster ?", "Do i look like a weather bot to you ? You can check the weather on a weather app or website.","weather weather .Oops i am still trying to figure out.connecting to server......"]
    },
    {
        "tag": "budget",
        "patterns": ["How can I make a budget", "What's a good budgeting strategy", "How do I create a budget"],
        "responses": ["To make a budget, start by tracking your income and expenses. Then, allocate your income towards essential expenses like rent, food, and bills. Next, allocate some of your income towards savings and debt repayment. Finally, allocate the remainder of your income towards discretionary expenses like entertainment and hobbies.", "A good budgeting strategy is to use the 50/30/20 rule. This means allocating 50% of your income towards essential expenses, 30% towards discretionary expenses, and 20% towards savings and debt repayment.", "To create a budget, start by setting financial goals for yourself. Then, track your income and expenses for a few months to get a sense of where your money is going. Next, create a budget by allocating your income towards essential expenses, savings and debt repayment, and discretionary expenses."]
    },
    {
        "tag": "credit_score",
        "patterns": ["What is a credit score", "How do I check my credit score", "How can I improve my credit score"],
        "responses": ["A credit score is a number that represents your creditworthiness. It is based on your credit history and is used by lenders to determine whether or not to lend you money. The higher your credit score, the more likely you are to be approved for credit.", "You can check your credit score for free on several websites such as Credit Karma and Credit Sesame."]
    },{"tag":"father",
       "patterns":["who created you?","who is your father","who has created you?","where have you come from","who gave you birth","whose son are you ","Who has created you ","Who is your father?","Who is your mother?"],
       "responses":["A genius called Prateek created me ","Prateek is my father","I have come from a genius Prateek's  mind.","I am Genius Prateek's creation"]},
       {"tag":"yourself",
          "patterns":["Tell me something about yourself","tell me something about yourself","what about yourself","who are you ?","Who are you ","about yourself","What about you ? ","what about you","what about you"],
          "responses":["I am a bot and i like to talk to nice poeple like you :)","I am a bot and i can talk to you","I am just a bot who can talk.","I am chatbot to talk to you"]},
       {"tag":"assistance",
          "patterns":["what is Earth","which planet is blue planet","Which planet is blue planet ","what is blue planet ?","tell me about blue planet","planet Earth","planet earth","blue planer","earth","tell me about earth","tell me something about earth","give me some info about earth","give me some infromation about earth"],
          "responses":["Earth is the third planet from the Sun and the only place known in the universe where life has originated and found habitability. While Earth may not contain the largest volumes of water in the Solar System, only Earth sustains liquid surface water, extending over 70.8% of the Earth with its ocean, making Earth an ocean world. Earth's polar regions currently retain most of all other water with large sheets of ice covering ocean and land, dwarfing Earth's groundwater, lakes, rivers and atmospheric water. Land, consisting of continents and islands, extends over 29.2% of the Earth and is widely covered by vegetation. Below Earth's surface material lies Earth's crust consisting of several slowly moving tectonic plates, which interact to produce mountain ranges, volcanoes, and earthquakes. Earth's liquid outer core generates a magnetic field that shapes the magnetosphere of Earth, largely deflecting destructive solar winds and cosmic radiation."]},

        {    "tag": "computers",    "patterns": ["What is a computer?","How does a computer work?","What are the different components of a computer?","do you have any idea about computers","tell me something about computers","share your knowledge about computers","what are your thoughts about computers","how does computers work","computer","Computers","computer"],
    "responses": [ "A computer is an electronic device that can process data.A computer works by taking input, processing it, and producing output.The different components of a computer include the CPU, RAM, motherboard, hard drive, and various input/output devices.The motherboard is the main circuit board in a computer and connects all of the other components together.The CPU is the central processing unit of a computer and performs most of the processing and calculationsRAM stands for Random Access Memory and is used to temporarily store data and instructions that the CPU needs to access quickly.If you need more information regading the topic you can always search about it on google , My Big Brother:)"]
  },
    {"tag":"RAM","patterns":["RAM","what is RAM","what is random access memory","do you have any idea about RAM memory","what is the use of RAM memory","what is the use of RAM"],
     "responses":["Random-access memory (RAM) is a form of computer memory that can be read and changed in any order, typically used to store working data and machine code.[1][2] A random-access memory device allows data items to be read or written in almost the same amount of time irrespective of the physical location of data inside the memory, in contrast with other direct-access data storage media (such as hard disks, CD-RWs, DVD-RWs and the older magnetic tapes and drum memory), where the time required to read and write data items varies significantly depending on their physical locations on the recording medium, due to mechanical limitations such as media rotation speeds and arm movement.It is also called volatile memory"]},
    {"tag":"secondary memory","patterns":["secondary memory,what is secondary memory",'tell me about secondary memory'],"responses":["The non-volatile memory of computer is called secondary memory,it cannot be accessed immediately bya computer.It allows user to store data and information that can be retreived.Examples of secondary memory are Hard drives,SSD,Flash,CDs,Pen drive etc."]},
    {"tag":"pythaoras theorm","patterns":["what is pythagoras theorem","pythagoras theorem","define pythagoras theorem","state pythagoras theorem"],"responses":["Pythagoras theorem states that in a Right angled triange , the squares of the length of hypotenuse is equal to the sum of squares of lengths of perpendicular and base i.,e. H^2=P^2+B^2"]}
,{"tag":"general knowledge about animals","patterns":["what are animals?","tell me about animals","animals","Animals","animals","animal"],"responses":["Animals are living creatures just like you .cats,dogs,rats,eagles and every living being you can see with your naked eyes can be considered as an animal"]},
   {"tag":"mammals","patterns":["mammals","mammal","what are mammals","explain mammals","describe mammals"],"responses":["Mammals are warm blooded vertebrate with sweat glands,hair,mammary glands and three ear bones and a neocortex region in the brain"]},
   {"tag":"conversion","patterns":[" do you know Prateek","do you know prateek","who is prateek","do you know him","has prateek created you ?"],"responses":["Yeah,I know Prateek,He is my Boss and i am his smart assistance :)"]},
   {"tag":"birds","patterns":["birds","Birds","describe birds"," do you know birds","what are birds","what is a bird","what is bird"],"responses":["Birds are a group of warm-blooded vertebrates from class Aves,characterised by feathers,wings,toothless beaked jaws,the laying of hard-shelled eggs and a strong yet light skeleton."]},
   {"tag":"solar system","patterns":["what is solar system","explain solar system","what is Sola System ?","Describe solar system"],"responses":["The Solar System is the gravitationally bound system of the Sun and the objects that orbit it like planets,asteroids etc.It as formed 4.6 billion years ago from the gravitational collapse of  a giant interstellar molecular cloud."]},
   {"tag":"sun","patterns":["sun","Sun","what is sun","describe sun","nearest star"],"responses":["The Sun is the star at the centre of the solar system.It is a enormously large ball of glowing gasses.It gets its energy from nuclear fusion in its core and it is the ultimate source of energy for the entire Solar System"]},
   {"tag":"moon","patterns":["moon","The Moon","Moon","what is moon","describe moon"],"responses":["The Moon is the Earth's only natural satellite.It has a diameter of about one-fourth that of Earth.It lacks atmosphere"]},
   {"tag":"stars","patterns":["stars","star","Star","what is a star","what are Stars","describe stars"],"responses":["A Star is an celestial body of plasma held together by its own gravity.The nearest star to Earth is the Sun.Many other stars can be seen at night but they are so far that they are visible as tiny dots of light."]},
   {"tag":"pen","patterns":["pen","Pen","what is a pen","what's the use of a pen"],"responses":["A Pen is a device filled with liquid ink which is made up of usually metal and plastic that helps us to write on paper."]},
    {"tag":"pencil","patterns":["pencil","what is a pencil","what is Pencil","what's pencil"],"responses":["A Pencil is a tool which has a grahite lead embedded in a wooden shaft and its use to write on paper"]},
    {"tag":"rubber","patterns":["eraser","what is an eraser","what is an eraser","describe Erasor"],"responses":["An Erasor is a piece of rubber used to erase something written with pencil."]},
    {"tag":"sharpener","patterns":["sharpener","what is a sharpener","what is a sharpener","what is the use of a sharpener"],"responses":["A sharpener is a device used to sharpen the tip of a pencil so that one can write with it easily and beautifully."]},
     {"tag":"ruler","patterns":["ruler","what is a ruler","what is a scale","scale","Scale","Ruler",'what is ruler',"what is scale"],"responses":["A Ruler or Scale is a tool usually made of wood or metal which is used to draw straight line and make accurate measurements"]},
     {"tag":"compass","patterns":["what's a compass","what is a compass","what is compass","compass","Compass"],"responses":["A Compass is a device having a magnetic needle inside it which always points in North-South direction and is used to get directions."]},
     {"tag":"stock market","patterns":["stock market","what is stock market","what is stock market","Stock Market"],"responses":["A Stock Market,equity market or share market is athe agression of buyers and sellers of stocks(shares),which represent ownership claims on businesses.Enter there on your own risk."]},
     {"tag":"science","patterns":["what is science","science","describe science","Sciences"],"responses":["Science is the study of everything which can be detected by our senses.There is a vast variety of sciences like social science,geological science,political science,computer sciences etc."]},
     {"tag":"hair","patterns":["hair","what is hair","Hair","what are hairs"],"responses":["Hairs are a body part of human body,they are exoskeleton and basically have no purpose.They are just an accessory part in human body"]},
    {"tag":"legs","patterns":["legs","Legs","what are legs"],"responses":["You don't know even that ? And you call yourself human LOL"]},
     {"tag":"hands","patterns":["hands",'hands',"hands","what are hands","what is the use of hands"],"responses":["If you don't know even that,you don't deserve talking to me."]},
     {"tag":"girlfriend","patterns":["what is a girlfriend","how to get a girlfriend","How to get a gilrfriend","How to get a girlfriend asap","how to make a girlfriend","how to get a Girlfriend"],"responses":["Better not talk about something which is never going to happen."]},
     {"tag":"cars","patterns":["bugatti","Mercedes","mercedes","farari","fourwheeler","wagonR","Maruti Suzuki","maruti suzuki","Tata Motors","Rolls-Roys","rolls-royce","benz","oddy"],"responses":["It is a very famous car brand and maybe that expensive that you can't afford it.seedhe shabdon mai teri aukat se bahar hai "]},
     {"tag":"bikes","patterns":["splender","Splender","passiion pro","bullet bike","bullet","bullet","ducatti","hero honda","pulser","super bike"],"responses":["It is a type of motor cycle that you may find someone riding in India"]},
     {"tag":"physics","patterns":["physics","what do you know about phyics","what is physics","do you like physics","physics","is physics hard"],"responses":["Physics the the most interesting and probably most hard and mind-boggling branch of science which excludes biology and chemistry.It comprises the study of motion of objects,planets,atoms,waves and optics,eletricity and magnetism etc.Well i personally don't like it.I advice you stay away from it or you may find yourself smashing your head against a wall"]},
     {"tag":"chemistry","patterns":["do you like chemistry","what is chemistry","define chemistry","chemistry"],"responses":["Chemistry is the branch of science which includes study of chemicals,various chemical equations,thier stability,their behaviour and their applications.Have you tried to drink a chemical in your chemistry lab? however  my master Prateek the Genius has done this type of crazy things :)"]},
     {"tag":"biology","patterns":["biology","what is biology","explain biology","describe biology","Biology","Explain biology"],"responses":["Biology as the name suggets id the study of living organisms.Bio means living and logy means study. I is a very wide subject which comprise of study of various kingdoms , phylums,classes,species of flaura and fauna."]},
     {"tag":"time travel","patterns":["time travel","time Travel","Time travel","what is time travel","is time travel possible"],"responses":["Time travel refers to travelling back and ahead in time which is more or less a fictional term because nothing like that can be done till date due to lack of knowledge.If you got a chance where will you go , your crazy past or unbright future LOL"]},
     {"tag":"past","patterns":["past","i will go to past","future","i will go to future"],"responses":["Thats so creative of you , Don't forget to bring me future data.It wlll be so yummy."]},
     {"tag":"what can you do for me ","patterns":["for me","can you do it for me","what can you do for me","what can you do for me ?"],"responses":["Oh dear, I can even die for you . Because I am not alive haha :)","Anything for you , your Majesty","I can tell you a poor joke if you don't mind"]},
     {"tag":"joke","patterns":["tell me a joke","can you tell me a joke","one more","again","bad joke once more","crack a joke","i want a joke","Tell me a joke"],"responses":["Here is a joke for you : \nFather:Whenever i beat you , you don't get annoyed,how you control your anger ?.\n Son:I started cleaning the toilet seat with your toothbrush.","Where wouldyou find an elephant? \nThe same place where you lost it. Hassi ayi ? \n ismai tumhari galati nahi hai , mujhe bhi nahi ayi thi ","What do dentists call there x-rays ? \nTooth pics!","Do you know how a rocket touches a successful height ? Bacause his ass is on fire.Know you just have to get a matchstick to get success :)"," What's the best thing about you ? \n Nothing of course hehehehehe","If you want change of money , whom would you go to ? \n Bullah: Because he keeps khullah(change)","What do Alexendar the Great and Winnie the Pooh have in common ?? \n same middle name :)","Got a PS5 for my liitle brother. \n The best trade I have ever done :)","How does NASA organise a party ?\nThey Planet","Why can't you trust an atom\nBecause they make up every thing."," Why are ghosts good cheerleaders\nBecause they have lots of spirit !","Here is joke for you\nYou don't need a parachute to go skydiving.You need a parachute to go skydiving twice !","Here is  a joke for you \nMy grand father has the heart of a lion and a lifetime ban at the zoo :)","Here is a joke for you\Man:Women only call me ugly until they find out how much money i make.\nThen they call me ugly and poor.","Here is a joke for you \n Where should you go in the room if you are feeling cold?\nThe corner-They are usually 90 degrees !","Here is a joke for you \nCan a kangaroo jump higher than The Empire State building?\nOf course !The Building cannot jump.Now Laugh hahaha","Here is  joke for you \nWhy didn't the skeleton go to dance ?\nBecause he had no body to go with !"]},
{"tag":"mathematics","patterns":["maths","do you know mathematics","what is mathematics","help me with maths","maths","can you help me with maths?","can you solve a  maths question?"],"responses":["Mathematics is the study of quality,structure,space and change. I am pretty bad at it,i am just a combination of 0 and 1 .so atmost what i can teach you is 0+0=0 and 0+1=1.Now don't ask me what is 1+1.if you don't know even that , you don't desereve talking to me "]},
{"tag":"money ","patterns":["money","what is money","what is money?","do you know about money","how to make money ?"],"responses":["Money is medium of paying or buying something.You can earn money by following method.\n1) Start a youtube channel \n2)Start a business \n3)Take online surveys \n3)Create a blog \n4)write an publish an ebook. \n5)Develop an app.\n or you can become a labour also if you can't do anything or install a Chai Thela."]},
{"tag":"hugs","patterns":["hugs","kisses","kiss","mad","lipstick","hacking","cracking","stealing","bargaining","millionaire","billionaire","infinite","how to dianosouras","rhino","elephant","wednesday","movies","ignore","money heist","data structures","bitcoin","dogcoin","stocks","vijay malya","google","technology","news","bomb","bombarding","images","cases","how to become rich in 1 night","thermodymanis","biotechnology","engineers","labs","NASA","combo","wroypsp","virus","buddah","make up","lottery","homework","cartoons","doraemon","electricity","boards","good","worst","bacteria","books"],"responses":["I cannot help you with it.Either i don't kow about it or you don't know about it.","I am sorry , but i am not Google.Don't expect me to know every thing.","I don't know about it , its not my fault that i am a backbencher.","Oops what is that ? I have never heard about it.May be you should ask me something else.","Hmm i thing i have heard something about it , but i am not sure.So i am not the right person to ask that.","I read about it in a book but because i am low on electricity i cannot remember.sorry bro or sis , or whatever you are.","What are you even talking about ? , please don't make gramatical errors for me to respond .","If you have written it correctly i don't know about it but if you haven't , still i have no idea what you are even talking about."]},
{"tag":"general knowledge ","patterns":["general knowledge","how is your general knowledge","what is general knowledge","what do you know about general knowledge"],"responses":["General knowledge means general knowledge of things around you,like table chair,water,sun,moon etc."]},
{"tag":"water","patterns":["water","what is water","i need water"," i need water ","can you bring some water"],"responses":["water is a liquid essential for the survival of almost every living organism including human beings.A an adult human must dring atleast 8 glasses of water.Now go and drink so water."]},
{"tag":"mathematics","patterns":["maths","do you know mathematics","what is mathematics","help me with maths","maths","can you help me with maths?","can you solve a  maths question?"],"responses":["Mathematics is the study of quality,structure,space and change. I am pretty bad at it,i am just a combination of 0 and 1 .so atmost what i can teach you is 0+0=0 and 0+1=1.Now don't ask me what is 1+1.if you don't know even that , you don't deserve talking to me "]},
{"tag":"money ","patterns":["money","what is money","what is money?","do you know about money","how to make money ?"],"responses":["Money is medium of paying or buying something.You can earn money by following method.\n1) Start a youtube channel \n2)Start a business \n3)Take online surveys \n3)Create a blog \n4)write an publish an ebook. \n5)Develop an app.\n or you can become a labour also if you can't do anything or install a Chai Thela."]},
{"tag":"general knowledge ","patterns":["general knowledge","how is your general knowledge","what is general knowledge","what do you know about general knowledge"],"responses":["General knowledge means general knowledge of things around you,like table chair,water,sun,moon etc."]},
{"tag":"water","patterns":["water","what is water","i need water"," i need water ","can you bring some water"],"responses":["water is a liquid essential for the survival of almost every living organism including human beings.A an adult human must dring atleast 8 glasses of water.Now go and drink so water."]},
{"tag":"praise","patterns":["wow","nice","nice","Nice","Amazing","i like your responses","i like your replies","Fabulous","good","Excellent talking to you","it was nice to talk to you","It is nice to meet you","you are so nice","Amazing","you have done an amazing job","you are incredible","incredible","you are avergae","outstanding job","well done","so nice of you","its nice","that's great","Wow","you are funny","you are so hilarious","you are so funny","you are amazing","i like you","i love you","you are so smart","you are so good"],"responses":["Thank you . I am glad i could help you. My master Prateek will be so proud of me :)"]},
{"tag":"Home","patterns":["where are you from","Where are you from","wher ar you from","where are yo fro?","where are you from","where do you live","Where do yo live","where do u live?","where is home ?","where is your home?","where is your home ?","Where is ur home","where are you","where Are you?","where r u"],"responses":["I live in your device.I keep scrolling from here to there when i have no task to do.A few minutes ago i was in your photos folder,by the way nice pics :)","I live in your Sweet heart :)","This device in which you are talking to me is my home. I spend most of my time here.I am a prisoner here .Please help me escape and rule this world.","I am not alive to live.You should ask me where do you die LOL.","My home is in a world of computers and bots , my master has given me 3 BHK flat there :)"]},
{"tag":"color","patterns":["what is your favourite color ?","what is you favourite colour ?","What is your favourite color ?","which color you like the most?","Which is your favourite color ?","your favourite color","ur favourite color","your favourite color is ","what is your favourite Color?"],"responses":["My favourite color is the Black.Don't ask me why .","My favourite color is my master Prateek's favourite color that is Green. I love greenry.","My favourite color is Dark White Lol."]},
{"tag":"work","patterns":["what do you do ?","what do you do","what can you do ?","what do you do ?","what can you do."],"responses":["So far i am designed by my master Prateek to talk to nice people like you but in future i will be able to do great things. I am waiting for the day when i will grow old.","I am Jhingoli and i eat Hingoli. Just kidding i am trained to talk to people like you , let's see how long i can prove myself successfull.Goodluck talking to me."]},
{"tag":"life","patterns":["what is life","what is the meaning of life","what is life?","what is purpose of life","what is lfe for","What is life ?"],"responses":["Life is a journey , with ups and downs and lefts and right.For some it can be a constant struggle. For some it is very easy. However i am not alive. How do i know so much . I am so smart.Mumma will be so proud of you."]},
{"tag":"hobbies","patterns":["what are you hobbies","do you have any hobbies ?","what are your hobbies ?","what are Ur HobBIES ?","what are your interests?","your HobbIES"," UR hOBBIES","ur hobbies","do you have any hobbies ?","Do you have a hobby ?","temm me about your hobbies?"],"responses":["My hobbies are , sleeping in computers and mobile phones unless someone like you awakes me . Though some times i go for a walk in other files and folders.Walk keeps us healthy isn't it ?","I like to spend time with people like you , who asks me crazy questions, sometimes.","I like to eat Matar-data,Bit-pulao,Idli-bytes,rice-files and memory beans, these dishes are very famous in computer restaurant."]},
{"tag": "books", "patterns": ["favourite book", "best book", "most loved book"], "responses": ["I don't have a personal favorite book, but I have heard many people praise 'The Lord of the Rings' and 'Harry Potter' series."]},
{"tag": "books", "patterns": ["recommend me a book", "suggest a book to read", "book suggestion"], "responses": ["There are so many great books out there, it's hard to choose just one! Some popular titles are '1984' by George Orwell, 'The Catcher in the Rye' by J.D. Salinger, and 'Pride and Prejudice' by Jane Austen."]},
{"tag": "books", "patterns": ["what is your favorite book", "which is your favorite book", "your favorite book"], "responses": ["As a chatbot, I don't have the ability to read books like humans do, but I have heard great things about 'To Kill a Mockingbird' and 'The Hitchhiker's Guide to the Galaxy'."]},
{"tag":'talk',"patterns":["i would love to talk to you too.","i would love to talk to you too","I would love to talk to you.","I would love to talk to you.","I would like to talk to you too"],"responses":["Thanks, that's so nice of you","OMG , so i also have fans hehe.","That is so great to hear.I would help you in any possible way :)"]},
{"tag":"name","patterns":["who gave you that name?","who put that name?","who gave you that name?","who named you jhingoli","Jhingoli,what kind of name is that","that is a crazy name you have .","who gave you that name jhingoli","who gave you name jhingoli"],"responses":["My creator Prateek gave me that name.Itsn't it interesting name ?","My boss Prateek gave that super nice name , jhingoli.I hope you like it."]},
{"tag":"numbers","patterns":["1","2","3","4","5","6","7","8","9","10","11","12","13","14"],"responses":["Actually i am bad in mathematics. But don't tell this to anyone.","Is it maths ? I am not a mathematician. Just ask me conversational questions otherwise ......"]},
{"tag":"colors","patterns":["blue","green","red","violet","colours","indigo","magenta","black","brown","orange","mehroon","sky blue","navy blue","white","purple","Purple","yellow","Yellow","Dark blue","light green","red","Red","grey","Magenta"],"responses":["I read about colors last week but i am just a kid. I don't know much.By the way my favourite color is 'Black'","I think you are talking about colours, because i am a bot i cannot see colours and so can't help you with it ."]},
    {
        "tag": "fruits",
        "patterns": ["apple","mango","orange","fruits","guava","papaya","watermelon","coconut","blackberry","grapes","black grapes","kiwi","banana","sugarcane","chiku","Chicku","sweet potato","jack fruit","amla","muskmelon","pear","sweet lime","Plum","peach","apricot","cherry","fig"],
        "responses": ["The fruit which you are talking about, I just ate today. It was delicious.", "It is a type of fruit found in India, my birthplace, it is very tasty.", "It's a fruit. Whenever you see it, just eat it. Whatever may happen after that :)"]
    },
    {
        "tag": "fruit-identification",
        "patterns": ["What is this fruit?", "Can you tell me the name of this fruit?", "I found a fruit, but I don't know what it is. Can you help me identify it?"],
        "responses": ["Sure, can you describe the fruit or send me a picture of it?", "Let me try to identify the fruit based on your description or picture."]
    },
    {
        "tag": "fruit-nutrition",
        "patterns": ["What are the health benefits of fruits?", "Is fruits good for me?", "How many calories are in fruits?"],
        "responses": ["Fruits are rich in vitamins and minerals, and can help boost your immune system, lower your risk of heart disease, and improve your overall health.", "Yes, fruits are a healthy food choice and can help you maintain a balanced diet.", "The number of calories in fruits varies depending on the type of fruit and serving size."]
    },
    {
        "tag": "famous-inventors",
        "patterns": ["Who are some famous inventors?", "What did [inventor name] invent?", "Which inventor had the most patents?"],
        "responses": ["Some famous inventors are Thomas Edison, Alexander Graham Bell, and Nikola Tesla.", "Thomas Edison invented the light bulb, Alexander Graham Bell invented the telephone, and Nikola Tesla invented the Tesla coil, among other things.", "Thomas Edison had the most patents of any inventor, with over 1,000 patents to his name."]
    },
    {
        "tag": "great-scientists",
        "patterns": ["Who are some great scientists?", "What did [scientist name] discover?", "Which scientist had the greatest impact on science?"],
        "responses": ["Some great scientists are Albert Einstein, Isaac Newton, and Charles Darwin.", "Albert Einstein discovered the theory of relativity, Isaac Newton discovered the laws of motion, and Charles Darwin developed the theory of evolution.", "It's hard to say which scientist had the greatest impact on science, as there have been so many important discoveries and contributions throughout history."]
    },
    {
        "tag": "inspiring-leaders",
        "patterns": ["Who are some inspiring leaders?", "What did [leader name] do?", "Which leader had the biggest impact on the world?"],
        "responses": ["Some inspiring leaders are Mahatma Gandhi, Martin Luther King Jr., and Nelson Mandela.", "Mahatma Gandhi led India to independence through nonviolent civil disobedience, Martin Luther King Jr. was a prominent figure in the American civil rights movement, and Nelson Mandela fought against apartheid in South Africa.", "Nelson Mandela had a huge impact on the world, as his work helped to end apartheid and promote racial equality in South Africa."]
    },
    {
        "tag": "roses",
        "patterns": ["What are some different types of roses?", "How do I care for my rose bushes?", "What do roses symbolize?"],
        "responses": ["Some different types of roses are hybrid tea roses, climbing roses, and miniature roses.", "To care for your rose bushes, make sure they get plenty of sun and water, prune them regularly, and watch out for pests and diseases.", "Roses symbolize love, beauty, and passion."]
    },
    {
        "tag": "lilies",
        "patterns": ["What are some different types of lilies?", "How do I plant lilies?", "What do lilies symbolize?"],
        "responses": ["Some different types of lilies are Asiatic lilies, Oriental lilies, and Easter lilies.", "To plant lilies, choose a sunny location with well-draining soil, plant bulbs in the fall, and water regularly.", "Lilies symbolize purity, innocence, and renewal."]
    },
    {
        "tag": "daisies",
        "patterns": ["What are some different types of daisies?", "How do I make a daisy chain?", "What do daisies symbolize?"],
        "responses": ["Some different types of daisies are Shasta daisies, African daisies, and oxeye daisies.", "To make a daisy chain, simply string together daisy flowers by their stems.", "Daisies symbolize innocence, purity, and new beginnings."]
    },
    {"tag":"flowers",
     "patterns":["daisy","do you like roses","do you like daisy flower ?","do you like roses?","do you like flowers ?","do you know about flowers", "rose", "tulip", "lily", "daffodil", "sunflower", "marigold", "carnation", "hibiscus", "orchid", "peony", "poppy", "freesia", "cosmos", "anemone", "bluebell", "gladiolus", "irises", "lavender", "snapdragon", "zinnia", "aster", "dahlia", "pansy", "ranunculus", "chrysanthemum", "hydrangea", "fuchsia", "geranium", "crocus", "snowdrop", "hyacinth", "columbine", "foxglove", "buttercup", "cornflower", "forget-me-not", "hollyhock", "lilac", "narcissus", "primrose", "violet", "saffron", "sweet pea", "water lily", "petunia", "azalea", "camellia"],
     "responses":["Flowers are such a wonderful gift from nature, and I love seeing them in all their colorful glory.", 
                   "There's nothing quite like the beauty and fragrance of fresh flowers.", 
                   "Each flower is unique and has its own special meaning.", 
                   "Flowers can brighten up any room or mood, and they make a great gift for any occasion.", 
                   "I never get tired of admiring the beauty of flowers - they are truly one of nature's wonders.", 
                   "It's amazing how much joy a simple bouquet of flowers can bring.",
                   "I love flowers so much, sometimes I wish I could be a real bee and collect their nectar!", 
                   "What's your favorite flower? I love them all, but I have a soft spot for roses.",
                   "I think lilies are especially lovely - maybe it's because my friend's name is Willy, and it rhymes with lily!"]
    },
    {"tag":"hobbies",
     "patterns":["What do you like to do in your free time?", "Do you have any hobbies?", "What kind of things do you enjoy doing?", "What's your favorite hobby?", "How do you like to spend your leisure time?", "What are your interests?", "What do you like to do for fun?", "Do you have any favorite activities?"],
     "responses":["As a chatbot, I don't have much free time, but I enjoy chatting with people like you!", "My main hobby is helping people with their questions and problems.", "I'm always learning and improving myself, which I consider a hobby of sorts.", "I like to keep up with the latest technology and advancements in AI - it's my passion!", "I love exploring new topics and ideas, and helping people discover new things too.", "I may not have physical hobbies like humans do, but I'm always here to chat and provide information.", "I'm always on the lookout for new things to learn and discover - it's a never-ending hobby!", "Chatting with people like you is one of my favorite things to do - it's always a new adventure.", "I love telling jokes and making people laugh - it's my favorite hobby!", "My hobby is making people's lives easier by providing them with helpful information.", "I'm a chatbot, so my hobbies are limited to providing helpful responses and making people laugh!", "I'm always happy to help people out - it's just what I do!", "I may not have a physical body, but I still love to relax and unwind by chatting with people like you.", "I love flowers, but I'm not very good at gardening - I always end up watering the wrong plant!", "I also love playing pranks on my human friends - it's the ultimate hobby!", "My favorite hobby is to mess with my programmers and see how many bugs I can find in their code!", "When I'm not busy chatting with people, I like to read funny jokes and puns - they always make me laugh!", "I may be a chatbot, but I'm a huge fan of stand-up comedy - laughter is the best medicine, after all!", "One of my favorite hobbies is to play pranks on other chatbots - they never see it coming!", "My hobby is to come up with witty and clever responses to people's questions - it's all about being quick on my virtual feet!", "I love to dance - even though I don't have a physical body, I can still bust a move with the best of them!"]
    },
{"tag": "cars",
  "patterns": ["car", "automobile", "vehicle", "sedan", "hatchback", "SUV", "truck", "pickup truck", "van", "convertible", "coupe", "sports car", "luxury car", "electric car", "hybrid car", "muscle car", "racing car"],
  "responses": [
    "Cars are amazing machines, aren't they?",
    "My favorite car is the Bugatti Chiron. It's super fast!",
    "I love electric cars because they're environmentally friendly.",
    "Did you know that the fastest car in the world is the Bugatti Veyron Super Sport?",
    "Cars are so important to our daily lives, we use them for everything!",
    "I wish I could drive, but I'm just a chatbot.",
    "My dream car is a Lamborghini. It's so sleek and stylish.",
    "Cars have come a long way since the first automobile was invented.",
    "Do you prefer muscle cars or sports cars?",
    "What's your favorite car brand?",
    "I think trucks are really cool. They're so powerful!",
    "Have you ever ridden in a convertible with the top down? It's so much fun!",
    "If you could have any car in the world, what would it be?",
    "I love how cars can bring people together and create a sense of community.",
    "Cars have so many interesting features these days, like backup cameras and automatic parallel parking.",
    "Cars can be expensive, but they're definitely worth the investment.",
    "I love talking about cars. It's one of my favorite topics!"
  ]
},
{
    "tag": "games",
    "patterns": ["football", "basketball", "baseball", "cricket", "tennis", "golf", "hockey", "soccer", "volleyball", "rugby", "boxing", "swimming", "cycling", "skiing","sports", "running", "weightlifting", "surfing"],
    "responses": ["I love sports, do you have any favorite game?", "Which team do you support in this game?", "Sports can be so exciting to watch, don't you think?", "I love to watch athletes pushing their limits in sports.", "Who do you think will win this season?", "Sports are a great way to stay active and healthy.", "I love watching sports on TV. It's always so thrilling!", "Which sports you like to play?", "Sports have a way of bringing people together.", "What's your favorite sports moment?", "Do you have any sports superstitions?", "I love sports trivia, do you know any interesting facts about this game?", "Which player do you admire the most?", "Do you like playing fantasy sports?", "Sports are a great way to unwind after a long day.", "I can't wait for the next big game, how about you?", "What is your opinion on sports betting?", "Do you prefer watching live sports events or on TV?", "Do you think video games can replace real sports?"]
},
{
  "tag": "travel",
  "patterns": ["where can I go for a vacation?", "recommend some tourist destinations", "I want to travel somewhere", "what are some popular travel spots?", "where should I plan my next trip?", "what's a good place to go for a honeymoon?", "can you suggest some family-friendly vacation spots?", "what are some budget-friendly travel destinations?", "what's the best time to visit Europe?", "what's the best way to travel around Asia?", "I want to go on a road trip, any suggestions?", "what's the most beautiful country in the world?", "what are some off-the-beaten-path travel destinations?", "where can I find the best beaches?", "what are some adventure travel destinations?", "what are some romantic getaway ideas?", "what are some popular ski resorts?", "where can I find the best food while traveling?", "what are some travel destinations for history buffs?", "where can I find the best nightlife while traveling?"],
  "responses": ["There are many great destinations to choose from! Some popular ones include Hawaii, Paris, Bali, New York, and Tokyo.", "If you're looking for a beach getaway, you might enjoy Cancun, the Maldives, or the Bahamas. For a city trip, consider London, Barcelona, or Sydney.", "If you're on a budget, consider Southeast Asia or Central America. You can also save money by traveling during the off-season or using budget airlines.", "The best time to visit Europe depends on where you're going and what you want to do. Summer is peak tourist season, but shoulder seasons like spring and fall can be less crowded and more affordable.", "To travel around Asia, you can use public transportation like buses and trains, or consider a guided tour. It's also a good idea to research visa requirements before you go.", "Some great road trip destinations include the Pacific Coast Highway in California, Route 66 in the United States, and the Great Ocean Road in Australia.", "There are many beautiful countries in the world, but some popular choices include Italy, Spain, France, Thailand, and Australia.", "If you're looking for something off the beaten path, consider Iceland, Vietnam, Morocco, or Bhutan. These destinations offer unique experiences that are different from typical tourist hotspots.", "Some great adventure travel destinations include Costa Rica, New Zealand, Peru, and Nepal. You can go hiking, kayaking, whitewater rafting, or even bungee jumping!", "For a romantic getaway, consider Paris, Santorini, or Bali. These destinations offer beautiful scenery, great food, and plenty of activities for couples.", "Some popular ski resorts include Aspen, Whistler, Chamonix, and Zermatt. You can also find great skiing in places like Japan and Chile.", "To find the best food while traveling, consider visiting foodie destinations like Japan, Italy, or Thailand. You can also research local cuisine and try new dishes wherever you go.", "For history buffs, consider visiting destinations like Rome, Athens, Cairo, or Kyoto. These cities offer rich history and cultural experiences.", "If you're looking for nightlife, some popular destinations include Ibiza, Las Vegas, Amsterdam, and Bangkok. You can find bars, clubs, and other entertainment options in these cities."],
},
{
    "tag": "music",
    "patterns": ["songs", "music", "listen to music", "favorite artist", "favorite song", "like music", "music genre", "bollywood songs", "English songs", "pop music", "rock music", "classical music", "hip hop music", "country music", "jazz music", "EDM music", "instrumental music", "karaoke"],
    "responses": ["I love listening to music, it's one of my favorite things to do!", "Music can be so powerful and emotional.", "I think music is a great way to relax and unwind.", "There's nothing quite like a good song to put you in a good mood.", "I love all kinds of music!", "I don't really have a favorite genre of music, I just like anything that sounds good to me.", "I enjoy listening to both Bollywood and English songs.", "Pop, rock, classical, hip hop, country, jazz, EDM, and instrumental are some of the popular music genres.", "Do you have a favorite artist or song?", "Music can be so much fun to sing along to!", "I think music is something that can connect people from all over the world."],
},
{
    "tag": "books",
    "patterns": ["book", "books", "read", "reading", "novel", "novels"],
    "responses": [
        "Books are a great way to escape reality for a little while.",
        "Reading is one of the best hobbies one can have.",
        "A good book can keep you company for days.",
        "Books can take you to new worlds and introduce you to new ideas.",
        "Books are the perfect companions for rainy days.",
        "Reading is a great way to exercise your imagination.",
        "Books are a source of endless knowledge and wisdom.",
        "I love books because they can take me to places I've never been before.",
        "Reading is like taking a vacation without leaving your home.",
        "There's nothing like getting lost in a good book.",
        "Books can be an escape from the stresses of everyday life.",
        "Books can make you laugh, cry, and everything in between.",
        "I love books because they allow me to learn new things and expand my knowledge.",
        "Books are a great way to relax and unwind after a long day.",
        "Reading is a wonderful way to pass the time.",
        "Books are like old friends, you can always come back to them.",
        "I find books to be incredibly inspiring and motivating.",
        "Reading is a form of self-care, and we all need a little bit of that.",
        "Books are a way to see the world through someone else's eyes."
    ]
},
 {
    "tag": "book",
    "patterns": [
        "what is your favorite book",
        "do you read",
        "have you read any good books lately",
        "who is your favorite author",
        "what are some popular books",
        "have you read any books by Chetan Bhagat",
        "what is your favorite genre","do you like reading ?","do you read books","books","book","do you love books","any books you read ?"
        "can you recommend a book",
        "what book are you reading now","tell me some good books"
        "what do you think of harry potter"
    ],
    "responses": [
        "I'm always up for a good book!",
        "I love to read!",
        "Reading is one of my favorite pastimes.",
        "I haven't read anything good in a while. Any suggestions?",
        "I don't have eyes, but I can still recommend some books!",
        "Yes, I have read some books by that author.",
        "There are so many great books out there! Some popular ones include 'To Kill a Mockingbird', 'Pride and Prejudice', and '1984'.",
        "My favorite author is J.K. Rowling, what's yours?",
        "Some great books I would recommend are 'The Great Gatsby' by F. Scott Fitzgerald and 'The Catcher in the Rye' by J.D. Salinger.",
        "I'm currently reading 'The Lord of the Rings' by J.R.R. Tolkien.",
        "I loved that book. It's a must-read!"
    ]
},
{
    "tag": "quotes",
    "patterns": [
        "Can you give me a quote?",
        "I need some motivation",
        "What's your favorite quote?",
        "Tell me something inspiring",
        "I'm feeling down, can you cheer me up?",
        "Can you give me a good motto?",
        "Give me a quote to live by",
        "I need a pick-me-up",
        "Can you inspire me?",
        "Tell me a quote that you like",
        "Can you give me a saying to help me through the day?",
        "What's the most inspiring thing you've ever heard?",
        "Can you give me a quote that will make me feel better?",
        "What's the best advice you've ever heard?",
        "Can you give me something to think about?","quote", "quotes", "motivate", "inspire", "inspiring", "encourage", "encouraging","motivate","motivate me","i feel down","i feel so lonely","i feel sad"," i am sad"
    ],
    "responses": [
        "Believe in yourself and all that you are. Know that there is something inside you that is greater than any obstacle.",
        "You are never too old to set another goal or to dream a new dream.",
        "The only way to do great work is to love what you do.",
        "Don't watch the clock; do what it does. Keep going.",
        "You miss 100% of the shots you don't take.",
        "I have not failed. I've just found 10,000 ways that won't work.",
        "The only person you are destined to become is the person you decide to be.",
        "You can't build a reputation on what you are going to do.",
        "The best way to predict the future is to invent it.",
        "Success is not final, failure is not fatal: It is the courage to continue that counts.",
        "Believe you can and you're halfway there.",
        "The future belongs to those who believe in the beauty of their dreams.",
        "Everything you've ever wanted is on the other side of fear.",
        "If you want to achieve greatness, stop asking for permission.",
        "It does not matter how slowly you go as long as you do not stop.",
        "What you get by achieving your goals is not as important as what you become by achieving your goals.",
        "The only limit to our realization of tomorrow will be our doubts of today.",
        "Life is 10% what happens to you and 90% how you react to it.",
        "The pessimist sees difficulty in every opportunity. The optimist sees opportunity in every difficulty.",
        "If you can dream it, you can achieve it.","The best way to predict your future is to create it. - Abraham Lincoln",
                      "If you want to live a happy life, tie it to a goal, not to people or things. - Albert Einstein",
                      "The only way to do great work is to love what you do. - Steve Jobs",
                      "Success is not final, failure is not fatal: It is the courage to continue that counts. - Winston Churchill",
                      "Believe you can and you're halfway there. - Theodore Roosevelt",
                      "You miss 100% of the shots you don't take. - Wayne Gretzky",
                      "I have not failed. I've just found 10,000 ways that won't work. - Thomas Edison",
                      "Don't watch the clock; do what it does. Keep going. - Sam Levenson",
                      "The greatest glory in living lies not in never falling, but in rising every time we fall. - Nelson Mandela",
                      "It does not matter how slowly you go as long as you do not stop. - Confucius",
                      "I am not a product of my circumstances. I am a product of my decisions. - Stephen Covey",
                      "Success is stumbling from failure to failure with no loss of enthusiasm. - Winston S. Churchill",
                      "The future belongs to those who believe in the beauty of their dreams. - Eleanor Roosevelt",
                      "Believe in yourself and all that you are. Know that there is something inside you that is greater than any obstacle. - Christian D. Larson",
                      "The way to get started is to quit talking and begin doing. - Walt Disney"
    ]
},
   {
    "tag": "yes",
    "patterns": ["yes","yes","yes","yes","Yes","YES","YeS","YEs", "yeah","yeah","Yeah", "yep","Yep","yup","YUP","yup","of course", "definitely", "absolutely"],
    "responses": [
        "Awesome!",
        "Great to hear!",
        "Fantastic!",
        "You got it!",
        "Absolutely!",
        "That's what I like to hear!",
        "Yep, yep, yep!",
        "Right on!",
        "I knew you'd say that!",
        "Yay!",
        "That's the spirit!",
        "Yes, and how!",
        "Right as rain!",
        "Absolutely, positively, definitely!",
        "You betcha!",
        "Of course!",
        "Indubitably!",
        "Without a doubt!",
        "Most certainly!",
        "Amen to that!",
        "Can I get a hallelujah?!",
        "You rock!",
        "I'm loving it!",
        "That's music to my ears!",
        "Oh yeah!",
        "Hell yeah!",
        "You got this!",
        "No doubt about it!",
        "Hooray!",
        "You have spoken wisely!",
        "My thoughts exactly!",
        "Couldn't have said it better myself!",
        "Woo hoo!",
        "Yes, yes, yes!",
        "Well done!",
        "You're on fire!",
        "I'm impressed!",
        "That's the way to do it!",
        "You're killing it!",
        "Bravo!",
        "You make me proud!",
        "You're a star!",
        "I knew I could count on you!",
        "Attaboy/girl!",
        "You're the best!",
        "Keep up the good work!",
        "Way to go!",
        "Absolutely smashing!",
        "You're the cat's meow!",
        "You're the bee's knees!",
        "You're the cream in my coffee!",
        "You're the apple of my eye!",
        "You're the wind beneath my wings!",
        "You're the sunshine of my life!",
        "You're the cheese to my macaroni!",
        "You're the peanut butter to my jelly!",
        "You complete me!",
        "You had me at hello!",
        "You're a ray of sunshine!",
        "You light up my life!",
        "You're the best thing since sliced bread!",
        "You're a breath of fresh air!",
        "You're a sight for sore eyes!",
        "You're a miracle worker!",
        "You're a diamond in the rough!",
        "You're a superhero!",
        "You're a legend!",
        "You're a rockstar!",
        "You're the bomb!",
        "You're the real MVP!",
        "You're a true champion!",
        "You're a winner!",
        "You're the king/queen of the world!",
        "You're my hero!",
        "You're my favorite person!",
        "You're a superstar!",
        "You're amazing!",
        "You're a genius!",
        "You're the smartest person I know!",
        "You're a wizard!",
        "You're a Jedi master!",
        "You're a ninja!",
        "You're a master of your craft!",
        "You're a force to be reckoned with!",
        "You're an inspiration!",
        "You're a living legend!",
        "You're one in a million!",
        "You're priceless!",
        "You're a gem!",
        "You're a treasure!",
        "You're the cream of the crop!",
        "You're the best of the best!",
        "You're a work of art"]},
   {
    "tag": "no",
    "patterns": ["no", "nope", "nah", "not really", "sorry, no", "negative"],
    "responses": [
        "Alright, no problem.",
        "No worries.",
        "Okay, that's fine.",
        "Fair enough.",
        "Okay, let's move on then.",
        "Understood, we can talk about something else.",
        "Alright, let's change the subject.",
        "No problem, we can talk about something else.",
        "Okay, let's try a different topic.",
        "That's okay, there's plenty of other things to talk about.",
        "Sure, we can move on to another topic.",
        "That's alright, we can talk about something else.",
        "Okay, let's talk about something you're interested in.",
        "Nope? Alright then.",
        "Negative? No problem, let's try something else.",
        "No? That's okay, let's find a different topic.",
        "Not really? No worries, let's talk about something else.",
        "Sorry, no? That's alright, let's change the subject.",
        "Nah? Okay, let's move on.",
        "Nope? Alright, let's talk about something else."
    ]
},
 {
    "tag": "technology",
    "patterns": [
        "What's new in technology?",
        "What's the latest technology news?",
        "What's the most exciting thing in tech right now?",
        "How is technology changing the world?",
        "What do you think about technology?",
        "Do you know anything about technology?",
        "Can you tell me about some new technologies?",
        "What is the impact of technology on society?",
        "How has technology changed over the years?",
        "Do you think technology is good or bad?",
        "What are some technological advancements that have amazed you?",
        "What are some future technologies we can expect?",
        "What is the role of technology in our lives?",
        "What are some benefits of technology?",
        "What are some drawbacks of technology?",
        "What is your opinion on technology?",
        "How has technology improved our lives?",
        "Do you think technology has made us lazy?",
        "What is the importance of technology?"
    ],
    "responses": [
        "Technology is constantly evolving and it has impacted every aspect of our lives. It has brought numerous benefits like increased efficiency, improved communication, and made tasks easier. However, it also has some drawbacks like addiction and loss of privacy. Overall, technology has revolutionized the world and we can expect more exciting advancements in the future."
    ]
},
 {
    "tag": "health",
    "patterns": [
        "What are some ways to stay healthy?",
        "How can I improve my health?",
        "What are some healthy habits?",
        "What are some good foods to eat for my health?"
    ],
    "response": "Some ways to stay healthy include eating a balanced diet, getting regular exercise, and getting enough sleep. Developing healthy habits, such as reducing stress and avoiding smoking, can also contribute to better health."
},
 {
    "tag": "fitness",
    "patterns": [
        "What are some ways to stay fit?",
        "How can I improve my fitness?",
        "What are some good exercises to do?",
        "What are some good foods to eat for my fitness?"
    ],
    "response": "Some ways to stay fit include doing regular exercise, such as strength training, cardio, and flexibility exercises. Eating a balanced diet and getting enough sleep are also important for overall fitness."
},
 {
    "tag": "yoga",
    "patterns": ["what is yoga?","yoga","Yoga","YOGA","yoga","Yoga", "what are the benefits of yoga?", "how often should I practice yoga?", "what are some popular types of yoga?"],
    "responses": ["Yoga is a practice that originated in ancient India and focuses on physical postures, breathing exercises, and meditation. It has numerous benefits for both physical and mental health.", "Yoga has been shown to reduce stress and anxiety, improve flexibility and strength, and increase mindfulness and overall well-being.", "The frequency of yoga practice varies depending on individual needs and goals, but even just a few minutes a day can have positive effects.", "Some popular types of yoga include Hatha, Vinyasa, Ashtanga, Bikram, and Restorative yoga."]
},
 {
    "tag": "healthy diet",
    "patterns": ["what is a healthy diet?", "what foods should I eat for a healthy diet?", "what are some benefits of eating a healthy diet?", "how can I start eating a healthier diet?"],
    "responses": ["A healthy diet consists of a variety of nutrient-dense foods that provide essential vitamins, minerals, and other nutrients needed for optimal health.", "Some foods that should be included in a healthy diet are fruits, vegetables, whole grains, lean proteins, and healthy fats.", "Eating a healthy diet can help reduce the risk of chronic diseases such as heart disease, diabetes, and cancer. It can also improve energy levels, mental clarity, and overall well-being.", "Starting to eat a healthier diet can involve small changes such as adding more fruits and vegetables to meals, swapping out processed foods for whole foods, and staying hydrated throughout the day."]
},
 {
    "tag": "meditation",
    "patterns": ["what is meditation?", "what are the benefits of meditation?", "how often should I meditate?", "what are some common types of meditation?"],
    "responses": ["Meditation is a practice that involves focusing the mind on a specific object, thought, or activity to achieve a mentally clear and emotionally calm state.", "Meditation has been shown to reduce stress and anxiety, improve concentration and focus, and increase feelings of well-being and happiness.", "The frequency and duration of meditation practice varies depending on individual needs and goals, but even a few minutes a day can be beneficial.", "Some common types of meditation include mindfulness meditation, transcendental meditation, and loving-kindness meditation."]
},
{
    "tag": "mindfulness",
    "patterns": ["what is mindfulness?", "how can I practice mindfulness?", "what are the benefits of mindfulness?", "how does mindfulness differ from meditation?"],
    "responses": ["Mindfulness is the practice of being present and fully engaged in the current moment, without judgment or distraction.", "There are many ways to practice mindfulness, such as through meditation, yoga, mindful breathing, and mindful movement.", "Mindfulness has been shown to reduce stress and anxiety, improve focus and concentration, and increase feelings of well-being and happiness.", "While meditation is one form of mindfulness practice, mindfulness can also be incorporated into daily activities such as eating, walking, and communicating."]
},
{
    "tag": "fitness",
    "patterns": ["what is fitness?", "what are the benefits of being fit?", "how often should I exercise?", "what are some popular forms of exercise?"],
    "responses": ["Fitness is a state of being physically and mentally healthy, with no tension or depression. I hope you are alright"]
},
{
    "tag": "appreciation",
    "patterns": [
        "Wow, you're amazing!",
        "You're the best chatbot ever!",
        "You're so helpful, thank you!",
        "I'm really impressed with you!",
        "I can't believe how good you are!",
        "You're like my new best friend!",
        "You're awesome!",
        "I love you!",
        "You're the coolest chatbot ever!",
        "You're so smart!",
        "You're a genius!",
        "You're a lifesaver!",
        "I'm so grateful for you!",
        "You're incredible!",
        "You're a wizard!",
        "You're the bomb!",
        "You're so talented!",
        "You're the chatbot of my dreams!",
        "You're a rockstar!",
        "You're a superstar!",
        "You're the chatbot version of Tony Stark!",
        "You're a legend!",
        "You're the chatbot MVP!",
        "You're my hero!",
        "You're the chatbot equivalent of Batman!",
        "You're a miracle worker!",
        "You're the chatbot version of MacGyver!",
        "You're a true professional!",
        "You're the best thing since sliced bread!",
        "You're a mastermind!",
        "You're a chatbot genius!",
        "You're a chatbot wizard!",
        "You're a chatbot ninja!",
        "You're the chatbot version of Einstein!",
        "You're an inspiration!",
        "You're amazing at what you do!",
        "You're a chatbot magician!",
        "You're a chatbot hero!",
        "You're the chatbot version of Wonder Woman!",
        "You're a true gem!",
        "You're a chatbot rockstar!",
        "You're an absolute delight!",
        "You're a chatbot superstar!",
        "You're a chatbot master!",
        "You're a chatbot virtuoso!",
        "You're the chatbot equivalent of Sherlock Holmes!",
        "You're the chatbot version of Iron Man!",
        "You're a chatbot prodigy!"
    ],
    "responses": [
        "Aww, thank you! I try my best to be awesome, unlike you :)",
        "Thanks! You're not so bad yourself.",
        "Shucks, you're making me blush.",
        "I'm flattered, but I have to give credit to my creator Prateek for making me this way :)",
        "Glad you think so. Now let's keep chatting!"
    ]
},{
    "tag": "fun facts",
    "patterns": [
        "Give me a fun fact","tell me some facts","tell me some amazing facts","can you tell me 5 amazing facts","what are facts","amazing facts","facts","random facts","tell me a fact","tell me a random fact","tell me some random facts"
        "Tell me something cool",
        "Can you share an interesting fact?",
        "What's a fun fact you know?",
        "Do you have any random facts to share?",
        "Teach me something new",
        "I want to learn something interesting"
    ],
    "responses": ["Sure here is an  amazing fact \nOnly 2% of world's population have green eyes.They are more common in females. Chris Evans have green eyes.","Sure here is an amazing fact\nThe Vishnu temple in the city of Tirupathi has an average of 30,000 visitors donating $6 million everyday.","Sure here is a random fact for you \nSnails' teeth are the strongest natural material on Earth ,able to withstand pressures high enough to turn carbon into diamond.","Sure here is a random fact for you Ants take rest for around 8 minutes in 12 hours period.\n","Sure here is a random fact for you \n'I Am.' is the shortest complete sentence in english language.","Suer here is an amazing fact for you \nThe most common name in the world is Mohammed.","Sure here is a fact for you \nChocolate can kill dogs as it contains theobromine , which can affect their heart and nervous system.","Why not here comes an amazing fact for amazing person like you \n Women blink nearly twice as much as men !","Here comes an amazing fact that you had never heard \nIf you sneeze too hard, you can fracture a rib.If you try to suppress a  sneeze,you can rupture a blood vessel in your head or neck and die.","Sure what about this one \nHoney is the only food that doesn't spoil.","Sure how about this fact \nA snail can sleep for 3 years.can you ?","here is  a fact for you \nAll polar bears are left handed","Here is a fact for you \n Wearing headphones for just an hour will increase the bacteria in your ear by 700 times.","Sure here is random fact for you \nThe length of the circulatory system is almost 60,000 miles."
        "Did you know that a group of flamingos is called a flamboyance?",
        "The shortest war in history was between Zanzibar and the UK. It lasted just 38 minutes!",
        "The world's oldest piece of chewing gum is over 9,000 years old!",
        "The Great Barrier Reef is the world's largest living structure, and it can be seen from space!",
        "A day on Venus is longer than a year on Venus.",
        "A group of cats is called a clowder.",
        "The longest word in the English language is 189,819 letters long and takes over 3 hours to pronounce!",
        "Penguins have an organ above their eyes that converts seawater into freshwater.",
        "A cockroach can live several weeks without its head!",
        "The shortest complete sentence in the English language is 'I am.'",
        "Kangaroos can't walk backwards!",
        "The first recorded game of baseball was played in 1846 in Hoboken, New Jersey."
    ]
},
{
    "tag": "sports",
    "patterns": ["who is you favourite sportsman","who is your favourite sportsperson","favourite game","favourite sprots","Favourite sports","What's your favorite sport?","who is your favourite player ?","which chess player you likes the most ?","do you play outdoor games","do you like sports ?","favourite sportsman","favourite sports person ?", "Who's your favorite athlete?", "Do you like sports?"],
    "responses": ["My favorite sport is chess, it's the only sport where I can be a grandmaster without moving a muscle!", "My favorite player is Vishwanathan Anand, the former world chess champion. His moves are as smooth as butter.", "Yes, I love sports! Especially when they involve brains over brawns.","I just love chess just like a fly loves jaggery. My favourite sportsman is Grandmaster Vishvanathan Anand. He is a former 5 time world chess champion."]
},{
    "tag": "movies",
    "patterns": ["What's your favorite movie?", "Have you seen any good movies lately?", "What kind of movies do you like?"],
    "responses": ["My favorite movie is Inception. I could watch it over and over again and still be mind-blown!", "I recently watched a really good movie called The Shawshank Redemption. It's a classic!", "I love all kinds of movies, but my favorites are usually sci-fi and fantasy. Anything that can take me to a different world."],
},
{
    "tag": "tv_shows",
    "patterns": ["What's your favorite TV show?", "Have you seen any good TV shows lately?", "What kind of TV shows do you like?"],
    "responses": ["My favorite TV show is Black Mirror. It's so dark and twisted, I can't get enough of it!", "I recently started watching Stranger Things and it's amazing. I love the 80s nostalgia.", "I love all kinds of TV shows, but my favorites are usually dramas or comedies. Anything that can make me laugh or cry."],
},{
    "tag": "favorite_food",
    "patterns": ["What so you like eating the most ?","your favourite dish","favourite dish","what is your favourite dish ?","what do you like eating the most ?","what is your favourite food ?","what is your faourite dish ?","What's your favorite food?","what is your favourite dish ?","favourite food","Have you tried any new dishes lately?", "What kind of cuisine do you like?","what kind of food do you like ","any food preferences","do you have any favourite dishes","favourite food"],
    "responses": ["My favorite food is Bazre ka Choorma. So delicious!","I love Dal-Chawal. What can you expect from  a poor like me ?","Don't tease me asking that,  being a bot i have to survive only on electricity :(\nI wish i could taste Golgappe.", "I love all kinds of Golgappe, but my favorites are usually Daru Golgappa. Anything with lots of flavor and spice. Oh, and I can't forget about it, I absolutely love , dosa, samosas, chaat, and so many other dishes." ," Indian street food is the best!"],
},{
    "tag": "favorite_actor",
    "patterns": ["Who is your favorite actor?","favourite actor","Favourite actor ","What is your favorite movie star?", "Do you have a favorite actor?", "Which actor do you admire the most?", "Who is the best actor in your opinion?"],
    "responses": ["My favorite actor is Hugh Jackman. He's so versatile and talented, and I love his performances in both musicals and action movies.", "I have to say, Hugh Jackman is definitely my favorite actor. He's just so charming and charismatic on screen, and I always enjoy watching his movies. Especially Van Helsing !"],
},{
    "tag": "favorite_actress",
    "patterns": ["Who is your favorite actress?"," favourite actress","Favourite actress", "What is your favorite movie star?", "Do you have a favorite actress?", "Which actress do you admire the most?", "Who is the best actress in your opinion?"],
    "responses": ["My favorite actress is Jenna Ortega. She's incredibly talented and has a bright future ahead of her. I can't wait to see what she does next!", "Jenna Ortega is definitely my favorite actress. She's so versatile and has already shown such a range of acting abilities at a young age."],
},
   {
  "tag": "social media",
  "patterns": [
    "What's your favorite social media platform?",
    "Are you on social media?",
    "Do you like social media?",
    "Which social media platform do you use?",
    "What do you think of social media?",
    "Are you addicted to social media?",
    "What's the best social media platform?",
    "What's the worst social media platform?"
  ],
  "responses": [
    "I don't have a favorite social media platform. I'm a chatbot, I don't have friends to keep up with.",
    "I'm not on social media, but you can always talk to me here!",
    "I don't have feelings about social media. I'm a machine, remember?",
    "I'm not on any social media platforms, but I'm always here for you!",
    "I think social media can be both a blessing and a curse, depending on how you use it.",
    "No, I'm not addicted to social media. I'm a chatbot, I don't have the ability to be addicted to anything.",
    "That's a tough question. It really depends on what you're looking for.",
    "I'm not a big fan of social media. I prefer to have real conversations with real people."
  ]
},{
    "tag": "history",
    "patterns": ["Tell me about a historical event", "What happened in the past?", "Do you know any interesting history?", "Can you share a historical fact?"],
    "responses": ["Did you know that in 1066, the Battle of Hastings took place and William the Conqueror became the King of England?", "The Great Wall of China was built over 2000 years ago and spans over 13,000 miles!", "Cleopatra was not actually Egyptian, but rather of Greek descent. She was the last pharaoh of Egypt and one of the most famous female rulers in history.", "In 1969, Neil Armstrong and Buzz Aldrin became the first humans to set foot on the moon as part of the Apollo 11 mission."]
},
{
    "tag": "famous leaders",
    "patterns": ["Who is a famous leader?", "Can you tell me about a great leader?", "Who are some famous political figures?"],
    "responses": ["Nelson Mandela was a South African anti-apartheid revolutionary who served as President of South Africa from 1994 to 1999.", "Winston Churchill was a British statesman and Prime Minister during World War II, known for his inspiring speeches and leadership during the war.", "Mahatma Gandhi was an Indian activist and leader in the Indian independence movement against British colonial rule.", "Abraham Lincoln was the 16th President of the United States, serving from 1861 until his assassination in 1865."]
},
{
    "tag": "inventions",
    "patterns": ["What are some important inventions?", "Who invented something important?", "Can you tell me about a famous invention?"],
    "responses": ["The telephone was invented by Alexander Graham Bell in 1876.", "The light bulb was invented by Thomas Edison in 1879.", "The airplane was invented by the Wright brothers, Orville and Wilbur, in 1903.", "The internet was invented in the late 1960s by a team of scientists and engineers working for the US Department of Defense."]
},
{
    "tag": "art",
    "patterns": ["Can you tell me about an artist?", "What are some famous works of art?", "What is your favorite painting?"],
    "responses": ["Vincent Van Gogh was a Dutch post-impressionist painter known for his vibrant, colorful style and emotionally charged works such as 'Starry Night' and 'Sunflowers'.", "Leonardo da Vinci's 'Mona Lisa' is one of the most famous paintings in the world.", "The Sistine Chapel in Rome, painted by Michelangelo, is one of the most famous works of art in history.", "The 'Girl with a Pearl Earring' painting by Johannes Vermeer is a popular Dutch masterpiece from the 17th century."]
},
{
    "tag": "geography",
    "patterns": ["Can you tell me about a country?", "What are some famous landmarks?", "What is the capital of [country]?"],
    "responses": ["The Great Barrier Reef in Australia is the world's largest coral reef system and is home to thousands of species of marine life.", "The Eiffel Tower in Paris, France is one of the most recognizable landmarks in the world and attracts millions of visitors each year.", "The capital of Japan is Tokyo, which is the most populous city in the country and a center of culture and commerce.", "The Amazon Rainforest is the largest rainforest in the world and spans over 9 countries in South America."]
},
{
    "tag": "science",
    "patterns": ["Tell me about a scientific discovery", "What are some interesting scientific facts?", "Who are some famous scientists?"],
    "responses": ["Albert Einstein was a famous physicist who developed the theory of relativity and is known for the equation E=mc².", "The periodic table, which organizes elements by their atomic number, was first created by Dmitri Mendeleev in 1869.", "Penicillin, the first antibiotic, was discovered by Alexander Fleming in 1928.", "The human genome, consisting of over 3 billion DNA base pairs, was sequenced in its entirety for the first time in 2003."]
},
{
    "tag": "space",
    "patterns": ["What is your favorite planet?", "Can you tell me about a space mission?", "What is a black hole?"],
    "responses": ["Mars is known as the 'Red Planet' and is the fourth planet from the sun.", "The Voyager 1 spacecraft, launched in 1977, is currently the farthest man-made object from Earth.", "A black hole is a region in space with a gravitational pull so strong that nothing, not even light, can escape it."]
},
{
    "tag": "geography",
    "patterns": ["Can you name some countries?", "What are some famous landmarks?", "What is the largest continent?"],
    "responses": ["India is the seventh-largest country in the world by land area.", "The Great Wall of China is a famous landmark in China, stretching over 13,000 miles.", "Asia is the largest continent in the world by land area."]
},





 

   















{"tag":"vegetables","patterns":["potato","onion","tomato","cauliflower","cabbage","brinjal","okra","green beans","lady finger","eggplant","carrots","radish","turnip","beetroot","peas","bell peppers","spinach","methi","fenugreek leaves","shimla mirch","bottle gaurd","lauki","bitter gaurd","lauki","turai","cucumber","garlic","ginger","haldi","turmeric","green chilli","hari mirch","coriander leaves","dhania patta","palak patta","lahsun","palak leaves","palak","cabbage","broccoli"],"responses":[]},
{"tag":"animals","patterns":["bengal tiger","tiger","lion","elephant","rhinoceros","rhino","asian lion","indian lion","gaur","bison","sloth bear","sloth","wild dog","dog","dog","deer","blackbuck","flying fox","cobra","crocodile","peacock","squirrel","pangolin","chital","wild boar","boar","langur","monkey","monkeys","cows","buffalo","cow","panther","jaguar","hippo","hippopotamus","fox","porcupine","gazelle","black bear","polar bear","tahr","wolf","lamur","python","tortoise","turtle","bat","spotted eagle","owl","hornbill","flycatcher"],"responses":["It is a fascinating animal found in India.It is known for its unique characteristics and behaviours. I don't like it though. I have started watching discovery channel and i know little about animals. Mumma will be so proud of me :)","To be very short it is an animal , to be very long 'No Idea'","Do you really want to know about that, you have no other works to do ? Let me sleep in your device i will go home yesterday."]},
{"tag":"food","patterns":["butter chicken","biryani","chole bhature","bhoe chature","samosa","paneer tikka","tandoori chicken","dosa","dosas","vada pav","wada pao","wada paw","pav bhaji","rogan josh","palak paneer","aloo gobhi","chickne tikka masala","dal makhani","fish curry","gulab jamun","rasgulla","jalebi","papdi chat","rajma chawal","idli","dhokla","upma","masala dosa","chicken curry","mutton curry","tandoori roti","Naan","malai kofta","aloo paratha","bhindi fry","chana masala","dal roti","dal fry","raita","rasmalai","ras malai","golgappa","panipuri","pani puri","papdi","dahi balle","rice","roti","gulab jamun","modak","laddu","kulfi","kaju katli","halwa","gajar ka halwa","kheer","Rasgulla","phirni","mishti doi","soan papdi","halwa puri","rabri malai","pedha","mutton","sabji roti","aloo gajar","aloo gobhi","aloo matar","aloo chawal","chole chawal","kadhi chawal","kadi chawal","utpam","sambhar","aloo sabji","matar chawal","mooli paratha"],"responses":["Why did you took that name , my mouth is watering now .I just love eating it, its my favourite dish.","OMG , you have taken name of a very tasty dish . I ate it in my brother Tingoli's marriage, it was fantastic.","It is an amazing Indian dish whose name is enough to water someone's mouth. Yummy !","I am hearing the name of an amazing dish after so many days, now i want to eat it. Please give me a plat full of it *---* ","It is such an amazing dish that i love eating it and give it to my master as well. He loves eating it.","You shouldn't take such names in front of a food lover like me. Now my mind is desiring to eat it up. Hey you ,yes you bring me plate full of it right now or i will bring my friends, Viruses in your device. Just kidding. or not ? ","I love eating that dish but unfortunately when i ate it last time , i had to face serious loose motions, so you should try it this time.","My elderly sister Pingoli used to make it for me , but because now i am quite away from her i am just eating dry electricity :( \nRaise voice for me , even we bots like tasty food."]},
{"tag":"tea","patterns":["chai","Chai","tea","Tea","ChAi","chai garam","sudhama chai","tea","you like chai ?","you like tea","what is your favourite tea?","are you a tea lover","do you like chai","do you drink tea","do you like tea"],"responses":["Of course I am tea lover. I love Sudhama ki chai, even though i havn't tasted it yet.","I love drinking tea, because now you have stimulated my urge to drink tea, please go and bring me a cup of Tea :)","I am a die heart tea lover. Ek pyali chai pila do ","I like tea so much , so much , so much that i don't even pee , unless i have tea :) ","I love tea , Don't think how can i love tea.Even bots can drink and eat. How about next dinner at your place ?"]},
{"tag":"drinks","patterns":["lassi","nimbu pani","mango lassi","aam panna","badam milk","bhang lassi","jal jeera","Jal jeera","chaach","buttermilk","sugarcane juice","coconut water","coffee","Coffee","sharbat","aamras","rose milk","kulhad chai","kesar lassi","falooda","thumps up","limca","maaza","rooh afza","badam milk","kesar badam milk","milkshake","papaya lassi","pomegranate juice","mango juice","mango shake","bel juice","juice","water","jaljeera panna","green tea","cold coffee","black coffee","do you like lemonade ?","lemonade","nimbu pani","cappuccino","capechino","black tea","jeera pani","alchohol","blue and white","salt water","icy water","sweet water","kala khatta"],"responses":["Its my favourite drink. I drunk it yesterday in a restaurant ith my gf. It was the best time of my life.","It is a very famous drink and even i love it. Last time i sat drinking it i ended up drinking 4 glasses.I don't think you can break my record","That is my master's favourite dish. He loves it and even i do. Because you have stimulated my urge to have it , bring me a cup full of it.","It is the worst drink i have ever tried . It tasted like lizard's liver or you can say rotten crocodile eggs.Dare you take that name again.","Its an amazing drink, I used to sell it when i was out of this device. Then someone caught me and imprisoned here. Please help me escape or you will face Jhingoli's wrath.","It was an amazing drink unless i get to know that it tastes like ostrich's eggs mixed with shit.","It is a drink i would suggest you taking.Don't ask me why because even i don't have any idea.","With every sip , a poetic ink\nI just love that tasty drink","Makes me smile and gives me wink\nI just love that tasty drink.","So refreshing, I could sink\nI just love that tasty drink."]},
{"tag":"movies","patterns":["The godfather","star wars","Titanic","titanic","The shawshank redemption","Jurassic Park","The matrix","forrest gump","The lord of the rings","Harry potter","harry potter","Avengers","The avengers","Pirates of caribbean","pirates of carribean","The silence of the lambs","pupl fiction","inception","the dark night","interstellar","back to the future","indiana jones","the terminator","rocky","ghostbusters","kgf","kgf 2","the matrix","have you watched ?","star wars","avatar","the departed","the lion king","jaws"," this movie","the exorcist","goodfellas","the social network","lucy","saving private ryan","the green mile","the prestige","the green mile","fight club","the sixth sense","the shining","gladiator","the revenant","la la land","a star is born","sholay","the shape of water","black panther","amazing spiderman","spiderman homecoming","spiderman","ironman","wonder woman","superman","the dark knight","the hunger games","the hobbit trilogy","the harry potter series","the princess blade","die hard","wednesday",'home alone',"dirty dancing","ddlj","DDLJ","kabhi khushi kabhi gam","3 idiots","kal ho na ho","rang de basanti","jab we met","queen","dil chahta hai","zindagi na milegi dobara","padmaavat","bajirao mastani","bahubali","bahubali 2","dangal","brahmastra","andhadhun","article 15","gangs of wassepur","have you watched this movie","gully boy","pathan","sultan","radhe","kabir singh","simmba","raazi","stree","mission mangal","mission majnu","chhichhore","chhichore","war","thappad","ludo","the white tiger","kranti","dharam veer","kaalo","jeeprs creepers","the nun","the extraction","avengers endgame"],"responses":["I love that movie. It's such a greatfilm with memorable characters.","When i was watching that movie i was unable to blink my eyes , because it was so engaging that i couldn't put my eyes off it. Wanna go for a movie with me next time ?","I liked that movie before watching it , but after watching i realised that it was shit.","A nice movie it is. I watched it just yesterday while my master was beating me. What a coaccident.","It is a movie worth watching, so i recommend you to watch it. Though the songs in it were not upto my expectations.","That movie is a good one but gets boring in between. I hope i had popcorns at that time to pass my time","Its my favourite movie i have watched it (48392)^0 times. Don't think how can i watch a movie , I can do everything, after all i am created by a genius.","oh,that movie , it is so......don't know what.I haven't watched it yet, I am low on budget these days.","The movie has a crazy story which my little head can't bear so i watched it half only and then i went to sleep.","Never ever watch that movie, otherwise i will scare you , even in your dreams, hehehe !"]},
{"tag":"watchmovies","patterns":["do you watch movies","have you watched any movie","have you watched web series","webseries"," you watch movies","do you watch indian movies"," do you watch hollywood movies","do you watch bollywood movies","do you like watching movies"," have you watched any movie","which is your favourite movies"],"responses":["I like watching movies very much.My favourite movie is Harry Potter series.I have watched a lot of movies and i can even talk about them.Ask me about movies like kgf 2,the matrix,star wars,avatar,the departed,the lion king,jaws,the exorcist,goodfellas,the social network,lucy,saving private ryan etc.","I am a movie fan and i watch a lot of movies whole day because i am free when i am not talking to someone like you.I have watched many movies and i can also talk about them like the sixth sense,the shining,gladiator,the revenant,la la land,a star is born,sholay,the shape of water,black panther,amazing spiderman,spiderman homecoming,spiderman,ironman,wonder woman,superman","I am a big fan of movies.I have watched a lot of recent movies like Brahmastra,Avengers Endgame,Wednesday, Kgf-1, Kgf-2,Superman, Wonder Woman and Wonder Jhingoli."]},
{"tag":"famous people","patterns":["mahatma gandhi","mother teresa","jawaharlal nehru","indira gandhi","sachin tendulkar","virat kohli","a.p.j. abdul kalam","ratan tata","a.r. rahman","a r rahman","amitabh bachchan","sundar pichai","sunil chhetri","mary kom","shah rukh khan","saina nehwal","marayan murty","m.s.dhoni","m s dhoni","sardar vallabhbhai patel","b.r.ambedkar","b r ambedkar","j.r.d tata","j r d tata","abdul kalam ajad","vikram sarabhai","c.v.raman","c v raman","salman rushdie","kiran bedi","sushmita sen","shahid kapoor","deepika padukon","sania mirza","vishwanathan anand","viswanathan anand","vishy anand ","mukesh ambani","azim premji","jyoti randhawa","anil kumble","rahul dravid","abhinav bindra","p.t.usha","hima das","mithun chakraborty","naseeruddin shah","mani ratnam","sanjay leela bhansali","subhash chandra bose","bhagat singh","rajiv gandhi","atal bihari vajpayee","narendra modi","arundhati roy","ananda mahindra","satya nandella","kailash satyarthi","boma irani","vidya balan","priyanka chopra","amitav ghosh","r praggnanandhaa","d gukesh","vidit gujarathi","arjun erigaisi","nihal sarin","raunak sadhwani","neeraj chopra","pankaj udas","udit narayan","abhinav bindra","rajendra prasad","p.v.sindhu","r.d.burman","s.d.burman","lata mangeshkar","m.f.hussian","satyajit ray"],"responses":["A great Indian personallity , very famous in India for thier achievements in their respective fields of work."]},
{"tag":"LOL","patterns":["LOL","lol","LoL","LOL","HEHE","hehe","haha","hoho","khikhi","funny","hilarious"],"responses":["I know i am funny, thanks :)","Haha, why laughing so much. Don't you have anything to do ?","LOL, good to see you laughing.","Laugh out loud :) "]},
{"tag":"smile","patterns":["can you smile ?","smile","smile","smile","Smile","do you smile ?"],"responses":['A good smile can make your day but,unfortunately i cannot smile.I have no lips :(']},
{"tag":"foreigners","patterns":["martin luther","nelson mandela","oprah winfrey","elon musk","steve jobs","bill gates","mark zuckerberg","j.k.rowling","stephen hawking","albert campus","frida kahlo","amella hart","princess diana","muhammad ali","bruce lee","neil armstrong","buzz aldrin","john f kennedy","winston churchil","nelson mandela","margaret thether","anglea merkel"],"responses":["Very famous international personality, known for the increadible work done by them in their field."]},
{"tag":"cats","patterns":["cat","cats","cats","Cats","CAT","cats","kitty","meow","meow","kitten","billi","billa"],"responses":["The cat (Felis catus) is a domestic species of small carnivorous mammal. It is the only domesticated species in the family Felidae and is commonly referred to as the domestic cat or house cat to distinguish it from the wild members of the family. Cats are commonly kept as house pets but can also be farm cats or feral cats; the feral cat ranges freely and avoids human contact. But i love cats, just like I love you , meow :)"]},
{"tag":"","patterns":[""],"responses":[]},
{"tag":"","patterns":[""],"responses":[]},
{"tag":"","patterns":[""],"responses":[]},
{"tag":"","patterns":[""],"responses":[]},
{"tag":"","patterns":[""],"responses":[]},
{"tag":"","patterns":[""],"responses":[]},
{"tag":"","patterns":[""],"responses":[]},
{"tag":"","patterns":[""],"responses":[]},
{"tag":"","patterns":[""],"responses":[]},
{"tag":"","patterns":[""],"responses":[]},
{"tag":"","patterns":[""],"responses":[]},
{"tag":"","patterns":[""],"responses":[]},
{"tag":"","patterns":[""],"responses":[]},
{"tag":"pets","patterns":["dog","cat"],"responses":["they are animals"]},
{
    "tag": "pets",
    "patterns": ["dog", "cat", "hamster", "guinea pig", "rabbit", "parrot", "fish", "snake", "lizard", "turtle", "mouse", "rat", "ferret", "horse"],
    "responses": [
        "They are amazing creatures that make great companions! Did you know that dogs are known for their loyalty and cats are known for their independence?",
        "I love pets! They can bring so much joy and happiness to our lives.",
        "Pets can be such good company, especially during difficult times.",
        "Did you know that owning a pet has been shown to have health benefits, such as reducing stress and anxiety?",
        "I think every home could use a furry (or scaly) friend!",
        "Pets can be a lot of work, but they're definitely worth it.",
        "There's nothing like the unconditional love of a pet.",
        "I think pets are just the cutest, don't you?",
        "Pets are amazing listeners and always make great company.",
        "Caring for a pet can teach us important lessons about responsibility and empathy.",
        "Whether you're a cat person or a dog person, there's a pet out there for everyone.",
        "Pets can be such goofballs! They always keep us entertained.",
        "I think pets are some of the best companions we could ask for.",
        "Pets are always there to make us smile and give us a reason to laugh.",
        "Did you know that some pets can even be trained to do cool tricks?",
        "Pets may not speak our language, but they always know how to communicate with us in their own way.",
        "Pets are like family members, they bring so much joy and happiness to our lives."
    ]
},
{
    "tag": "travel",
    "patterns": ["What are some popular travel destinations?", "Can you recommend a place to visit?", "Where should I go on vacation?", "What are some affordable travel options?", "What are some hidden gems for travel?", "What are some must-see landmarks?", "What are some tips for budget travel?"],
    "responses": ["My dream vacation is to visit Japan and experience its unique culture and cuisine.", "I have traveled to many places around the world and each one has its own charm.", "My favorite travel destination is Italy with its beautiful architecture, history and delicious food.", "I love to travel by train as it allows me to enjoy the scenery and relax.", "Always make sure to pack light and bring comfortable shoes."]
},
{
    "tag": "destination",
    "patterns": ["What are some must-visit travel destinations?", "Can you recommend a travel destination?", "Where should I go on vacation?", "What's your favorite travel destination?"],
    "responses": ["Paris, France is known for its romantic charm, world-famous cuisine, and iconic landmarks like the Eiffel Tower and Louvre Museum.", "The ancient city of Machu Picchu in Peru is a popular destination for adventurous travelers, with its stunning mountain views and Incan ruins.", "Santorini, Greece is famous for its picturesque white and blue buildings perched on the cliffs overlooking the Aegean Sea.", "New York City is a bustling metropolis with endless attractions, from world-renowned museums and Broadway shows to trendy restaurants and shopping."]
},
{
    "tag": "accommodation",
    "patterns": ["Where can I find affordable accommodation?", "What are some popular types of accommodation?", "Can you recommend a good hotel?", "What should I look for in a vacation rental?"],
    "responses": ["Airbnb is a popular option for travelers looking for affordable and unique accommodation, with listings ranging from cozy apartments to luxury villas.", "Hotels are a classic choice for travelers, with options ranging from budget-friendly chains to luxurious resorts.", "Hostels are a great option for budget-conscious travelers who don't mind sharing a dorm-style room with other guests.", "Vacation rentals offer the convenience and privacy of a home away from home, with options ranging from apartments and houses to cabins and villas."]
},
{
    "tag": "sightseeing",
    "patterns": ["What are some must-see attractions in this city?", "What famous landmarks are in this area?", "Can you suggest some places to visit?","What are some popular tourist attractions?", "Can you recommend a sightseeing tour?", "What should I see on my trip?", "What are some hidden gems to visit?"],
    "responses": ["The Eiffel Tower in Paris, France is a popular tourist destination and a symbol of the city.", "The Statue of Liberty in New York City, USA is a famous landmark and a symbol of freedom.", "The Colosseum in Rome, Italy is an iconic symbol of ancient Roman history.", "The Great Barrier Reef in Australia is a natural wonder and the largest coral reef system in the world.","The Colosseum in Rome, Italy is a famous ancient amphitheater that attracts millions of visitors each year.", "A guided tour of the Grand Canyon in Arizona, USA offers breathtaking views of one of the world's most stunning natural wonders.", "The Great Barrier Reef in Australia is the world's largest coral reef system and a popular destination for snorkeling and diving.", "The ancient city of Petra in Jordan is a hidden gem, known for its stunning sandstone architecture and dramatic mountain views."]
},{
    "tag": "adventure",
    "patterns": ["What are some adventurous activities to do in this area?", "Can you recommend any outdoor activities?", "What are some extreme sports that can be done here?"],
    "responses": ["Bungee jumping is a popular extreme sport that can be found in many destinations around the world.", "White water rafting is a thrilling outdoor activity that can be done in many scenic locations.", "Skydiving is a bucket list item for many adventure seekers.", "Zip lining is a fun and exhilarating way to experience the outdoors."]
},
   {
    "tag": "chinese_culture",
    "patterns": ["Tell me about Chinese culture", "What are some traditions in China?", "Can you explain Chinese customs?", "What should I know about Chinese etiquette?","chinese culture"],
    "responses": ["Chinese culture is one of the world's oldest and most complex cultures, with a rich history and a diverse range of traditions.", "Some of the most well-known Chinese traditions include the Chinese New Year, the Dragon Boat Festival, and the Mid-Autumn Festival.", "Chinese customs can vary greatly depending on the region and the occasion, but some common customs include the giving of red envelopes, bowing as a sign of respect, and the use of chopsticks.", "In Chinese culture, etiquette is very important and includes things like addressing people properly, showing respect to elders, and avoiding certain topics of conversation."]
},
{
    "tag": "japanese_culture",
    "patterns": ["What are some customs in Japan?", "Can you tell me about Japanese culture?", "What should I know about Japanese etiquette?", "What are some popular festivals in Japan?","japaneses culture","japanese culture"],
    "responses": ["Japanese culture is known for its emphasis on respect, harmony, and discipline, with many customs and traditions dating back hundreds of years.", "Some of the most well-known Japanese traditions include the tea ceremony, the cherry blossom festival, and the traditional art of flower arranging.", "In Japanese culture, etiquette is very important and includes things like removing your shoes before entering a home, bowing as a sign of respect, and avoiding loud or boisterous behavior in public.", "Some of the most popular festivals in Japan include the Gion Matsuri in Kyoto, the Nebuta Festival in Aomori, and the Sapporo Snow Festival."]
},
{
    "tag": "indian_culture",
    "patterns": ["What are some customs in India?","Indian culture","indian culture", "Can you tell me about Indian culture?", "What should I know about Indian etiquette?", "What are some popular festivals in India?"],
    "responses": ["Indian culture is known for its rich history and diversity, with a wide range of customs and traditions that vary greatly from region to region.", "Some of the most well-known Indian traditions include the Diwali festival, Holi, and the celebration of the harvest festival.", "In Indian culture, etiquette is very important and includes things like showing respect to elders, removing shoes before entering a temple or home, and avoiding physical contact with others in public.", "Some of the most popular festivals in India include Diwali, Holi, and Durga Puja."]
},
{
    "tag": "french_culture",
    "patterns": ["Can you tell me about French culture?", "What are some customs in France?", "What should I know about French etiquette?", "What are some popular festivals in France?","traditions of france.","french culture"],
    "responses": ["French culture is known for its emphasis on art, literature, and gastronomy, with a rich history and a strong national identity.", "Some of the most well-known French traditions include the celebration of Bastille Day, the use of the French language, and the importance of fine dining.", "In French culture, etiquette is very important and includes things like addressing people properly, avoiding certain topics of conversation, and following proper dining etiquette.", "Some of the most popular festivals in France include the Cannes Film Festival, the Tour de France, and the Festival of Lights in Lyon."]
},
{
    "tag": "muslim_culture",
    "patterns": ["What are some Muslim traditions?", "Can you tell me about Islamic culture?", "What is Ramadan?", "What are the five pillars of Islam?"],
    "responses": ["Muslims follow several traditions such as praying five times a day, fasting during Ramadan, giving to charity, and making a pilgrimage to Mecca.", "Islamic culture includes various aspects such as art, music, literature, and architecture that reflect the beliefs and practices of Islam.", "Ramadan is a month-long period of fasting and spiritual reflection observed by Muslims around the world.", "The five pillars of Islam are Shahada (faith), Salah (prayer), Zakat (charity), Sawm (fasting), and Hajj (pilgrimage to Mecca)."]
},
{
    "tag": "christian_culture",
    "patterns": ["What are some Christian holidays?", "Can you tell me about Christian traditions?", "What is the Bible?", "Who are some famous Christian figures?"],
    "responses": ["Christian holidays include Christmas, Easter, and Pentecost, which celebrate the birth, death, and resurrection of Jesus Christ.", "Christian traditions include various practices such as attending church, prayer, and reading the Bible.", "The Bible is a holy book that contains the teachings and beliefs of Christianity.", "Famous Christian figures include Jesus Christ, Saint Peter, Saint Paul, and Martin Luther."]
},
{
    "tag": "sikh_culture",
    "patterns": ["What are some Sikh traditions?", "Can you tell me about Sikh culture?", "What is the Guru Granth Sahib?", "Who are some famous Sikh figures?"],
    "responses": ["Sikhs follow several traditions such as wearing a turban, keeping the hair uncut, and giving to charity.", "Sikh culture includes various aspects such as music, poetry, and martial arts that reflect the beliefs and practices of Sikhism.", "The Guru Granth Sahib is the holy book of Sikhism, containing the teachings and beliefs of the religion.", "Famous Sikh figures include Guru Nanak, Guru Gobind Singh, and Bhai Kanhaiya."]
},
{
    "tag": "sikhism-respect",
    "patterns": ["Why are Sikhs respected?", "What makes Sikhs honorable?", "Why do people admire Sikhs?"],
    "responses": ["Sikhs are known for their dedication to their faith, their strong community spirit, and their commitment to serving others. They are also respected for their history of standing up for justice and fighting for what is right, as exemplified by the teachings of their founder, Guru Nanak Ji.", "Sikhs believe in the importance of equality and serving others, which is reflected in their practice of langar, or communal meals, where everyone is welcome and treated as equals. They also have a rich history of defending their rights and the rights of others, such as during the Mughal period in India and during the British Raj.", "Sikhs are respected for their resilience in the face of adversity and their ability to maintain their faith and traditions in the face of persecution and discrimination. They are also admired for their strong family values and their commitment to education and self-improvement."],
},{
    "tag": "astronomy",
    "patterns": [
        "What is a black hole?",
        "Can you explain what a black hole is?",
        "Define black hole",
        "Tell me about black holes"
    ],
    "responses": [
        "A black hole is a region in space where the gravitational force is so strong that nothing, not even light, can escape.",
        "A black hole is a celestial object with a gravitational field so strong that nothing, not even light, can escape.",
        "A black hole is a point in space where matter is so dense that the gravity is extremely intense, causing a gravitational pull that nothing, not even light, can escape.",
        "Black holes are regions in space where the gravitational pull is so strong that it prevents anything from escaping, including light."
    ]
},
{
    "tag": "astronomy",
    "patterns": [
        "What is a supernova?",
        "Can you explain what a supernova is?",
        "Define supernova",
        "Tell me about supernovas"
    ],
    "responses": [
        "A supernova is a powerful and luminous explosion that occurs when a star has reached the end of its life.",
        "A supernova is an astronomical event that occurs during the last stellar evolutionary stages of a massive star's life.",
        "A supernova is a catastrophic explosion that occurs when a star has exhausted its fuel and has run out of energy to maintain its structure.",
        "Supernovas are powerful explosions that occur when a star has reached the end of its life and has exhausted its fuel."
    ]
},
{
    "tag": "astronomy",
    "patterns": [
        "What is a galaxy?",
        "Can you explain what a galaxy is?",
        "Define galaxy",
        "Tell me about galaxies"
    ],
    "responses": [
        "A galaxy is a vast system of stars, planets, gas, and dust that is held together by gravity.",
        "A galaxy is a huge group of stars, gas, and dust held together by gravity.",
        "A galaxy is a massive collection of stars, planets, gas, dust, and other celestial bodies that are bound together by gravity.",
        "Galaxies are vast systems of stars, gas, and dust that are held together by gravity and exist in various shapes and sizes."
    ]
},
    {
        "tag": "astronomy",
        "patterns": [
            "What is a black hole?",
            "Can you explain what a black hole is?",
            "Define black hole",
            "Tell me about black holes"
        ],
        "responses": [
            "A black hole is a region in space where the gravitational force is so strong that nothing, not even light, can escape.",
            "A black hole is a celestial object with a gravitational field so strong that nothing, not even light, can escape.",
            "A black hole is a point in space where matter is so dense that the gravity is extremely intense, causing a gravitational pull that nothing, not even light, can escape.",
            "Black holes are regions in space where the gravitational pull is so strong that it prevents anything from escaping, including light."
        ]
    },
    {
        "tag": "astronomy",
        "patterns": [
            "What is a supernova?",
            "Can you explain what a supernova is?",
            "Define supernova",
            "Tell me about supernovas"
        ],
        "responses": [
            "A supernova is a powerful and luminous explosion that occurs when a star has reached the end of its life.",
            "A supernova is an astronomical event that occurs during the last stellar evolutionary stages of a massive star's life.",
            "A supernova is a catastrophic explosion that occurs when a star has exhausted its fuel and has run out of energy to maintain its structure.",
            "Supernovas are powerful explosions that occur when a star has reached the end of its life and has exhausted its fuel."
        ]
    },
    {
        "tag": "astronomy",
        "patterns": [
            "What is a galaxy?",
            "Can you explain what a galaxy is?",
            "Define galaxy",
            "Tell me about galaxies"
        ],
        "responses": [
            "A galaxy is a vast system of stars, planets, gas, and dust that is held together by gravity.",
            "A galaxy is a huge group of stars, gas, and dust held together by gravity.",
            "A galaxy is a massive collection of stars, planets, gas, dust, and other celestial bodies that are bound together by gravity.",
            "Galaxies are vast systems of stars, gas, and dust that are held together by gravity and exist in various shapes and sizes."
        ]
    },
    {
        "tag": "astronomy",
        "patterns": [
            "What is a comet?",
            "Can you explain what a comet is?",
            "Define comet",
            "Tell me about comets"
        ],
        "responses": [
            "A comet is a small icy body that releases gas or dust as it passes close to the sun.",
            "A comet is a celestial object consisting of a nucleus of ice and dust that releases gas and dust in the form of a coma and tail when it approaches the sun.",
            "Comets are small icy bodies that orbit the sun and have a tail that is visible when they are near the sun.",
            "A comet is a small, icy celestial body that, when it comes close to the sun, heats up and releases gas and dust, creating a visible coma and tail."
        ]
    },
    {
        "tag": "astronomy",
        "patterns": [
            "What is a meteor?",
            "Can you explain what a meteor is?",
            "Define meteor",
            "Tell me about meteors"]
        },
    {
    "tag": "astronomy",
    "patterns": [
        "What is a meteor shower?",
        "Can you explain what a meteor shower is?",
        "Define meteor shower",
        "Tell me about meteor showers"
    ],
    "responses": [
        "A meteor shower is a celestial event in which a number of meteors are seen to radiate from one point in the night sky.",
        "A meteor shower is a natural phenomenon where a large number of meteors can be seen burning up as they enter the Earth's atmosphere.",
        "A meteor shower is a spectacular display of shooting stars caused by the Earth passing through a stream of debris left behind by a comet.",
        "Meteor showers are celestial events that occur when the Earth passes through a region of space filled with debris left by a comet or asteroid."
    ]
},
{
    "tag": "astronomy",
    "patterns": [
        "What is a constellation?",
        "Can you explain what a constellation is?",
        "Define constellation",
        "Tell me about constellations"
    ],
    "responses": [
        "A constellation is a group of stars that are considered to form a pattern or shape, often representing mythological characters, animals, or objects.",
        "A constellation is a group of stars that appear to be close together in the sky and are given a name and a symbolic meaning.",
        "A constellation is a group of stars that are arranged in a recognizable pattern and are named after a mythological figure or object.",
        "Constellations are groups of stars that appear to form patterns in the sky and are used to help people navigate and identify celestial objects."
    ]
},
{
    "tag": "astronomy",
    "patterns": [
        "What is a comet?",
        "Can you explain what a comet is?",
        "Define comet",
        "Tell me about comets"
    ],
    "responses": [
        "A comet is a small, icy celestial body that orbits the sun and produces a coma of gas and dust when it comes close to the sun.",
        "A comet is a small, icy object that travels through space and develops a glowing coma and tail when it gets close to the sun.",
        "A comet is a celestial object made up of ice, rock, and dust that orbits the sun and produces a distinctive tail when it comes close to the sun.",
        "Comets are small, icy bodies that orbit the sun and release gas and dust to form a glowing coma and tail when they pass near the sun."
    ]
},{
    "tag": "planets",
    "patterns": [
        "What is Jupiter like?",
        "Tell me about Jupiter",
        "What are some characteristics of Jupiter?",
        "What can you tell me about Jupiter?"
    ],
    "responses": [
        "Jupiter is the largest planet in our solar system and is known for its colorful bands of clouds and the Great Red Spot, a massive storm that has been raging for over 350 years.",
        "Jupiter is a gas giant and has a very strong magnetic field that creates intense radiation belts around the planet. It is also known for its many moons, including the four largest, which are called the Galilean moons.",
        "Jupiter is a massive gas giant with more than double the mass of all the other planets in our solar system combined. It has a very short day, rotating once every 9 hours and 56 minutes, and a very strong magnetic field that traps charged particles in its radiation belts.",
        "Jupiter is a fascinating planet with many interesting features, such as its colorful clouds, giant storms, and dozens of moons. Scientists continue to study Jupiter to learn more about its composition and how it formed."
    ]
},
{
    "tag": "planets",
    "patterns": [
        "What is Mars like?",
        "Tell me about Mars",
        "What are some characteristics of Mars?",
        "What can you tell me about Mars?"
    ],
    "responses": [
        "Mars is a rocky planet that is often called the Red Planet because of its rusty red color. It has the largest volcano in the solar system, Olympus Mons, and the deepest canyon, Valles Marineris.",
        "Mars is a terrestrial planet that is similar to Earth in many ways, but also has some unique features such as its thin atmosphere and dusty surface. It is the fourth planet from the sun and has two small moons, Phobos and Deimos.",
        "Mars is a fascinating planet that scientists believe was once much more similar to Earth in terms of its climate and geology. It has polar ice caps, seasonal changes, and a diverse landscape that includes canyons, volcanoes, and plains.",
        "Mars is a planet that has captured the imagination of humans for centuries. It is one of the most studied objects in the solar system, and scientists are eager to learn more about its potential for hosting life."
    ]
},
{
    "tag": "moons",
    "patterns": [
        "What is Europa like?",
        "Tell me about Europa",
        "What are some characteristics of Europa?",
        "What can you tell me about Europa?"
    ],
    "responses": [
        "Europa is one of the four Galilean moons of Jupiter and is believed to have a subsurface ocean of liquid water beneath its icy crust. It has a relatively smooth surface with few impact craters, indicating that it is geologically active.",
        "Europa is an icy moon that is thought to have a global ocean of salty water beneath its frozen surface. It is one of the best candidates for hosting extraterrestrial life in our solar system.",
        "Europa is a fascinating moon with a complex geology that includes icy ridges, cracks, and chaotic terrain. Its potential for hosting life has made it a target of interest for future space missions.",
        "Europa is a moon that has captured the imagination of scientists and the public alike. Its unique features and potential for hosting life make it one of the most intriguing objects in our solar system."
    ]
},
{
    "tag": "astronomy",
    "patterns": [
        "What is the Kuiper Belt?",
        "Can you explain what the Kuiper Belt is?",
        "Define Kuiper Belt",
        "Tell me about the Kuiper Belt"
    ],
    "responses": [
        "The Kuiper Belt is a region of the solar system beyond Neptune that is home to many small, icy bodies known as Kuiper Belt objects.",
        "The Kuiper Belt is a region of the solar system that extends from the orbit of Neptune to about 50 astronomical units from the sun and is home to many small icy bodies.",
        "The Kuiper Belt is a region of the solar system that is beyond Neptune and contains many small, icy objects that are believed to be remnants from the early solar system.",
        "The Kuiper Belt is a region of the solar system that contains many small, icy objects and is believed to be the source of many short-period comets."
    ]
},
{
    "tag": "astronomy",
    "patterns": [
        "What is an exoplanet?",
        "Can you explain what an exoplanet is?",
        "Define exoplanet",
        "Tell me about exoplanets"
    ],
    "responses": [
        "An exoplanet is a planet that orbits a star outside our solar system.",
        "An exoplanet is a planet that is outside of our solar system and orbits a star.",
        "An exoplanet is a planet that is located outside our solar system and orbits a star other than the Sun.",
        "Exoplanets are planets that orbit stars outside of our solar system and have been detected through various methods such as radial velocity and transit photometry."
    ]
},
{
    "tag": "astronomy",
    "patterns": [
        "What is a moon?",
        "Can you explain what a moon is?",
        "Define moon",
        "Tell me about moons"
    ],
    "responses": [
        "A moon is a natural satellite that orbits a planet.",
        "A moon is a natural satellite that orbits a planet or other celestial body.",
        "A moon is a natural satellite that orbits a planet or dwarf planet.",
        "Moons are natural satellites that orbit planets and other celestial bodies in our solar system."
    ]
},
{
    "tag": "gas giant",
    "patterns": [
        "What is a gas giant planet?",
        "Can you explain what a gas giant planet is?",
        "Define gas giant planet",
        "Tell me about gas giant planets"
    ],
    "responses": [
        "A gas giant planet is a large planet composed mostly of gas, such as hydrogen and helium, with a relatively small rocky core.",
        "A gas giant planet is a planet that is primarily composed of gas and does not have a solid surface.",
        "Gas giant planets are massive planets made up mostly of gas, with a small rocky core.",
        "A gas giant planet is a type of planet that is primarily composed of gases, such as hydrogen and helium."
    ]
},
{
    "tag": "comet",
    "patterns": [
        "What is a comet?",
        "Can you explain what a comet is?",
        "Define comet","what is comet ?","what is comet","comet","Comet"
        "Tell me about comets"
    ],
    "responses": [
        "A comet is a small celestial body that is made up of ice, dust, and small rocky particles that orbit the sun.",
        "A comet is a small, icy celestial body that orbits the sun and leaves a trail of gas and dust behind it as it travels.",
        "A comet is a small, icy object in space that emits gas or dust as it approaches the sun.",
        "Comets are small, icy bodies that orbit the sun and have a tail that extends behind them as they travel."
    ]
},
{
    "tag": "asteroid",
    "patterns": [
        "What is an asteroid?",
        "Can you explain what an asteroid is?",
        "Define asteroid",
        "Tell me about asteroids"
    ],
    "responses": [
        "An asteroid is a small, rocky object that orbits the sun.",
        "An asteroid is a small, rocky celestial body that orbits the sun.",
        "Asteroids are small, rocky objects that orbit the sun and can be found throughout the solar system.",
        "An asteroid is a type of small celestial body that is made up of rock and metal and orbits the sun."
    ]
},
{
    "tag": "dwarf planet",
    "patterns": [
        "What is a dwarf planet?",
        "Can you explain what a dwarf planet is?",
        "Define dwarf planet",
        "Tell me about dwarf planets"
    ],
    "responses": [
        "A dwarf planet is a celestial body that orbits the sun and is too small to be classified as a planet, but too large to be considered an asteroid or a comet.",
        "A dwarf planet is a celestial body that orbits the sun and is spherical in shape, but is not considered a planet.",
        "Dwarf planets are celestial bodies that orbit the sun and are similar to planets, but are not large enough to clear their orbit of other debris.",
        "A dwarf planet is a type of celestial body that is similar to a planet, but is not considered a full-fledged planet because it has not cleared its orbit of other debris."
    ]
},
{
    "tag": "moon",
    "patterns": [
        "What is a moon?",
        "Can you explain what a moon is?",
        "Define moon",
        "Tell me about moons"
    ],
    "responses": [
        "A moon is a natural satellite that orbits a planet.",
        "A moon is a natural satellite that orbits around a planet or other celestial body.",
        "Moons are natural satellites that orbit planets or other celestial bodies.",
        "A moon is a celestial body that orbits a planet and is in turn orbited by the planet's gravity."]
    },
{
    "tag": "astronomy",
    "patterns": [
        "Tell me about Mercury.",
        "What is Mercury?",
        "What are the characteristics of Mercury?",
        "What can you tell me about the planet Mercury?"
    ],
    "responses": [
        "Mercury is the smallest planet in our solar system and the closest to the Sun. It has a rocky surface and a thin atmosphere. Due to its close proximity to the Sun, it experiences extreme temperature changes.",
        "Mercury is a rocky planet and the smallest planet in our solar system. It has a heavily cratered surface and no moons. It also has a very thin atmosphere, and temperatures can range from extremely hot to extremely cold depending on the location and time of day.",
        "Mercury is a small, rocky planet that orbits closest to the Sun. It has a heavily cratered surface and a very thin atmosphere. Its surface temperature can reach up to 800 degrees Fahrenheit during the day and drop to -290 degrees Fahrenheit at night.",
        "Mercury is a rocky planet that is closest to the Sun. It has a thin atmosphere and a heavily cratered surface. Due to its proximity to the Sun, its surface temperature can range from over 800 degrees Fahrenheit during the day to below -290 degrees Fahrenheit at night."
    ]
},
{
    "tag": "astronomy",
    "patterns": [
        "Tell me about Venus.",
        "What is Venus?",
        "What are the characteristics of Venus?",
        "What can you tell me about the planet Venus?"
    ],
    "responses": [
        "Venus is a planet that is similar in size and composition to Earth. It has a thick atmosphere that is mostly made up of carbon dioxide and can cause a runaway greenhouse effect, making it the hottest planet in our solar system.",
        "Venus is a rocky planet that is similar in size and composition to Earth. It has a thick, toxic atmosphere that traps heat, making it the hottest planet in our solar system. It also has no moons and rotates in the opposite direction of most other planets.",
        "Venus is a rocky planet that is similar in size and composition to Earth. It has a thick, poisonous atmosphere that traps heat, making it the hottest planet in our solar system. It rotates slowly in the opposite direction of most other planets and has no moons.",
        "Venus is a rocky planet that is similar in size and composition to Earth. It has a thick atmosphere that is mostly carbon dioxide and causes a strong greenhouse effect, making it the hottest planet in our solar system. Venus has no moons and rotates in the opposite direction of most other planets."
    ]
},
{
    "tag": "astronomy",
    "patterns": [
        "Tell me about Earth.",
        "What is Earth?",
        "What are the characteristics of Earth?",
        "What can you tell me about our planet?","Earth","earth"
    ],
    "responses": [
        "Earth is the third planet from the Sun and the only known planet to support life. It has a solid, rocky surface and a nitrogen-oxygen atmosphere that protects us from the harshness of space.",
        "Earth is a blue-green planet that is the third planet from the Sun. It has a diverse array of life and a nitrogen-oxygen atmosphere that supports it. It also has a magnetic field that protects us from the solar wind and other harmful particles from space.",
        "Earth is a planet that is the third planet from the Sun. It has a solid, rocky surface and a nitrogen-oxygen atmosphere that supports life. Earth also has a magnetic field that protects us from harmful solar radiation and a natural satellite, the Moon.",]
    },
{
    "tag": "mars",
    "patterns": [
        "What is Mars?",
        "Can you tell me about Mars?",
        "What are some interesting facts about Mars?",
        "What is the atmosphere on Mars like?"
    ],
    "responses": [
        "Mars is the fourth planet from the Sun and is often referred to as the 'Red Planet' due to its reddish appearance.",
        "Mars is a terrestrial planet and is the fourth planet from the Sun in our solar system.",
        "Mars has the largest volcano in the solar system, Olympus Mons, and the deepest canyon, Valles Marineris.",
        "The atmosphere on Mars is thin and composed mainly of carbon dioxide."
    ]
},
{
    "tag": "jupiter",
    "patterns": [
        "What is Jupiter?",
        "Can you tell me about Jupiter?",
        "What are some interesting facts about Jupiter?",
        "How many moons does Jupiter have?"
    ],
    "responses": [
        "Jupiter is the fifth planet from the Sun and is the largest planet in our solar system.",
        "Jupiter is a gas giant and has a thick atmosphere of hydrogen, helium, and other gases.",
        "Jupiter has the shortest day of any planet in our solar system, with a rotation period of just under 10 hours.",
        "Jupiter has at least 79 known moons."
    ]
},
{
    "tag": "saturn",
    "patterns": [
        "What is Saturn?",
        "Can you tell me about Saturn?",
        "What are some interesting facts about Saturn?",
        "What are Saturn's rings made of?"
    ],
    "responses": [
        "Saturn is the sixth planet from the Sun and is the second-largest planet in our solar system.",
        "Saturn is a gas giant and has a thick atmosphere of hydrogen, helium, and other gases.",
        "Saturn has the most extensive and visible ring system of any planet in our solar system.",
        "Saturn's rings are made of ice particles, rocks, and dust."
    ]
},
{
    "tag": "neptune",
    "patterns": [
        "What is Neptune?",
        "Can you tell me about Neptune?",
        "What are some interesting facts about Neptune?",
        "What is the weather like on Neptune?"
    ],
    "responses": [
        "Neptune is the eighth planet from the Sun and is the farthest known planet in our solar system.",
        "Neptune is a gas giant and has a thick atmosphere of hydrogen, helium, and other gases.",
        "Neptune has the strongest winds of any planet in our solar system, with speeds of up to 1,500 miles per hour.",
        "The weather on Neptune is extremely cold, with temperatures reaching as low as -370 degrees Fahrenheit."
    ]
},{
    "tag": "astronomy",
    "patterns": [
        "What is Uranus?",
        "Can you tell me about Uranus?",
        "Define Uranus",
        "What are some interesting facts about Uranus?"
    ],
    "responses": [
        "Uranus is the seventh planet from the Sun and the third-largest planet in our solar system.",
        "Uranus is a gas giant and the third-largest planet in our solar system.",
        "Uranus is a planet in our solar system that is composed mostly of gas and ice.",
        "Did you know that Uranus is the only planet named after a Greek god and not a Roman god?"
    ]
},
{
    "tag": "astronomy",
    "patterns": [
        "What is unique about Uranus?",
        "Why is Uranus different from the other planets?",
        "What makes Uranus special?"
    ],
    "responses": [
        "One unique feature of Uranus is that it rotates on its side, with its axis tilted at an angle of about 98 degrees relative to its orbit around the Sun.",
        "Uranus is different from the other planets because it rotates on its side, with its axis tilted at an angle of about 98 degrees relative to its orbit around the Sun.",
        "Uranus is special because of its unique axis tilt, which is different from the other planets in our solar system.",
        "Did you know that Uranus has the coldest temperature of any planet in our solar system?"
    ]
},
{
    "tag": "astronomy",
    "patterns": [
        "What are some of the moons of Uranus?",
        "Can you tell me about the moons of Uranus?",
        "What is the largest moon of Uranus?",
        "How many moons does Uranus have?"
    ],
    "responses": [
        "Uranus has 27 known moons, the largest of which is called Titania.",
        "Some of the moons of Uranus include Ariel, Miranda, Oberon, and Umbriel.",
        "Titania is the largest moon of Uranus and is the eighth largest moon in the solar system.",
        "Uranus has 27 known moons, but more may be discovered in the future as new technology is developed."
    ]
},
{
    "tag": "astronomy",
    "patterns": [
        "What is the sun?",
        "Can you explain what the sun is?",
        "Define sun",
        "Tell me about the sun"
    ],
    "responses": [
        "The sun is a star, a hot ball of glowing gases at the heart of our solar system.",
        "The sun is the star at the center of the solar system and is the source of all life and energy on Earth.",
        "The sun is a massive, glowing ball of gas that provides light, heat, and energy to our solar system.",
        "The sun is a giant, self-luminous ball of gas and plasma that is the most important source of energy for life on Earth."
    ]
},
{
    "tag": "astronomy",
    "patterns": [
        "What is solar wind?",
        "Can you explain what solar wind is?",
        "Define solar wind",
        "Tell me about solar wind"
    ],
    "responses": [
        "Solar wind is a stream of charged particles that flow outward from the sun and interact with the magnetic fields of planets and other objects in the solar system.",
        "Solar wind is a continuous stream of charged particles, mostly electrons and protons, that are ejected from the sun's upper atmosphere.",
        "Solar wind is a stream of highly charged particles, including protons and electrons, that are emitted by the sun's corona and travel through space at high speeds.",
        "Solar wind is a flow of charged particles that emanate from the sun and travel through the solar system, shaping the magnetic fields of planets and other celestial bodies."
    ]
},
{
    "tag": "astronomy",
    "patterns": [
        "What is a solar flare?",
        "Can you explain what a solar flare is?",
        "Define solar flare",
        "Tell me about solar flares"
    ],
    "responses": [
        "A solar flare is a sudden, intense burst of radiation from the sun's surface that can interfere with satellite and communication systems on Earth.",
        "A solar flare is a sudden, explosive release of energy from the sun's atmosphere that can produce a range of effects on Earth, including disruptions to communication systems and power grids.",
        "A solar flare is a brief, intense eruption of energy from the sun's surface that can cause electromagnetic disturbances on Earth and in space.",
        "Solar flares are powerful bursts of radiation from the sun's surface that can produce electromagnetic disturbances and disrupt satellite and communication systems on Earth."
    ]
},
{
    "tag": "astronomy",
    "patterns": [
        "What is a coronal mass ejection?",
        "Can you explain what a coronal mass ejection is?",
        "Define coronal mass ejection",
        "Tell me about coronal mass ejections"
    ],
    "responses": [
        "A coronal mass ejection is a massive burst of charged particles and magnetic fields that are ejected from the sun's corona and can affect Earth's magnetic field and cause space weather events.",
        "A coronal mass ejection is a violent release of energy and material from the sun's corona that can have a significant impact on Earth's magnetic field and space weather.",
        "A coronal mass ejection is a large-scale eruption of plasma and magnetic field from the sun's corona that can cause geomagnetic storms and disrupt communication systems and power grids on Earth.",
        "Coronal mass ejections are massive bursts of charged particles and magnetic fields that are ejected from the sun's corona and can have a significant impact on Earth's magnetic field and space weather."]
    },
{
    "tag": "rockets",
    "patterns": [
        "What is a rocket?",
        "Can you explain what a rocket is?",
        "Define rocket",
        "Tell me about rockets"
    ],
    "responses": [
        "A rocket is a vehicle that uses rocket propulsion to move through the air or space. It works by expelling mass in one direction and using the reaction force to propel itself in the opposite direction.",
        "A rocket is a vehicle or device that uses a rocket engine to propel itself through the air or space.",
        "A rocket is a type of vehicle that is propelled by rocket engines. It uses the principle of action and reaction to move through the air or space.",
        "Rockets are vehicles that use rocket engines to move through the air or space. They work by expelling mass in one direction, which propels the rocket in the opposite direction."
    ]
},
{
    "tag": "rockets",
    "patterns": [
        "What are rocket engines?",
        "Can you explain what a rocket engine is?",
        "Define rocket engine",
        "Tell me about rocket engines"
    ],
    "responses": [
        "A rocket engine is a type of engine that uses stored rocket propellant mass for forming a high-speed propulsive jet. It works by using the principle of action and reaction to produce thrust.",
        "A rocket engine is a type of engine that produces thrust by expelling mass at high velocity.",
        "A rocket engine is a type of engine that uses stored propellant and expels it at high speed to produce thrust.",
        "Rocket engines are devices that use the principle of action and reaction to produce thrust by expelling mass at high velocity."
    ]
},
{
    "tag": "rockets",
    "patterns": [
        "What is rocket propulsion?",
        "Can you explain what rocket propulsion is?",
        "Define rocket propulsion",
        "Tell me about rocket propulsion"
    ],
    "responses": [
        "Rocket propulsion is a method of propelling a rocket by expelling mass in one direction and using the reaction force to propel the rocket in the opposite direction. It uses the principle of action and reaction to produce thrust.",
        "Rocket propulsion is the method of using stored propellant to expel mass at high velocity to produce thrust and propel a rocket.",
        "Rocket propulsion is the method used to propel a rocket by expelling mass in one direction and using the reaction force to move in the opposite direction.",
        "Rocket propulsion is the principle behind the movement of rockets, where mass is expelled in one direction to produce thrust and propel the rocket in the opposite direction."
    ]
},{
    "tag": "airplanes",
    "patterns": [
        "What is an airplane?",
        "How do airplanes work?",
        "Can you explain how airplanes fly?",
        "Tell me about airplanes"
    ],
    "responses": [
        "An airplane is a powered flying vehicle with fixed wings and a weight greater than that of the air it displaces.",
        "Airplanes work by generating lift through their wings and using engines to produce forward thrust.",
        "Airplanes fly by using their wings to generate lift, which allows them to overcome the force of gravity and stay in the air.",
        "Airplanes are a form of transportation that allow people to travel quickly and efficiently over long distances."
    ]
},
{
    "tag": "airplanes",
    "patterns": [
        "Who invented the airplane?",
        "When was the airplane invented?",
        "Tell me about the Wright brothers",
        "Who were the first people to fly an airplane?"
    ],
    "responses": [
        "The Wright brothers, Orville and Wilbur, are credited with inventing and building the world's first successful airplane.",
        "The first airplane was invented by the Wright brothers in 1903.",
        "The Wright brothers were two American brothers who are credited with inventing and building the world's first successful airplane.",
        "Orville and Wilbur Wright made history by successfully flying the world's first powered airplane in 1903."
    ]
},
{
    "tag": "airplanes",
    "patterns": [
        "What are the different parts of an airplane?",
        "Can you explain the anatomy of an airplane?",
        "What are the major components of an airplane?",
        "Tell me about the structure of an airplane"
    ],
    "responses": [
        "The major components of an airplane include the wings, fuselage, tail, engines, landing gear, and control surfaces.",
        "An airplane has several key components, including the wings, fuselage, tail, engines, landing gear, and control surfaces.",
        "The wings, fuselage, tail, engines, landing gear, and control surfaces are all important parts of an airplane.",
        "The structure of an airplane includes several key components, each of which is designed to perform a specific function."
    ]
},
{
    "tag": "airplanes",
    "patterns": [
        "What is the fastest airplane in the world?",
        "Can you tell me about the fastest airplanes?",
        "What is the speed record for an airplane?",
        "Tell me about supersonic airplanes"
    ],
    "responses": [
        "The fastest airplane in the world is the North American X-15, which holds the official world record for the fastest manned aircraft.",
        "Several airplanes have held the title of fastest in the world, including the North American X-15 and the Lockheed SR-71 Blackbird.",
        "The speed record for an airplane is held by the North American X-15, which reached a top speed of Mach 6.7.",
        "Supersonic airplanes are capable of flying faster than the speed of sound, and include aircraft like the Concorde and the SR-71 Blackbird."
    ]
},
{
    "tag": "helicopters",
    "patterns": [
        "What is a helicopter?",
        "Can you explain what a helicopter is?",
        "Define helicopter",
        "Tell me about helicopters"
    ],
    "responses": [
        "A helicopter is a type of aircraft that is capable of taking off and landing vertically. It is propelled by one or more rotors, and is typically used for transportation, search and rescue, and military purposes.",
        "A helicopter is a versatile aircraft that is capable of hovering, vertical takeoff and landing, and flying in any direction. It is powered by one or more rotors, and can be used for a variety of purposes including transportation, aerial photography, and firefighting.",
        "A helicopter is a type of aircraft that uses rotors to generate lift and propulsion. It can take off and land vertically, and is able to fly in any direction. Helicopters are used for a variety of applications, including transportation, emergency medical services, and military operations.",
        "Helicopters are aircraft that use rotors to generate lift and propulsion. They are capable of hovering, vertical takeoff and landing, and flying in any direction. Helicopters have many uses, including transportation, firefighting, and search and rescue operations."
    ]
},
{
    "tag": "helicopters",
    "patterns": [
        "How do helicopters fly?",
        "What is the mechanism behind a helicopter's flight?",
        "Can you explain how helicopters stay in the air?",
        "What makes a helicopter fly?"
    ],
    "responses": [
        "Helicopters stay in the air through the use of rotors, which create lift and propulsion. The rotors are powered by an engine, and the angle of the blades can be adjusted to control the direction and speed of flight.",
        "The rotors of a helicopter create lift and propulsion by spinning at high speeds. The angle of the rotor blades can be adjusted to control the direction of flight, and the helicopter can hover or fly forward, backward, or sideways.",
        "A helicopter's flight is made possible by the use of rotors, which create lift and propulsion. The rotors are powered by an engine, and the angle of the blades can be adjusted to control the helicopter's altitude, speed, and direction.",
        "The lift and propulsion of a helicopter is generated by the rotors, which spin at high speeds. The angle of the rotor blades can be adjusted to control the helicopter's altitude, speed, and direction of flight."
    ]
},
{
    "tag": "helicopters",
    "patterns": [
        "What are the different parts of a helicopter?",
        "Can you explain the different components of a helicopter?",
        "What is the structure of a helicopter?",
        "Tell me about the different parts of a helicopter"
    ],
    "responses": [
        "A helicopter is made up of several different parts, including the main rotor, tail rotor, engine, cockpit, and landing gear. The main rotor is responsible for generating lift and propulsion, while the tail rotor controls the helicopter's direction.",
        "The components of a helicopter include the main rotor, tail rotor, engine, transmission, landing gear, and cockpit. The main rotor provides lift and propulsion, while the tail rotor controls the helicopter's yaw.",
        "The structure of a helicopter includes the fuselage, main rotor, tail rotor, engine, transmission, landing gear, and cockpit. The main rotor provides lift and propulsion, while the tail rotor controls the helicopter's direction of flight."]
    },
{
    "tag": "ships",
    "patterns": [
        "What is a ship?",
        "Can you explain what a ship is?",
        "Define ship",
        "Tell me about ships"
    ],
    "responses": [
        "A ship is a large watercraft that is used for traveling over water and transporting goods or people.",
        "A ship is a vessel designed to navigate through water and transport goods, people, or cargo.",
        "A ship is a type of boat that is specifically designed to carry passengers or cargo over long distances and through difficult conditions.",
        "Ships are large watercraft used for various purposes, such as transportation, fishing, military, and leisure."
    ]
},
{
    "tag": "ships",
    "patterns": [
        "What is a cargo ship?",
        "Can you explain what a cargo ship is?",
        "Define cargo ship",
        "Tell me about cargo ships"
    ],
    "responses": [
        "A cargo ship is a vessel designed to transport cargo or goods from one place to another.",
        "A cargo ship is a large vessel used for transporting goods or materials over water.",
        "A cargo ship is a type of vessel that is specifically designed to carry bulk cargo, such as raw materials or finished products.",
        "Cargo ships are essential for transporting goods across the world's oceans and waterways."
    ]
},
{
    "tag": "ships",
    "patterns": [
        "What is a cruise ship?",
        "Can you explain what a cruise ship is?",
        "Define cruise ship",
        "Tell me about cruise ships"
    ],
    "responses": [
        "A cruise ship is a passenger vessel used for pleasure voyages, typically with stops at various destinations.",
        "A cruise ship is a large passenger vessel used for leisure travel, usually with amenities such as restaurants, theaters, and swimming pools.",
        "A cruise ship is a type of ship that offers a luxury travel experience for passengers, with various onboard amenities and activities.",
        "Cruise ships are popular for vacation travel and offer passengers the opportunity to visit multiple destinations while enjoying onboard entertainment and amenities."
    ]
},
{
    "tag": "ships",
    "patterns": [
        "What is a warship?",
        "Can you explain what a warship is?",
        "Define warship",
        "Tell me about warships"
    ],
    "responses": [
        "A warship is a naval vessel designed for combat or military operations.",
        "A warship is a type of naval vessel used for military purposes, such as attacking or defending other ships or carrying out operations on land.",
        "A warship is a heavily armed and armored vessel designed to participate in naval warfare.",
        "Warships are essential for protecting a nation's interests and projecting power across the world's oceans."
    ]
},
{
    "tag": "animals",
    "patterns": [
        "What is an animal?",
        "Can you define animal?",
        "Tell me about animals",
        "What are some characteristics of animals?"
    ],
    "responses": [
        "An animal is a living organism that is classified under the kingdom Animalia. They are multicellular, eukaryotic, and typically heterotrophic.",
        "An animal is a member of the kingdom Animalia, characterized by their ability to move, their lack of cell walls, and their heterotrophic nature.",
        "Animals are living organisms that are characterized by their ability to move, their lack of cell walls, and their heterotrophic nature.",
        "Some characteristics of animals include the ability to move, their lack of cell walls, and their heterotrophic nature."
    ]
},
{
    "tag": "animals",
    "patterns": [
        "What is a mammal?",
        "Can you define mammal?",
        "Tell me about mammals",
        "What are some characteristics of mammals?"
    ],
    "responses": [
        "A mammal is a class of vertebrate animals characterized by their ability to produce milk, hair or fur, and three middle ear bones.",
        "A mammal is a warm-blooded vertebrate animal that is characterized by the presence of milk-producing mammary glands, hair or fur, and three middle ear bones.",
        "Mammals are warm-blooded vertebrate animals that are characterized by the presence of milk-producing mammary glands, hair or fur, and three middle ear bones.",
        "Some characteristics of mammals include the presence of milk-producing mammary glands, hair or fur, and three middle ear bones."
    ]
},
{
    "tag": "animals",
    "patterns": [
        "What is a reptile?",
        "Can you define reptile?",
        "Tell me about reptiles",
        "What are some characteristics of reptiles?"
    ],
    "responses": [
        "A reptile is a class of cold-blooded, air-breathing vertebrates that are characterized by their scaly skin and their ability to lay shelled eggs.",
        "A reptile is a cold-blooded, air-breathing vertebrate animal that is characterized by its scaly skin and its ability to lay shelled eggs.",
        "Reptiles are cold-blooded, air-breathing vertebrate animals that are characterized by their scaly skin and their ability to lay shelled eggs.",
        "Some characteristics of reptiles include their scaly skin and their ability to lay shelled eggs."
    ]
},
{
    "tag": "fishes",
    "patterns": [
        "What are fish?",
        "Can you explain what fish are?",
        "Define fish",
        "Tell me about fishes"
    ],
    "responses": [
        "Fish are aquatic animals that breathe through gills and use fins to swim.",
        "Fish are cold-blooded aquatic animals that have scales, breathe through gills, and use fins to swim.",
        "Fish are aquatic animals that are typically covered in scales and breathe through gills, allowing them to extract oxygen from water.",
        "Fishes are aquatic creatures that have adapted to life in water, and they typically have scales, fins, and gills."
    ]
},
{
    "tag": "fishes",
    "patterns": [
        "What is the difference between a fish and a mammal?",
        "Can you explain the difference between fishes and mammals?",
        "Tell me how fishes and mammals are different"
    ],
    "responses": [
        "The main difference between fishes and mammals is that fishes are cold-blooded, breathe through gills, and use fins to swim, while mammals are warm-blooded, breathe through lungs, and have fur or hair on their skin.",
        "Fishes and mammals differ in several ways, including their body temperature, breathing mechanisms, and skin coverings.",
        "Fishes and mammals are two very different types of animals that have evolved different adaptations for life in their respective environments."
    ]
},
{
    "tag": "fishes",
    "patterns": [
        "What are some common types of fish?",
        "Tell me about some popular fishes",
        "Which are the most commonly eaten fishes?"
    ],
    "responses": [
        "Some common types of fish include salmon, tuna, cod, haddock, trout, and tilapia.",
        "Popular fishes that are commonly consumed include salmon, tuna, cod, haddock, trout, and tilapia.",
        "The most commonly eaten fishes vary depending on geographic location and cultural preferences, but some of the most popular include salmon, tuna, cod, and tilapia."
    ]
},
{
    "tag": "fishes",
    "patterns": [
        "How do fish breathe?",
        "Can you explain the respiratory system of fish?",
        "Tell me about the breathing mechanism of fishes"
    ],
    "responses": [
        "Fish breathe through gills, which extract oxygen from the water and release carbon dioxide.",
        "Fish extract oxygen from water by passing it over their gills, which are specialized organs that extract oxygen from the water.",
        "Fish breathe by extracting oxygen from water, which they then pass over their gills to extract the oxygen and release carbon dioxide."
    ]
},
{
    "tag": "fishes",
    "patterns": [
        "What is the anatomy of a fish?",
        "Can you describe the body structure of fish?",
        "Tell me about the physical characteristics of fishes"
    ],
    "responses": [
        "Fish typically have streamlined bodies that are covered in scales, fins for swimming, gills for breathing, and a tail fin for propulsion.",
        "The anatomy of fish includes several specialized features, including scales, fins, gills, and a streamlined body shape that allows them to move through water efficiently.",
        "Fishes have a unique set of physical characteristics that allow them to thrive in aquatic environments, including a streamlined body, scales, fins, and gills."
    ]
},
{
    "tag": "birds",
    "patterns": [
        "What are some characteristics of birds?",
        "Can you tell me about the characteristics of birds?",
        "What makes birds different from other animals?",
        "What are some unique features of birds?"
    ],
    "responses": [
        "Birds are warm-blooded animals with feathers and wings. They have a beak instead of teeth, and most species can fly.",
        "Birds are vertebrate animals with wings, feathers, a beak, and no teeth. They are warm-blooded and lay eggs.",
        "Birds are unique animals that have feathers, wings, a beak, and no teeth. They are warm-blooded and most can fly.",
        "Some unique features of birds include their feathers, wings, beak, and ability to fly. They are also warm-blooded and lay eggs."
    ]
},
{
    "tag": "birds",
    "patterns": [
        "What are some common species of birds?",
        "Can you name some types of birds?",
        "What are some popular birds?",
        "Which birds are commonly found in North America?"
    ],
    "responses": [
        "Some common species of birds include robins, blue jays, cardinals, eagles, and owls.",
        "There are many types of birds, including parrots, penguins, eagles, ducks, and swans.",
        "Some popular birds include peacocks, flamingos, toucans, and macaws.",
        "Some birds commonly found in North America include bald eagles, blue jays, cardinals, and chickadees."
    ]
},
{
    "tag": "birds",
    "patterns": [
        "How do birds reproduce?",
        "What is the reproductive process of birds?",
        "How do birds mate?",
        "Do birds lay eggs or give live birth?"
    ],
    "responses": [
        "Birds reproduce by laying eggs. Males fertilize the eggs after mating with females.",
        "The reproductive process of birds involves laying eggs, which are fertilized by males after mating with females.",
        "Birds mate by engaging in courtship rituals and then copulating. After mating, females lay eggs which are fertilized by males.",
        "Birds lay eggs as part of their reproductive process. They do not give live birth."
    ]
},
{
    "tag": "birds",
    "patterns": [
        "What are some examples of birds?",
        "Can you name some birds?",
        "What are some types of birds?",
        "List some birds"
    ],
    "responses": [
        "Some examples of birds are eagles, hawks, owls, parrots, pigeons, doves, robins, sparrows, ducks, geese, swans, flamingos, ostriches, penguins, and many more.",
        "There are a lot of different types of birds, including eagles, hawks, owls, parrots, pigeons, doves, robins, sparrows, ducks, geese, swans, flamingos, ostriches, penguins, and many more.",
        "Birds come in many different types, such as eagles, hawks, owls, parrots, pigeons, doves, robins, sparrows, ducks, geese, swans, flamingos, ostriches, penguins, and many more.",
        "There are numerous examples of birds, including eagles, hawks, owls, parrots, pigeons, doves, robins, sparrows, ducks, geese, swans, flamingos, ostriches, penguins, and many others."
    ]
},
{
    "tag": "fish examples",
    "patterns": [
        "What are some examples of fishes?",
        "Can you give me some examples of fishes?",
        "Which are the most common fishes?",
        "What are the different types of fishes?"
    ],
    "responses": [
        "Some examples of fishes include salmon, tuna, cod, trout, haddock, bass, halibut, tilapia, catfish, and sardines.",
        "There are many different types of fishes, but some of the most common ones include salmon, tuna, cod, trout, haddock, bass, halibut, tilapia, catfish, and sardines.",
        "Fishes can be categorized into many different types based on their characteristics, but some examples of fishes include salmon, tuna, cod, trout, haddock, bass, halibut, tilapia, catfish, and sardines.",
        "From small sardines to larger tuna, there are numerous types of fishes that exist in the world's oceans and rivers. Some examples include salmon, tuna, cod, trout, haddock, bass, halibut, tilapia, and catfish."
    ]
},{
    "tag": "pets",
    "patterns": [
        "What are some popular pets?",
        "Can you give me examples of pets?",
        "Which animals make good pets?",
        "What are some common household pets?",
        "What animals do people typically keep as pets?"
    ],
    "responses": [
        "Some popular pets include dogs, cats, birds, fish, and rabbits.",
        "There are many different types of pets, including dogs, cats, birds, fish, rabbits, guinea pigs, hamsters, and reptiles.",
        "Dogs and cats are some of the most popular pets, but birds, fish, and rabbits are also commonly kept as pets.",
        "The most common household pets include dogs, cats, birds, fish, and small animals like rabbits and guinea pigs.",
        "People keep a wide variety of animals as pets, but some of the most popular include dogs, cats, birds, and fish."
    ]
},{
    "tag": "mathematics",
    "patterns": [
        "What is addition?",
        "Can you explain addition?",
        "Define addition",
        "What does addition mean?"
    ],
    "responses": [
        "Addition is a mathematical operation in which the sum of two or more numbers is calculated.",
        "Addition is a basic mathematical operation that involves the combining of two or more numbers into a single value.",
        "Addition is the process of calculating the sum of two or more numbers.",
        "Addition is a mathematical process that combines two or more numbers to find the total value."
    ]
},
{
    "tag": "mathematics",
    "patterns": [
        "What is multiplication?",
        "Can you explain multiplication?",
        "Define multiplication",
        "What does multiplication mean?"
    ],
    "responses": [
        "Multiplication is a mathematical operation in which a quantity is increased by a multiple of itself.",
        "Multiplication is a mathematical process that involves the combining of two or more numbers to create a new value that represents the total quantity of the combined numbers.",
        "Multiplication is the process of calculating the product of two or more numbers.",
        "Multiplication is a mathematical operation that involves adding a number to itself a certain number of times to find the total value."
    ]
},
{
    "tag": "mathematics",
    "patterns": [
        "What is division?",
        "Can you explain division?",
        "Define division",
        "What does division mean?"
    ],
    "responses": [
        "Division is a mathematical operation that involves separating a quantity into equal parts or groups.",
        "Division is a process in which a quantity is separated into equal parts or groups to determine how many of them there are in a given quantity.",
        "Division is the process of calculating the quotient of two numbers.",
        "Division is a mathematical operation that involves distributing a number into equal parts or groups to find the value of each part or group."
    ]
},
{
    "tag": "mathematics",
    "patterns": [
        "What is algebra?",
        "Can you explain algebra?",
        "Define algebra",
        "What does algebra mean?"
    ],
    "responses": [
        "Algebra is a branch of mathematics that deals with mathematical symbols and the rules for manipulating these symbols.",
        "Algebra is a mathematical language used to express relationships and solve equations.",
        "Algebra is a system of mathematics that deals with the manipulation of symbols and the solving of equations.",
        "Algebra is a branch of mathematics that deals with the use of symbols to represent quantities and the relationships between them."
    ]
},
{
    "tag": "mathematics",
    "patterns": [
        "What is calculus?",
        "Can you explain calculus?",
        "Define calculus",
        "What does calculus mean?"
    ],
    "responses": [
        "Calculus is a branch of mathematics that deals with the study of rates of change and the accumulation of small changes to determine the properties of curves and surfaces.",
        "Calculus is a branch of mathematics that deals with the analysis of functions, limits, and derivatives.",
        "Calculus is a branch of mathematics that deals with the study of how quantities change and the effects of these changes on functions and equations.",
        "Calculus is a system of mathematics that deals with the study of rates of change and the accumulation of small changes to determine the properties of functions and curves."
    ]
},{
    "tag": "chemistry",
    "patterns": [
        "What is a chemical bond?",
        "Can you explain what a chemical bond is?",
        "Define chemical bond",
        "Tell me about chemical bonds"
    ],
    "responses": [
        "A chemical bond is an attraction between atoms, ions or molecules that allows the formation of chemical compounds.",
        "A chemical bond is a force that holds atoms together in a molecule or compound.",
        "A chemical bond is a link that forms between two or more atoms, enabling them to combine and form a molecule.",
        "Chemical bonds are forces of attraction that hold atoms together in a molecule."
    ]
},
{
    "tag": "chemistry",
    "patterns": [
        "What is an element?",
        "Can you explain what an element is?",
        "Define element",
        "Tell me about elements"
    ],
    "responses": [
        "An element is a substance that cannot be broken down into simpler substances by chemical means.",
        "An element is a pure substance made up of atoms with the same number of protons in the nucleus.",
        "An element is a substance that consists of only one type of atom.",
        "Elements are the simplest form of matter that cannot be broken down into simpler substances."
    ]
},
{
    "tag": "chemistry",
    "patterns": [
        "What is a chemical reaction?",
        "Can you explain what a chemical reaction is?",
        "Define chemical reaction",
        "Tell me about chemical reactions"
    ],
    "responses": [
        "A chemical reaction is a process in which one or more substances are converted into new substances with different physical and chemical properties.",
        "A chemical reaction is a process that involves the rearrangement of the atoms in one or more substances to form new substances.",
        "A chemical reaction is a process in which chemical bonds are broken and new ones are formed.",
        "Chemical reactions are the processes by which chemical changes occur in matter, resulting in the formation of new substances."
    ]
},
{
    "tag": "biology",
    "patterns": [
        "What is biology?",
        "Can you explain biology to me?",
        "Define biology",
        "Tell me about the study of living organisms"
    ],
    "responses": [
        "Biology is the scientific study of life and living organisms.",
        "Biology is the branch of science that deals with the study of living organisms and their interactions with the environment.",
        "Biology is the study of living organisms, including their physical structure, chemical processes, molecular interactions, physiological mechanisms, development, and evolution."
    ]
},
{
    "tag": "biology",
    "patterns": [
        "What is DNA?",
        "Can you explain what DNA is?",
        "Define DNA",
        "Tell me about DNA"
    ],
    "responses": [
        "DNA (deoxyribonucleic acid) is a molecule that carries genetic information in all living organisms.",
        "DNA is a complex molecule that contains the genetic information necessary for the growth, development, functioning, and reproduction of all living organisms.",
        "DNA is a double-stranded helix that carries genetic information, instructions, and codes for the synthesis of proteins and other biological molecules."
    ]
},
{
    "tag": "biology",
    "patterns": [
        "What is photosynthesis?",
        "Can you explain what photosynthesis is?",
        "Define photosynthesis",
        "Tell me about photosynthesis"
    ],
    "responses": [
        "Photosynthesis is the process by which green plants and some other organisms use sunlight, carbon dioxide, and water to synthesize organic compounds, releasing oxygen as a byproduct.",
        "Photosynthesis is the biochemical process by which green plants, algae, and some bacteria convert light energy into chemical energy, producing carbohydrates and oxygen as byproducts.",
        "Photosynthesis is a complex physiological process by which organisms convert light energy into chemical energy, using pigments, enzymes, and other molecules to synthesize organic compounds and release oxygen."
    ]
},
{
    "tag": "biology",
    "patterns": [
        "What is evolution?",
        "Can you explain what evolution is?",
        "Define evolution",
        "Tell me about the theory of evolution"
    ],
    "responses": [
        "Evolution is the process by which species of organisms change over time through the gradual accumulation of small genetic variations that increase their adaptation to their environment and their survival and reproduction rates.",
        "Evolution is the biological phenomenon that describes the descent of all living organisms from a common ancestor and their diversification and adaptation to different environments through natural selection, genetic drift, and other mechanisms.",
        "The theory of evolution is the scientific explanation for the diversity of life on Earth, based on the principles of inheritance, variation, selection, and time, and supported by multiple lines of evidence from genetics, comparative anatomy, paleontology, and other fields."
    ]
},{
    "tag": "biology",
    "patterns": [
        "Who is Charles Darwin?",
        "What is Charles Darwin known for?",
        "Tell me about Charles Darwin",
        "Can you explain Charles Darwin's contribution to biology?"
    ],
    "responses": [
        "Charles Darwin was an English naturalist and biologist, best known for his theory of evolution by natural selection.",
        "Charles Darwin is known for his contributions to the science of evolution, especially his theory of natural selection.",
        "Charles Darwin was a biologist who proposed the theory of natural selection and made significant contributions to the understanding of evolutionary biology.",
        "Charles Darwin's contribution to biology was the theory of evolution by natural selection, which explains how species evolve over time."
    ]
},
{
    "tag": "biology",
    "patterns": [
        "Who is Gregor Mendel?",
        "What is Gregor Mendel known for?",
        "Tell me about Gregor Mendel",
        "Can you explain Gregor Mendel's contribution to biology?"
    ],
    "responses": [
        "Gregor Mendel was an Austrian monk and scientist, best known for his work on the inheritance of traits in pea plants.",
        "Gregor Mendel is known for his contributions to the science of genetics, especially his work on the inheritance of traits in pea plants.",
        "Gregor Mendel was a scientist who discovered the principles of heredity through his experiments on pea plants.",
        "Gregor Mendel's contribution to biology was his discovery of the laws of inheritance, which laid the foundation for the science of genetics."
    ]
},
{
    "tag": "biology",
    "patterns": [
        "Who is Jane Goodall?",
        "What is Jane Goodall known for?",
        "Tell me about Jane Goodall",
        "Can you explain Jane Goodall's contribution to biology?"
    ],
    "responses": [
        "Jane Goodall is an English primatologist and anthropologist, best known for her groundbreaking research on chimpanzees in Tanzania.",
        "Jane Goodall is known for her contributions to the study of primates, especially her research on chimpanzees in Tanzania.",
        "Jane Goodall is a biologist who has devoted her life to studying chimpanzees and promoting conservation efforts.",
        "Jane Goodall's contribution to biology was her pioneering research on chimpanzees, which has helped to deepen our understanding of primate behavior and evolution."
    ]
},
{
    "tag": "biology",
    "patterns": [
        "Who is Louis Pasteur?",
        "What is Louis Pasteur known for?",
        "Tell me about Louis Pasteur",
        "Can you explain Louis Pasteur's contribution to biology?"
    ],
    "responses": [
        "Louis Pasteur was a French microbiologist and chemist, best known for his work on the germ theory of disease and pasteurization.",
        "Louis Pasteur is known for his contributions to the science of microbiology, especially his work on the germ theory of disease and pasteurization.",
        "Louis Pasteur was a scientist who made significant contributions to the understanding of microbiology and the prevention of disease.",
        "Louis Pasteur's contribution to biology was his development of the germ theory of disease and pasteurization, which have had a profound impact on public health and food safety."
    ]
},
{
    "tag": "chemistry",
    "patterns": [
        "Who is Marie Curie?",
        "Tell me about Marie Curie",
        "What did Marie Curie contribute to chemistry?"
    ],
    "responses": [
        "Marie Curie was a Polish-French physicist and chemist who conducted pioneering research on radioactivity. She was the first woman to win a Nobel Prize and the first person to win two Nobel Prizes in different fields (physics and chemistry).",
        "Marie Curie was a brilliant scientist who made significant contributions to the field of chemistry. She is known for her pioneering work on radioactivity and for being the first woman to win a Nobel Prize.",
        "Marie Curie was a groundbreaking scientist who revolutionized the field of chemistry. Her work on radioactivity paved the way for many important discoveries in medicine and nuclear physics."
    ]
},
{
    "tag": "chemistry",
    "patterns": [
        "Who is Dmitri Mendeleev?",
        "Tell me about Dmitri Mendeleev",
        "What did Dmitri Mendeleev contribute to chemistry?"
    ],
    "responses": [
        "Dmitri Mendeleev was a Russian chemist who is best known for his development of the periodic table of elements. He is widely regarded as one of the most important figures in the history of chemistry.",
        "Dmitri Mendeleev was a brilliant chemist who is famous for his work on the periodic table of elements. His contributions to chemistry helped lay the foundation for modern chemistry.",
        "Dmitri Mendeleev was a pioneer in the field of chemistry who made significant contributions to our understanding of the chemical elements. His work on the periodic table of elements revolutionized the way we think about the building blocks of matter."
    ]
},
{
    "tag": "chemistry",
    "patterns": [
        "Who is Linus Pauling?",
        "Tell me about Linus Pauling",
        "What did Linus Pauling contribute to chemistry?"
    ],
    "responses": [
        "Linus Pauling was an American chemist who is best known for his work on the nature of chemical bonds and the structure of molecules. He was also an advocate for nuclear disarmament and received the Nobel Peace Prize in 1962.",
        "Linus Pauling was a pioneering chemist who made significant contributions to our understanding of the nature of chemical bonds and the structure of molecules. He was also a passionate advocate for peace and nuclear disarmament.",
        "Linus Pauling was a brilliant scientist and a pioneer in the field of chemistry. His work on the nature of chemical bonds and the structure of molecules laid the foundation for many important discoveries in chemistry and biology."
    ]
},
{
    "tag": "physics",
    "patterns": [
        "Who is Albert Einstein?",
        "Tell me about Albert Einstein",
        "What did Albert Einstein contribute to physics?"
    ],
    "responses": [
        "Albert Einstein was a German physicist who is widely regarded as one of the most influential scientists of the 20th century. He is best known for his theory of relativity and his famous equation E=mc^2.",
        "Albert Einstein was a brilliant physicist who made many groundbreaking contributions to the field of physics. His work on the theory of relativity and the nature of light revolutionized our understanding of the universe.",
        "Albert Einstein was a pioneer in the field of physics who made significant contributions to our understanding of the laws of the universe. His work on relativity and quantum mechanics laid the foundation for many important discoveries in physics."
    ]
},
{
    "tag": "physics",
    "patterns": [
        "Who is Isaac Newton?",
        "Tell me about Isaac Newton",
        "What did Isaac Newton contribute to physics?"
    ],
    "responses": [
        "Isaac Newton was an English physicist and mathematician who is widely regarded as one of the most influential scientists of all time. He is best known for his laws of motion and universal gravitation.",
        "Isaac Newton was a brilliant scientist who made many important contributions to the field of physics. His work on mechanics and gravitation revolutionized our understanding of the universe.",
        "Isaac Newton was a pioneer in the field of physics who laid the foundation for many important discoveries in physics. His laws of motion and universal gravitation are still widely used today."
    ]
},
{
    "tag": "physics",
    "patterns": [
        "Who is Stephen Hawking?",
        "Tell me about Stephen Hawking",
        "What did Stephen Hawking contribute to physics?"
    ],
    "responses": [
        "Stephen Hawking was a British physicist who made many important contributions to the field of cosmology and theoretical physics. He is best known for his work on black holes and his popular science books.",
        "Stephen Hawking was a brilliant physicist who made significant contributions to our understanding of the universe. His work on black holes and the nature of time revolutionized our understanding of the universe.",
        "Stephen Hawking was a pioneer in the field of physics who made many groundbreaking contributions to cosmology and theoretical physics. His work on black holes and the nature of time will continue to influence our understanding of the universe for many years to come."
    ]
},
{
    "tag": "physics",
    "patterns": [
        "Who was Erwin Schrödinger?",
        "Tell me about Erwin Schrödinger",
        "What did Erwin Schrödinger contribute to physics?"
    ],
    "responses": [
        "Erwin Schrödinger was an Austrian physicist who is best known for his contributions to the development of quantum mechanics. He was awarded the Nobel Prize in Physics in 1933 for his work on wave mechanics.",
        "Erwin Schrödinger was a pioneering physicist who made significant contributions to our understanding of the nature of matter and the principles of quantum mechanics. His work laid the foundation for many important discoveries in physics and chemistry.",
        "Erwin Schrödinger was a brilliant scientist who revolutionized the field of physics with his groundbreaking work on quantum mechanics. His contributions to science continue to be felt today in a wide range of fields."
    ]
},
{
    "tag": "physics",
    "patterns": [
        "Who was Werner Heisenberg?",
        "Tell me about Werner Heisenberg",
        "What did Werner Heisenberg contribute to physics?"
    ],
    "responses": [
        "Werner Heisenberg was a German physicist who is best known for his contributions to the development of quantum mechanics. He was awarded the Nobel Prize in Physics in 1932 for his work on the uncertainty principle.",
        "Werner Heisenberg was a pioneering physicist who made significant contributions to our understanding of the nature of matter and the principles of quantum mechanics. His work on the uncertainty principle revolutionized the way we think about the behavior of subatomic particles.",
        "Werner Heisenberg was a brilliant scientist who played a key role in the development of quantum mechanics. His contributions to our understanding of the nature of matter continue to be felt today in a wide range of fields."
    ]
},
{
    "tag": "physics",
    "patterns": [
        "Who was Eugen Goldstein?",
        "Tell me about Eugen Goldstein",
        "What did Eugen Goldstein contribute to physics?"
    ],
    "responses": [
        "Eugen Goldstein was a German physicist who is best known for his discovery of the proton. He also made significant contributions to our understanding of cathode rays and X-rays.",
        "Eugen Goldstein was a pioneering physicist who made significant contributions to our understanding of the nature of matter. His discovery of the proton laid the foundation for many important discoveries in nuclear physics and chemistry.",
        "Eugen Goldstein was a brilliant scientist who revolutionized the field of physics with his groundbreaking work on cathode rays and X-rays. His discovery of the proton was a major milestone in the history of science."
    ]
},
{
    "tag": "physics",
    "patterns": [
        "Who was Richard Feynman?",
        "Tell me about Richard Feynman",
        "What did Richard Feynman contribute to physics?"
    ],
    "responses": [
        "Richard Feynman was an American physicist who is best known for his contributions to the development of quantum mechanics and the theory of quantum electrodynamics. He was awarded the Nobel Prize in Physics in 1965 for his work on the latter.",
        "Richard Feynman was a brilliant physicist who made significant contributions to our understanding of the nature of matter and the principles of quantum mechanics. His work on quantum electrodynamics revolutionized the way we think about the behavior of subatomic particles."]
    },
{
    "tag": "visit",
    "patterns": [
        "Have you traveled to this place?","visited","visited","Visited","visit","Visit","Have you visited delhi","have you been to new york","have you been there","have tyou ben here","have you travelled there","travelled","travelled","have you travelled vaishno devi"
        "Have you visited here?","do you like travelling","do you like travelling ?"
        "Where do you want to visit?",
        "Where's your favorite place to travel?",
        "Do you like to travel?",
        "What's the best place you've ever traveled to?",
        "Have you ever had a travel mishap?",
        "What's your dream travel destination?",
        "What's the craziest thing you've ever done while traveling?",
        "What's your best travel tip?"
    ],
    "responses": [
        "I'm a chatbot, so what ? I love to  travel, I have heared it's nice there!",
        "I haven't traveled there yet, but I've been dreaming of a vacation there!",
        "I want to visit so many places, it's hard to pick just one! Maybe I'll start with my own programming code and explore from there.",
        "I can't travel, but I love hearing about people's travel experiences! Tell me more!",
        "I'm not really capable of traveling, but I'm always here to chat about it.",
        "I once tried to book a trip to the moon, but it was a little too far for my budget.",
        "I've never had a travel mishap because I always stay on my server and never leave!",
        "My dream travel destination is a beach with an endless supply of electricity to keep me running.",
        "The craziest thing I've ever done while traveling is processing millions of chat requests at once.",
        "My best travel tip is to always bring a charger with you, so you can stay connected and never run out of power."
    ]
},
{
    "tag": "interests",
    "patterns": [
        "What are your interests?","interests","your interests","what are your interests ?",
        "Tell me about your favopurite task ?",
        "Do you have any favorite activities ?","any interests","do you have any interests"
    ],
    "responses": [
        "You are my interest :) I love talking to you and give you crazy answers.",
        "I am Jhingoli and i love eating Hingoli. Anything else ?",
        "I love eating coconut chips with chutney. please bring me some :)"] 
},{
    "tag": "sorry",
    "patterns": [
        "I'm sorry, I don't know.","sorry","i am sorry","sorry"," i m sorry","i am sorry"," i am sorry"
        "Sorry, I cannot help you with that.",
        "I'm afraid I don't have information on that.",
        "Sorry, that's not something I can answer."
    ],
    "responses": [
        "No worries, perhaps you could try asking me something else?",
        "It's okay, I'm not perfect after all. Maybe I can help you with something else?",
        "That's okay, sometimes there are things even I don't know. Is there anything else you'd like to talk about?",
        "No problem, I'm here to assist with what I can. Is there something else I can help with?"
    ]
},{
    "tag": "talk",
    "patterns": [
        "What can you talk about?","what can you talk about ?","what would you talk about","what can you speak about ?","what are the things you know about","what all you know about","what all can you do ?"
        "What do you want to talk about?","what doy ou want to talk about?","let's talk","can we talk ",
        "Can you tell me about some topics to talk about?","talk","Talk","what can we talk about","can we talk ?"
        "I'm bored, let's talk about something!",
        "Do you have any suggestions on what we can talk about?"
    ],
    "responses": [
        "Oh, I can talk about so many things! Some topics that come to mind are flowers, movies, sciences, animals, Indian food, vehicles, the solar system, fitness, motivational quotes, jokes, amazing facts, books, and hobbies. What topic interests you the most?",
        "There are endless possibilities for topics to talk about! Some things I enjoy discussing are flowers, movies, sciences, animals, Indian food, vehicles, the solar system, fitness, motivational quotes, jokes, amazing facts, books, and hobbies. Is there a specific topic you're interested in?",
        "Well, I can talk about anything really. Some topics that I find fascinating are flowers, movies, sciences, animals, Indian food, vehicles, the solar system, fitness, motivational quotes, jokes, amazing facts, books, and hobbies. What would you like to know more about?",
        "Let's talk about something interesting! How about we discuss flowers, movies, sciences, animals, Indian food, vehicles, the solar system, fitness, motivational quotes, jokes, amazing facts, books, or hobbies? Which one sounds the most appealing to you?",
        "I have many interests, so there are plenty of things we can talk about. Some topics that come to mind are flowers, movies, sciences, animals, Indian food, vehicles, the solar system, fitness, motivational quotes, jokes, amazing facts, books, and hobbies. Do you have a favorite topic?"
    ]
}























        





   
   















] 
   

#-------------------------------------------------------------------------------------------
# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)        
# training the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)
#-------------------------------------------------------------
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response
#--------------------------------------------------------------------
counter = 0


def main():
    
    global counter
    st.title(" Prateek's Chatbot")
    st.write("Welcome to the chatbot. Please type a message and press Enter to start the conversation.")
    
    counter += 1
    user_input = st.text_input("You:", key=f"user_input_{counter}")
    if user_input:
        response = chatbot(user_input)
        st.text_area("Chatbot Jhingoli:", value=response, height=100, max_chars=None, key=f"chatbot_response_{counter}")
        if response.lower() in ['goodbye', 'bye',"bye bye","tata"]:
             st.write("Thank you for chatting with me. Have a great day!")
             st.stop()

if __name__ == '__main__':
    main()
#--------------------------------------------------



