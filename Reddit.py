print("JAI BAJRANG BALI")
import praw #For Scraping the comments
from transformers import AutoTokenizer,AutoModelForSequenceClassification #For NLP of comment text
from scipy.special import softmax                                         #For Generating end result in form of Probability
import pandas as pd #For converting the end result into a CSV file
import matplotlib.pyplot as plt # Library for showing progress bar
from time import sleep #progress bar
from tqdm import tqdm#.....
from progress.bar import Bar#....
from alive_progress import alive_bar#....


#Fucntion for sentiment analysis
def NLP(Data):
 cmnt = Data
 #Preprocessing of tweet by removing user names and urls
 cmnt_words=[]
 # noises like emojis, improper or excessive spaces removal, punctuatuion etc removal
 for word in cmnt.split(' '):
   if word.startswith('@') and len(word)>1:
    word = '@user'
   elif word.startswith('http'):
     word="http"
   cmnt_words.append(word)
 cmnt_proc=" ".join(cmnt_words)
 #<-----loading model and tokenizer----->
 roberta = "cardiffnlp/twitter-roberta-base-sentiment"
 model = AutoModelForSequenceClassification.from_pretrained(roberta)
 tokenizer = AutoTokenizer.from_pretrained(roberta)
 lables = ['Negative','Neutral','Positive']
 #<-----Sentiment Analysis starts here----->
 encoded_cmnt = tokenizer(cmnt_proc,return_tensors='pt')
 output = model(**encoded_cmnt)
 scores = output[0][0].detach().numpy()
 scores=softmax(scores)
 ret = 0
 scr=scores[0]
 for i in range(len(scores)):
   l = lables[i]
   s = scores[i]
   if i>=1 and scores[i]>scores[i-1]:
      ret=i
      scr=scores[i]
   #print(l,s)
 if ret==0:
    return {"Negative":scr}
 elif ret==1:
    return {"Neutral":scr}
 else:
    return {"Positive":scr}






reddit = praw.Reddit(user_agent=True,client_id="7Ek3_8aZPCjzJXKDb-friA",client_secret="uS73HotWy3FT9-oOtiSVIfzwfhFeLQ",
        username="leather_Trainer2234",password="Deevanshu@2009")
# url="https://www.reddit.com/r/IndiaSpeaks/comments/16eafon/the_g20_summit_is_over_here_what_india_achieved/"
url=str(input("Enter url of reddit post to be Analyzed for sentiment analysis :"))
post = reddit.submission(url=url)
i=0
print(post.selftext)
cmnt=len(post.comments)

# <----------------creating dataframe-------->
columns = ["Sentiment","Score","Text","Author","Parent","Time Stamp"]
DB=[]
# <----------------For all Top,Second and Third level comments-------->

post.comments.replace_more(limit=None)
st=0
pv=0
nv=0
nu=0
cu=0
with alive_bar(cmnt) as bar:
 for comment in post.comments.list():
  cu = cu+1
  D=NLP(comment.body)
  if list(D.keys())[0]=="Positive" :
    pv=pv+1
  if list(D.keys())[0]=="Neutral" :
    nu=nu+1
  if  list(D.keys())[0]=="Negative":
    nv=nv+1
  DB.append([list(D.keys())[0],list(D.values())[0],comment.body,comment.author,comment.parent(),comment.created_utc])
  if cmnt==cu:
    break
  #i=i+1
  #st=st+1
  #print(st)
  bar()
  #if st==10 :
  #  break
#print(i)
DF = pd.DataFrame(DB,columns=columns)
print(DF)
DF.to_csv(r"C:\Users\Goura\Desktop\Scrappers\outputs\New1_Sentiment_Score.csv")

plt.style.use('bmh')

print(pv,nv,nu)
plt.xlabel('Sentiment',fontsize =10)
plt.ylabel('Score',fontsize =10)
bars=plt.bar(['Positive','Neutral','Negative'],[pv,nu,nv])
bars[0].set_color('Green')
bars[1].set_color('Yellow')
bars[2].set_color('Red')

plt.show()

# plt.pie(x2,labeldistance=x1,radius=1.2,autopct='%0.01f%%',shadow=True)

plt.show()



#   text = comment.body
#     author = comment.author
#     commurl = comment.permalink(fast=False)
#     parent = comment.parent()
#     comtime = comment.created_utc







# <-----------------For Top Level Comments------------------->

# for comment in post.comments:
#     print(comment.body +"/n")




# <-----------------For top level comments ignoring More comments option------------>


# for top_level_comment in post.comments:
#     if isinstance(top_level_comment, MoreComments):
#         continue
#     print(top_level_comment.body)




# print(reddit.user.me()) 