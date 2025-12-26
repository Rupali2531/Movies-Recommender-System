import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from datetime import datetime


rating = pd.read_csv("rating.csv")
movies = pd.read_csv("movies.csv")



movies = pd.read_csv("movies.csv")

rating = pd.read_csv("rating.csv")


movies.shape

ratings.shape

movies.head()

ratings.head()

plt.figure(figsize =(20,7))
generlist = movies['genres'].apply(lambda generlist_movie:str(generlist_movies).split("|"))
genres_count ={}

for generlist_movie in generlist:
    for gener in generlist_movie:
        if(geners_count.get(gener,False)):
            geners_count[gener]=geners_count[gener]+1
        else:
            genres_count[gener]=1
geners_count.pop("(no genser listed)")
plt.bar(geners_count.keys(),geners_count.values(),color='m')


ratings_df=ratings.copy()
ratings_df['timestamp']=ratings_df['timestamp'].apply(datetime.fromtimestamp)
rating_df['year']=ratings_df['timestamp'].dt.year
ratings_df['month']=ratings_df['timestamp'].dt.month
ratings_df = rating_df.sort_values('timestamp')
print('First 5:')
display(ratings_df.head())

year_counts= ratings_df[['year','rating']].groupby(['year']).count()
year_counts= year_counts.rename(index=str,columns={'rating':'# of Ratings'})
year_counts=year_counts.reset_index()
year_counts=year_counts.set_index('year',drop=True)
display(year_counts[0:5])
year_counts['# of Ratings'].plot(style='0-')
plt.ylabel('# of Ratings')
plt.title('# of Ratings per year')
plt.ylim([0,25000])
plt.gca().grid(which='minor')
plt.show()

data=pd.pivot(index='movieId',columns='userId',data=ratings,values='rating')
data.head()

print("Shape of frames:\n "+" Rating DataFrame" + str(ratings.shape)+"\n Movies DataFrame"+ str(movies.shape))

merge_ratings_movies =pd.merge(movies,ratings,on='moviesId',how='inner')
merge_ratings_movies.head(2)

merge_ratings_movies =merge_rating_movies.drop('timestamp',axis=1)
merge_rating_movies.shape

ratings_grouped_by_users=merge_rating_movies.groupby('userId').agg([np.size,np.mean])
ratings_grouped_by_users.head(2)

ratings_grouped_by_users['rating']['size'].sort_values(asending=False).head(10).plot(kind='bar',figsize=(10,5))

ratings_grouped_by_movies=merge_rating_movies.groupby('movieId').agg([np.mean],np.size)
ratings_grouped_by_movies=ratings_grouped_by_movies.drop('userId',axis=1)
ratings_grouped_by_movies['rating']['mean'].sort_values(ascending=False).head(10).plot(kind='barh',figsize=(7,6));

low_rated_movies_filter=rating_grouped_by_movies['rating']['mean']<1.5
low_rated_movies=rating_grouped_by_movies
[low_rated_movies_filter]
low_rated_movies.head(20).plot(kind='barh',figsize=(7,5));

numberOf_user_voted_for_movie=pd.DataFrame(rating.groupby('movieId')['rating'].agg('count'))
numberOf_user_voted_for_movie.head()

data.head()

data.fillna(0,inplace=True)
data.head()

f,ax=plt.subplots(1,1,fidsize=(16,4))
plt.scatter(numberOf_user_voted_for_movie.index,
numberOf_user_voted_for_movie,color='forestgreen')
plt.axhline(y=10,color='r')
plt.xlabel('MovieId')
plt.ylabel('No.of users voted')
plt.show()

f,ax=plt.subplots(1,1,figsize=(16,4))
plt.scatter(numberOf_movies_voted_by_user.index,
numberOf_movies_voted_by_user,color='forestgreen')
plt.axhline(y=50,color='r')
plt.xlabel('userId')
plt.ylabel('no.of movies voted')
plt.show()

data_final = data.loc[numberOf_user_voted_for_movie
[numberOf_user_voted_for_movie>10].index,:]
data_final=data_final.loc[:,numberOf_movies_voted_by_user
[numberOf_movies_voted_by_user>50].index]
data_final.head()

data_final.shape

data.shape

trial.sample=np.array([[0,0,3,0,0],[4,0,0,0,2],[0,0,0,1]])
sparsity =1.0-(np.count_nonzero(trial_sample)/float(trial_sample.size))
print(sparsity)

csr_sample= csr_matrix(trial_sample)
print(csr_sample)

csr_data=csr_matrix(data_final.values)
data_final.reset_index(inplace=True)
data_final.head()

movies.head()

import re
def clean_title(title):
    title=re.sub["^a-zA-Z0-9]","",title]
    return title

movies["clean_title"]=movies["title"].apply(clean_title)

movies

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidVectorizer(ngram_range=(1,2))
tfidf=vectorizer.fit_transform(movies["clean_title"])

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def search(title):
    title = clean_title(title)
    
    query_vec = vectorizer.transform([title])
    
    similarity = cosine_similarity(query_vec,tfidf).flatten()
    
    indices = np.argpartition(similarity,-5)[-5:]

    result =movies.iloc[indices].iloc[::-1]

    return results

search("Toy Story")


import ipywidgets as widgets
from IPython.display import display

movie_input = widgets.Text(
    value='Toy Story',
    description='Movie Title:',
    disabled=False
)
movie_list = widgets.Output()

def on_type(data):
    with movie_list:
        movie_list.clear_output()
        title = data["new"]
        if len(title) > 5:
            display(search(title))

movie_input.observe(on_type, names='value')

display(movie_input, movie_list)

movie_id =89745
movie = movies["movieId"]==movie_id


ratings = pd.read_csv(rating_file)

similar_users = ratings[(ratings["movieId"]==movie_id) & (ratings ["rating"]>4)]["userId"].unique()

similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"]>4)]["movieId"]

similar_user_recs = similar_user_recs.value_counts()/len(similar_users)

similar_user_recs = similar_user_recs[similar_user_recs >.10]


all_user = ratings[(rating[movieId].isin(similar_user_recs.index)) & (ratings["rating"]>4)]

all_user_recs = all_users["movieId"].value_counts()/len(all_users["userId"].unique())

rec_percentages = pd.concat([similar_user_recs,all_user_recs],axis=1)
rec_percentages.columns=["similar","all"]

rec_precentages

rec_percentages["score"]=rec_percentages["similar"]/rec_percentages["all"]

rec_percentages = rec_percentages.sort_values("score", ascending=False)

rec_precentages.head(10).merge(movies,left_index=True,right_on = "movieId")

def find_similar_movies(movies_id):
    similar_users = ratings[(ratings["movieId"]== movie_id) & (ratings["rating"]>4)]["userId"].unique()
    similar_user_recs = rating[(rating["userId"].isin(similar_users)) & (ratings["rating"]>4)]["movieId"]
    similar_user_recs = similar_user_recs.value_counts()/len
    (similar_users)

    similar_user_recs = similar_user_recs= similar_user_recs[similar_user_recs >.10]
    all_users=ratings[(ratings["movieId"].isin(similar_users_recs.index)) & (ratings["ratings"]>4)]
    all_user_recs = all_users["moviesId"].value_counts()/len
    (all_users["userId"].unique())
    rec_percentages.columns=["similar","all"]
    
    rec_percentages["score"]=rec_percentages["similar"]/rec_precentages["all"]
    rec_precentages = rec_precentages.sort_values("score", ascending=False)
    return rec_precentages.head(10).merge(movies,left_index=True,right_on="movieId")[["score","title","genres"]]

import ipywidgets as widgets
from iPython.display import display

movie_name_input=widgets.Text(
    value ='Toy Story',
    description='Movie Title:',
    disabled=False
)
movie_list = widgets.Output()

def on_type(data):
    with recommendation_list:
           recommendation_list.clear_output()

           title=data["new"]
           if len(title)>5:
               result=search(title)
               movie_id = results.iloc[0]["movieId"]
               display(find_similar_movies(movie_id))
movie_input.observe(on_type,names='value')
   
display(movie_input,movie_list)
   
#KNN-Based Collaborative filtering

numberOf_user_voted_for_movie = pd.DataFrame(ratings.groupby('movieId')['rating'].agg('count'))
numberOf_user_voted_for_movie.reset_index(level=0,inplace=True)
numberOf_user_voted_for_movie.head()

numberOf_movies_voted_by_user=pd.DataFrame(ratings.groupby('userId')['rating'].agg('count'))
numberOf_movies_voted_by_user.reset_index(level=0,inplace=True)
numberOf_movies_voted_by_user.head()

data_final1= data.loc[numberOf_user_voted_for_movie
                      [numberOf_user_voted_for_movie['rating']>10]['movieId'],:]
data_final1=data_final1.iloc[:,numberOf_movies_voted_by_user
                      [numberOf_movie_voted_by_user['rating']>60]['userId']]
data_final1.shape

from scipy.sparse import csr_matrix
csr_data=csr_matrix(data_final1.values)
data_final1.reset_index(inplace=True)

from sklearn.neighbors import NearestNeighbors
knn=NearestNeighbors(metric='cosine',algorithm='brute',n_neighbors=20)
knn.fit(csr_data)
def get_movie_recommendation(movie_name):
    n=10
    movie_list = movies[movies['title'].str.contains(movie_name)]
    if len(movie_list):
        movie_idx = movie_list.iloc[0]['movieId']
        movie_idx=data_final[data_final['movieId']==movie_idx].index[0]
        distances,indices=knn.kneighbors(csr_data[movie+idx], n_neighbores=n+1)

        rec_movie_indices=sorted(list(zip(indices.squeeze(),
        distances.squeeze())),key=lambda x: x[1])[1::1]
        recommend=[]
        recommend=[]
        for val in rec_movie_indices:
            movie_idx = data_final.iloc[val[0]]['movieId']
            idx=movie[movie['movieId']==movie_idx].index
            recommend.append(movies.iloc[idx]['title'].values[0])
            recommend2.append(val[1])
        df1=pd.DataFrame(recommend)
        df2=pd.DataFrame(recommend2)
        df=pd.concat([df1,df2],axis='colums')
        df.columns=['Title','Distance']
        df.set_index('Distance',inplace=True)
        return df
    else:
        return "No movies found.plaase check your input"
    get_movie_recommendation("Toy Story")
    print("Enter the number of movies you would love to watch from the above list of recommendation")
    print(input())
    print("enter the no of movie from the above list of recommendation that you would say is irrelevant to your taste")
    ir=int(input())

    precision=p/10
    recall=p/(10-ir)
    F_score=(2*precision*recall)/(precision+recall)
    print("F_score:",F_score)


#SVD-BASED COLLABORATIVE FILTERING

    
data=pd.pivot(index='movieId',columns='userId',data=ratings,values='rating')

numberOf_user_voted_for_movie=ratings.groupby('movieId')
['rating'].agg('count')
numberOf_movies_voted_by_user=rating.groupby('userId')['rating'].agg('count')
data.fillna(0,inplace=True)
data_final=data.loc[numberOf_user_voted_for_movie
[numberOf_user_voted_for_movie>10].index,:]
data_final=data_final.loc[:,numberOf_movies_voted_by_user
[numberOf_movies_voted_by_user >50].index]

csr_data =csr_matrix(data_finql.values)
data_final.reset_index(inplace=True)


from sklearn.utils.extmath import randomized_svd

U,S,V =randomized_svd(csr_data,
                      n_components=15,
                      n_iter=5,
                      random_state=42)

csr_data



#Making the recommendation funcation


movie_data=pd.read_csv(movies_file)
data=pd.read_csv(ratings_file)


def top_cosine_similarity(data,movie_id,top_n=10):
     index = movie_id -1
     movie_row = data[index,:]
     magnitude =np.sqrt(np.einsum('ij,ij-> i',data,data))
     similarity=np.dot(movie_row,data.T)/(magnitude[index]*magnitude)
     sort_indexes = np.argsort(-similarity)
     return sort_indexes[:top_n]

def print_similar_movies(movie_data,movie_id,top_indexes):
     print('Recommendation for {0}:\n'.format(
     movie_data[movie_data.movieId == movie_id].title.values))
     for id in top_indexes+1:
         print((movie_data[movie_data.movieId== id].title.values)[0])

#k-principal components to represent movies, movie_id to find recommendations, top_n print n results  
         
k=50
movie_id=10
top_n= 10
sliced=V.T[:,:k]
indexes =top_cosine_similarity(sliced,movie_id,top_n)

print_similar_movies(movie_data,movie_id,indexes)


#Evaluate Model

from surprise import SVD

from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise import Dataset,Reader
from surprise.model_selection import cross_validate

reader=Reader(line_format='user item rating',rating_scale=(0,5))


data=dataset.load_from_df(ratings[['userId','movieId','rating']],reader)

svd = SVD()
cross_validate(svd,data,measures=['RMSE','MAE'])

         
