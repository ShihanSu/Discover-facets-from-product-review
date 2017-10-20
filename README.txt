Run "make" to build "beer_seg". See the "main" function to modify the parameters of the experiment being run.

Files:

reviews/ -- review corpora. Example from BeerAdvocate:
1586259 19418 # Number of reviews, number of unique words
5 feel 10 look 10 overall 10 smell 10 taste 10 # Number of aspects, followed by the names of each aspect, and the number of possible ratings for each (in this case, 5 stars in half-star increments = 10 ratings)
3 feel mouthfeel palate # Names of each of the aspects. Used to initialize the model.
2 look appearance
2 overall drinkability
2 smell aroma
1 taste
851977 47986.data 34952 30566 8 # Review id, internal product id (i.e., the id on beeradvocate), product id, user id, number of sentences
3 5 3 4 3 # Ratings for each of the K aspects
2 10294 7162 # Number of words in the first sentence, followed by the words
1 10294
4 15692 1755 9782 17090
2 7823 16266
8 13752 4878 12172 4027 10208 3190 18542 7162
3 17230 9782 15943
1 17103
2 19269 1755

models/ -- learned models (will be overwritten if you learn new ones).
models/modelSeg* -- segmentation weights
models/modelSen* -- sentiment weights

wordids/ -- mapping from each wordid to the word it represents

groundtruth/ -- labeled groundtruth data. Example from BeerAdvocate:
114100 12 look None taste taste feel feel overall None overall overall None overall # Review id, number of sentences, labels for each sentence
1511382 13 None overall overall look look None taste taste feel overall overall None overall
