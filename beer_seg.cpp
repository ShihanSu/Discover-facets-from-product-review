#include "stdio.h"
#include "stdlib.h"
#include "vector"
#include "string.h"
#include "string"
#include "map"
#include "omp.h"
#include "limits"
#include "math.h"
#include "sstream"
#include "munkres/src/munkres.h"
using namespace std;

// make fopen report an error if a file doesn't exist
FILE* _fopen(string fname)
{
  FILE* f = fopen(fname.c_str(), "r");
  if (f == NULL)
  {
    printf("Failed to open %s\n", fname.c_str());
    exit(1);
  }
  return f;
}

// Sign of a double
int sgn(double val)
{
 return (0 < val) - (val < 0);
}

double bestll = -numeric_limits<double>::max();

/**
 * A single review. Basically an array of sentences, each of which is an array of words (each of which is an integer word id)
 */
class review
{
public:
  review(int reviewID, string itemID, string userID, int NS, int* vote) : reviewID(reviewID), itemID(itemID), userID(userID), NS(NS), vote(vote)
  {
    NW = new int [NS];
    sentences = new int* [NS];
  }

  ~review()
  {
    for (int i = 0; i < NS; i ++)
      delete [] sentences[i];
    delete [] sentences;
    delete [] vote;
    delete [] NW;
  }

  int reviewID;
  string itemID;
  string userID;

  int NS; // Number of sentences
  int* NW; // Number of words in each sentence

  int* vote; // Vector of votes
  int** sentences; // word ids for each word in each sentence
};

/**
 * Code to read and store all reviews in a corpus
 */
class corpus
{
public:
  // path = location of the corpus
  // wordPath = file mapping word ids to words
  // gtPath = location of groundtruth data
  // rmax = maximum reviews to use (0 uses all reviews in the corpus)
  corpus(string path, string wordPath, string gtPath, int rmax = 0)
  {
    nkTotal = 0;
    FILE* f = _fopen(gtPath);
    int reviewID;
    while (fscanf(f, "%d", &reviewID) == 1)
    {
      int NS;
      fscanf(f, "%d", &NS);
      groundTruth[reviewID] = new string [NS];
      gtSize[reviewID] = NS;
      char* asp = new char [500];
      for (int s = 0; s < NS; s ++)
      {
        fscanf(f, "%s", asp);
        groundTruth[reviewID][s] = string(asp);
      }
      delete [] asp;
    }
    fclose(f);

    f = _fopen(path);
    fscanf(f, "%d %d %d", &R, &V, &K);

    char* aspectName = new char [500];
    NK = new int [K];
    for (int k = 0; k < K; k ++)
    {
      fscanf(f, "%s %d", aspectName, &(NK[k]));
      //NK[k] /= 2;
      //NK[k] ++;
			NK[k] = 3; // Rating to be negative, positive and neutral
      nkTotal += NK[k];
      rAspects[string(aspectName)] = k;
      aspects.push_back(string(aspectName));
    }
    
    for (int k = 0; k < K; k ++)
    {
      aspectNames[k] = vector<string>();
      int nw;
      fscanf(f, "%d", &nw);
      for (int w = 0; w < nw; w ++)
      {
        fscanf(f, "%s", aspectName);
        aspectNames[k].push_back(string(aspectName));
      }
    }
    delete [] aspectName;

    char* junk = new char [500];
    char* itemID = new char [500];
    char* userID = new char [500];
    for (int rid = 0; rid < R; rid ++)
    {
      int reviewID, NS;
      int* vote = new int [K];
      if (fscanf(f, "%d %s %s %s %d", &reviewID, junk, itemID, userID, &NS) != 5)
      {
        printf("Failed to scan review %d\n", reviewID);
        exit(1);
      }
      for (int k = 0; k < K; k ++)
      {
        fscanf(f, "%d", &(vote[k]));
        vote[k] /= 2;
				
        /*if (vote[k] < 0 or vote[k] > NK[k] - 1)
        {
          printf("Got impossible vote for review %d.\n", reviewID);
          exit(1);
        }*/
      }

      review* r = new review(reviewID, string(itemID), string(userID), NS, vote);

      for (int s = 0; s < NS; s ++)
      {
        int NW;
        fscanf(f, "%d", &NW);
        r->NW[s] = NW;
        r->sentences[s] = new int [NW];
        for (int w = 0; w < NW; w ++)
          fscanf(f, "%d", &(r->sentences[s][w]));
      }

      if (groundTruth.find(reviewID) != groundTruth.end() or (rid % 2 == 0 and (rmax == 0 or rid < rmax)))
      {
        if (reviewMap.find(reviewID) != reviewMap.end())
             continue;

        reviews.push_back(r);
        reviewMap[reviewID] = (int) reviews.size() - 1;
        if (groundTruth.find(reviewID) != groundTruth.end() and NS != gtSize[reviewID])
        {
          printf("Review %d has the wrong size (%d != %d).\n", reviewID, NS, gtSize[reviewID]);
          exit(1);
        }
      }
      else
        delete r;
    }
    delete [] junk;
    delete [] itemID;
    delete [] userID;
    fclose(f);
    
    f = _fopen(wordPath);
    char* word = new char [500];
    int wid;
    while (fscanf(f, "%d \"%s", &wid, word) == 2)
    {
      word[strlen(word) - 1] = '\0';
      wordIds[string(word)] = wid;
    }
    fclose(f);
    delete [] word;
    
    R = (int) reviews.size();
  }
  ~corpus()
  {
    delete [] NK;
    for (vector<review*>::iterator it = reviews.begin(); it != reviews.end(); it ++)
      delete *it;
    for (map<int, string*>::iterator it = groundTruth.begin(); it != groundTruth.end(); it ++)
      delete [] it->second;
  }

  map<int, int> reviewMap; // Maps reviewIDs (from the groundtruth) to their index in the corpus
  vector<review*> reviews;
  map<int, string*> groundTruth;
  map<int, int> gtSize;
  
  map<string, int> wordIds;
  
  int R; // Number of reviews
  int V; // Vocabulary size
  int K; // Number of aspects
  int* NK; // Number of unique scores for each aspect
  int nkTotal;

  vector<string> aspects;
  map<string, int> rAspects;
  map<int, vector<string> > aspectNames;
};

/**
 * Aspect and sentiment weights for a particular corpus
 */
class model
{
public:
  corpus* C;

  double*** theta; // Indexed by topic, vote, word
  double** thetaSeg; // Indexed by topic, word
  unsigned int* seedptr;
	
	// Mapping of review, topic to a rating (i.e. the latent labels we wish to learn)
	
  int** rRatings;
  
  int** rTopics; // Mapping of review,sentence to a topic (i.e., the latent labels we wish to learn)
  
  bool supervision;
  
  model(corpus* C, unsigned int* seedptr, bool supervision): C(C), seedptr(seedptr), supervision(supervision)
  {
	
		rRatings = new int* [C->R]; // Added rating dimension
	
    rTopics = new int* [C->R];
    for (int ri = 0; ri < C->R; ri ++)
    {	// initialize ratings
			rRatings[ri] = new int [C->K];
			for (int k = 0; k < C->K; k++)
			{
				rRatings[ri][k] = rand() % 3; // random int between 0 and 2
			}
			
      rTopics[ri] = new int [C->reviews[ri]->NS];
      for (int s = 0; s < C->reviews[ri]->NS; s ++)
        rTopics[ri][s] = -1;
    }

    theta = new double** [C->K];
    thetaSeg = new double* [C->K];
    for (int k = 0; k < C->K; k ++)
    {
      theta[k] = new double* [C->NK[k]];
      thetaSeg[k] = new double [C->V];
      for (int v = 0; v < C->NK[k]; v ++)
        theta[k][v] = new double [C->V];
    }
    for (int k = 0; k < C->K; k ++)
    {
      for (int w = 0; w < C->V; w ++)
      {
        for (int v = 0; v < C->NK[k]; v ++)
          theta[k][v][w] = 0;
        thetaSeg[k][w] = -0.1 * (rand_r(seedptr) / (1.0*RAND_MAX));
      }
      for (vector<string>::iterator it = C->aspectNames[k].begin(); it != C->aspectNames[k].end(); it ++)
        thetaSeg[k][C->wordIds[*it]] = 1;
    }

    if (supervision)
    {
      // Set the topics to be the groundtruth topics
      for (map<int, string*>::iterator it = C->groundTruth.begin(); it != C->groundTruth.end(); it ++)
      {
        int reviewID = it->first;
        string* labels = it->second;
        int rid = C->reviewMap[reviewID];
        
        for (int s = 0; s < C->reviews[rid]->NS; s ++)
        {
          if (labels[s].compare("None") == 0) continue;
          rTopics[rid][s] = C->rAspects[labels[s]];
        }
      }
      
      // Set the weights using the groundtruth
      for (int k = 0; k < C->K; k ++)
        for (int w = 0; w < C->V; w ++)
        {
          for (int v = 0; v < C->NK[k]; v ++)
            theta[k][v][w] = 0;
          thetaSeg[k][w] = 0;
        }

      double** dsl = new double* [C->K];
      double*** dl = new double** [C->K];
      for (int k = 0; k < C->K; k ++)
      {
        dl[k] = new double* [C->NK[k]];
        dsl[k] = new double [C->V];
        for (int v = 0; v < C->NK[k]; v ++)
          dl[k][v] = new double [C->V];
      }
    
      for (int it = 0; it < 25; it ++)
      {
        gradient(dsl, dl, true);
        for (int k = 0; k < C->K; k ++)
          for (int w = 0; w < C->V; w ++)
          {
            for (int v = 0; v < C->NK[k]; v ++)
              theta[k][v][w] += C->nkTotal * 0.01 / C->R * dl[k][v][w];
            thetaSeg[k][w] += C->nkTotal * 0.01 / C->R * dsl[k][w];
          }
      }
        
      for (int k = 0; k < C->K; k ++)
      {
        for (int v = 0; v < C->NK[k]; v ++)
          delete [] dl[k][v];
        delete [] dl[k];
        delete [] dsl[k];
      }
      delete [] dl;
      delete [] dsl;
    }
  }

  ~model()
  {
    for (int k = 0; k < C->K; k ++)
    {
      for (int v = 0; v < C->NK[k]; v ++)
        delete [] theta[k][v];
      delete [] theta[k];
      delete [] thetaSeg[k];
    }
    delete [] theta;
    delete [] thetaSeg;
    
    for (int ri = 0; ri < C->R; ri ++)
    {  delete [] rTopics[ri];
			delete [] rRatings[ri];//added for deleting rRatings
    }
	delete [] rTopics;
		delete [] rRatings;//added
  }
	
  
  // Compute the kappa score of a model
  void eval(double& acc, double& kappa)
  {
    int ngt = 0;
    int nagree = 0;
    int ntotal = 0;
    for (map<int, string*>::iterator it = C->groundTruth.begin(); it != C->groundTruth.end(); it ++)
    {
      int reviewID = it->first;
      string* labels = it->second;
      if (C->reviewMap.find(reviewID) == C->reviewMap.end()) continue;
      ngt ++;
      int rid = C->reviewMap[reviewID];
      int nContentSentences = 0;

      for (int s = 0; s < C->reviews[rid]->NS; s ++)
      {
        if (labels[s].compare("None") == 0) continue;
        nContentSentences ++;
        if (labels[s].compare(C->aspects[rTopics[rid][s]]) == 0)
          nagree ++;
      }
      ntotal += nContentSentences;
    }
    acc = nagree * 1.0/ntotal;
    kappa = (acc - 1.0/C->K) / (1 - 1.0/C->K);
  }
  
  // Print out a precision/recall curve for a model
  void precisionRecall(void)
  {
    for (int k = 0; k < C->K; k ++)
    {
      string label = C->aspects[k];
      int totalRelevantDocs = 0;
      vector<pair<double, int> > scoresLabels;
      for (map<int, string*>::iterator it = C->groundTruth.begin(); it != C->groundTruth.end(); it ++)
      {
        int ri = C->reviewMap[it->first];
        review* r = C->reviews[ri];
        string* ss = it->second;
        for (int s = 0; s < r->NS; s ++)
        {
          //int predictedLabel = rTopics[ri][s];
          if (ss[s].compare("None") == 0) continue; // Ignore ambiguous sentences
          if (ss[s].compare(label) == 0) totalRelevantDocs ++;
          scoresLabels.push_back(pair<double, int>(-sentenceProb(r, s, k, ri), ss[s].compare(label) == 0));
        }
      }
      sort(scoresLabels.begin(), scoresLabels.end());
      printf("plot[\"%s\"] = [", C->aspects[k].c_str());
      for (int i = 0; i < (int) scoresLabels.size(); i ++)
      {
        int nretrieved = i + 1;
        int numer = 0;
        for (int x = 0; x < i + 1; x ++)
          if (scoresLabels[x].second) numer ++;
        double precision = numer * 1.0/nretrieved;
        double recall = numer * 1.0/totalRelevantDocs;
        if (i > 0) printf(", ");
        printf("(%f, %f)", recall, precision);
      }
      printf("]\n");
    }
  }
	

	// Calculating sentence score while fixing ratings
  double sentenceScore(review* r, int s, int k, int ri)
  {
    double score = 0;
    //int v = r->vote[k];
		int v = rRatings[ri][k];
    //if (vote != -1) v = vote;
    for (int i = 0; i < r->NW[s]; i ++)
    {
      int w = r->sentences[s][i];
      score += thetaSeg[k][w] + theta[k][v][w];
    }
    return score;
  }
	
	
	// Calculating sentence score while fixing topics
	double ratingScore(review* r, int k, int v, int ri) // v is the rating of aspect k
	{
		double score = 0;
		
		//squared loss
		double squaredLoss = 0;
		int overall = r->vote[2]; //rating for overall
		
		int rating_sum = 0;
		for (int i = 0; i < C->K; i++)
		{
			if (i == k)
				rating_sum += v;
			
			else
				rating_sum += rRatings[ri][i];
		}
		
		squaredLoss = -pow((static_cast<double>(rating_sum)/C->K - overall), 2);
		
		
		//sentence rating
		double senRatingScore = 0;
		
		for (int s = 0; s < r->NS; s++)
		{
			if (rTopics[ri][s] == k)
			{
				for (int i = 0; i < r->NW[s]; i ++)
				{
					int w = r->sentences[s][i];
					senRatingScore += theta[k][v][w];
				}
			}
		}
		
		score = squaredLoss + senRatingScore;
		return score;
	}
	

  
  double sentenceProb(review* r, int s, int k, int ri) // add ri
  {
    double Z = 0;
    for (int j = 0; j < C->K; j ++)
    {
      Z += exp(sentenceScore(r, s, j, ri));
    }
    return exp(sentenceScore(r, s, k, ri)) / Z;
  }
  
  // Compute the accuracy for the summarization task
  void summarizeAcc(double& acc, double& kappa)
  {
    int ntotal = 0;
    int nagree = 0;

    for (map<int, string*>::iterator it = C->groundTruth.begin(); it != C->groundTruth.end(); it ++)
    {
      int ri = C->reviewMap[it->first];
      string* labels = it->second;
      review* r = C->reviews[ri];
      if (r->NS < C->K) continue;

      Matrix<double> matrix(r->NS, C->K);

      for (int i = 0; i < r->NS; i ++)
        for (int j = 0; j < C->K; j ++)
          matrix(i,j) = -sentenceScore(r, i, j, ri);

      Munkres m;
      m.solve(matrix);
      for (int i = 0; i < r->NS; i ++)
        for (int j = 0; j < C->K; j ++)
        {
          if (matrix(i,j) == 0)
          {
            // Sentence i has topic j
            if (labels[i].compare("None"))
            {
              ntotal ++;
              if (labels[i].compare(C->aspects[j]) == 0)
                nagree ++;
            }
          }
        }
      acc = nagree * 1.0/ntotal;
      kappa = (acc - 1.0/C->K) / (1 - 1.0/C->K);
    }
  }
	
	int updateRatings(void)
	{
		int changed = 0;
		
		#pragma omp parallel for
		for (int ri = 0; ri < (int) C->reviews.size(); ri ++)
		{
			review* r = C->reviews[ri];
			int* bestRating = new int [C->K];
			
			for (int k = 0; k < C->K; k ++)
			{
				double rScore = -numeric_limits<double>::max();
				for (int v = 0; v < C->NK[k]; v ++)
				{
					double score = ratingScore(r, k, v, ri);
					if (score > rScore)
					{
						rScore = score;
						bestRating[k] = v;
					}
				}
			}
		
			for (int k = 0; k < C->K; k ++)
			{
				if (rRatings[ri][k] != bestRating[k])
				{
					#pragma omp critical
					changed ++;
				}
		
				rRatings[ri][k] = bestRating[k];
		
			}
			
			delete [] bestRating;
		}
		return changed;
	}


  int updateTopics(void)
  {
    int slackness = 0;
    int changed = 0;
    
    #pragma omp parallel for
    for (int ri = 0; ri < (int) C->reviews.size(); ri ++)
    {
      review* r = C->reviews[ri];
      int* bestTopic = new int [r->NS];
      Matrix<double> matrix(r->NS, r->NS + slackness);

      // Change this so that one topic can be discarded.
      for (int i = 0; i < r->NS; i ++)
      {
        double bestScore = -numeric_limits<double>::max();
        bestTopic[i] = -1;
        for (int k = 0; k < C->K; k ++)
        {
          double score = sentenceScore(r, i, k, ri); // add ri
          if (score > bestScore)
          {
            bestScore = score;
            bestTopic[i] = k;
          }
        }

        for (int j = 0; j < r->NS + slackness; j ++)
        {
          if (j < C->K and r->NS >= C->K)
            matrix(i,j) = -sentenceScore(r, i, j, ri);
          else
            matrix(i,j) = -bestScore;
        }
      }
      
      Munkres m;
      m.solve(matrix);
      for (int i = 0; i < r->NS; i ++)
      {
        for (int j = 0; j < r->NS + slackness; j ++)
        {
          if (matrix(i,j) == 0)
          {
            if (j < C->K and r->NS >= C->K)
              bestTopic[i] = j;
          }
        }
        
        if (supervision and
            C->groundTruth.find(r->reviewID) != C->groundTruth.end() and
            C->groundTruth[r->reviewID][i].compare("None") != 0)
        {
          continue;
        }

        if (rTopics[ri][i] != bestTopic[i])
        {
          #pragma omp critical
          {
            changed ++;
          }
        }
        rTopics[ri][i] = bestTopic[i];
      }
      delete [] bestTopic;
    }
    return changed;
  }

  void gradient(double** dst, double*** dt, bool init_supervision)
  {
    for (int k = 0; k < C->K; k ++)
      for (int w = 0; w < C->V; w ++)
      {
        // Regularizer
        dst[k][w] = -1*C->nkTotal*thetaSeg[k][w];
        for (int v = 0; v < C->NK[k]; v ++)
          dt[k][v][w] = -1*C->nkTotal*theta[k][v][w];
      }

    for (int ri = 0; ri < (int) C->reviews.size(); ri ++)
    {
      review* r = C->reviews[ri];
      if (init_supervision and C->groundTruth.find(r->reviewID) == C->groundTruth.end()) continue;
      for (int s = 0; s < r->NS; s ++)
      {
        int t = rTopics[ri][s];
        if (init_supervision and t == -1) continue;
        //int v = r->vote[t]; // use rRating instead
				int v = rRatings[ri][t]; // newly added
				
        double numer = 0;
        double denom = 0;
        for (int k = 0; k < C->K; k ++)
        {
          double expScore = 0;
          for (int i = 0; i < r->NW[s]; i ++)
          {
            int w = r->sentences[s][i];
						expScore += thetaSeg[k][w] + theta[k][rRatings[ri][k]][w];
            //expScore += thetaSeg[k][w] + theta[k][r->vote[k]][w];
          }
          expScore = exp(expScore);
          if (k == t) numer = expScore;
          denom += expScore;
        }

        double frac = numer/denom;

        for (int i = 0; i < r->NW[s]; i ++)
        {
          int w = r->sentences[s][i];
          {
            dst[t][w] += 1 - frac;
            dt[t][v][w] += 1 - frac;
          }
        }
      }
    }
  }

  double logLikelihood(void)
  {
    double ll = 0;

    for (int ri = 0; ri < (int) C->reviews.size(); ri ++)
    {
      review* r = C->reviews[ri];
      for (int s = 0; s < r->NS; s ++)
      {
        int t = rTopics[ri][s];
        double denom = 0;
        for (int k = 0; k < C->K; k ++)
        {
          double expScore = 0;
          for (int i = 0; i < r->NW[s]; i ++)
          {
            int w = r->sentences[s][i];
            //expScore += thetaSeg[k][w] + theta[k][r->vote[k]][w];
						expScore += thetaSeg[k][w] + theta[k][rRatings[ri][k]][w];
          }
          if (k == t) ll += expScore;
          expScore = exp(expScore);
          denom += expScore;
        }
        ll -= log(denom);
      }
    }

    return ll;
  }

  void train(int I, int GI, double lrate, string outputSeg, string outputSen)
  {
    double** dsl = new double* [C->K];
    double*** dl = new double** [C->K];
    for (int k = 0; k < C->K; k ++)
    {
      dl[k] = new double* [C->NK[k]];
      dsl[k] = new double [C->V];
      for (int v = 0; v < C->NK[k]; v ++)
        dl[k][v] = new double [C->V];
    }

    double ll = 0;
    double ll_prev = 0;
    int gi = 0;
    
    double accuracy = 0;
    double kappa = 0;

    for (int i = 0; i < I; i ++)
    {
      int changed = updateTopics();
			int changed2 = updateRatings(); // updateRatings
			
      if (i == 0)
        eval(accuracy, kappa);

      if (changed == 0 and changed2 == 0 and gi != GI) break;

      ll = logLikelihood();
      if (i == 0) ll_prev = ll;

      if (ll < ll_prev*1.01)
      {
        printf("Decrease in logLikelihood at line %d.\n", __LINE__);
        break;
      }

      for (gi = 0; gi < GI; gi ++)
      {
        ll_prev = ll;
        gradient(dsl, dl, false);
        for (int k = 0; k < C->K; k ++)
          for (int w = 0; w < C->V; w ++)
          {
            for (int v = 0; v < C->NK[k]; v ++)
              theta[k][v][w] += lrate*dl[k][v][w];
            thetaSeg[k][w] += lrate*dsl[k][w];
          }

        ll = logLikelihood();
        if (not (ll > ll_prev))
        {
          for (int k = 0; k < C->K; k ++)
            for (int w = 0; w < C->V; w ++)
            {
              thetaSeg[k][w] -= lrate*dsl[k][w];
              for (int v = 0; v < C->NK[k]; v ++)
                theta[k][v][w] -= lrate*dl[k][v][w];
            }
          ll = ll_prev;
          break;
        }
      }
      
      double** pikw = new double* [C->K];
      for (int k = 0; k < C->K; k ++)
      {
        pikw[k] = new double [C->V];
        for (int w = 0; w < C->V; w ++)
        {
          pikw[k][w] = 0;
          for (int v = 0; v < C->NK[k]; v ++)
            pikw[k][w] += theta[k][v][w];
          pikw[k][w] /= C->NK[k];
        }
      }
      
      for (int k = 0; k < C->K; k ++)
        for (int w = 0; w < C->V; w ++)
        {
          for (int v = 0; v < C->NK[k]; v ++)
            theta[k][v][w] -= pikw[k][w];
          thetaSeg[k][w] += pikw[k][w];
        }
        
      for (int k = 0; k < C->K; k ++)
        delete [] pikw[k];
      delete [] pikw;

      ll_prev = ll;
      #pragma omp critical
      {
        eval(accuracy, kappa);
        if (ll > bestll)
        {
          bestll = ll;
          printf("Iteration %d, thread %d:\tll = %f, accuracy = %f, kappa = %f\n", i + 1, omp_get_thread_num(), ll, accuracy, kappa);
          saveModel(outputSeg, outputSen);
        }
      }
    }

    for (int k = 0; k < C->K; k ++)
    {
      for (int v = 0; v < C->NK[k]; v ++)
        delete [] dl[k][v];
      delete [] dl[k];
      delete [] dsl[k];
    }
    delete [] dl;
    delete [] dsl;
  }

  void saveModel(string outputSeg, string outputSen)
  {
    FILE* f = fopen(outputSeg.c_str(), "w");
    for (int z = 0; z < C->K; z ++)
    {
      fprintf(f, "%s ", C->aspects[z].c_str());

      for (int w = 0; w < C->V; w ++)
        fprintf(f, " %f", thetaSeg[z][w]);
      fprintf(f, "\n");
    }
    fclose(f);

    f = fopen(outputSen.c_str(), "w");
    for (int z = 0; z < C->K; z ++)
      for (int v = 0; v < C->NK[z]; v ++)
      {
        fprintf(f, "%s %d", C->aspects[z].c_str(), v);

        for (int w = 0; w < C->V; w ++)
          fprintf(f, " %f", theta[z][v][w]);
        fprintf(f, "\n");
      }
    fclose(f);
  }
  
  void loadModel(string inputSeg, string inputSen)
  {
    FILE* f = _fopen(inputSeg);
    char* topic = new char [500];
    for (int z = 0; z < C->K; z ++)
    {
      fscanf(f, "%s", topic);
      int t = C->rAspects[string(topic)];
      for (int w = 0; w < C->V; w ++)
      {
        float weight = 0;
        fscanf(f, "%f", &weight);
        thetaSeg[t][w] = weight;
      }
    }
    fclose(f);
    f = _fopen(inputSen);
    for (int z = 0; z < C->K; z ++)
    {
      fscanf(f, "%s", topic);
      int t = C->rAspects[string(topic)];

      for (int v = 0; v < C->NK[t]; v ++)
      {
        if (v > 0) fscanf(f, "%s", topic);
        int v_;
        fscanf(f, "%d", &v_);
        for (int w = 0; w < C->V; w ++)
        {
          float weight = 0;
          fscanf(f, "%f", &weight);
          theta[t][v][w] = weight;
        }
      }
    }
    fclose(f);
    delete [] topic;
  }
};

void unsupervised(string reviewFile, string widFile, string gtFile, string segOut, string senOut)
{
  corpus C(reviewFile, widFile, gtFile, 0);
  #pragma omp parallel for
  for (int t = 0; t < omp_get_num_procs(); t ++)
  {
    unsigned int tseed = t;
    model M(&C, &tseed, false);
    M.train(100, 25, C.nkTotal * 0.01 / C.R, segOut, senOut);
  }
}

void supervised(string reviewFile, string widFile, string gtFile1, string gtFile2, string segOut, string senOut)
{
  corpus C(reviewFile, widFile, gtFile1, 0);
  unsigned int tseed = 0;
  model M(&C, &tseed, true);
  //M.loadModel(segUns, senUns);
  M.train(100, 25, C.nkTotal * 0.01 / C.R, segOut, senOut);
  
  corpus C2(reviewFile, widFile, gtFile2, 0);
  unsigned int tseed2 = 0;
  model M2(&C2, &tseed2, false);
  M2.loadModel(segOut, senOut);
  M2.updateTopics();
  double accuracy, kappa;
  M2.eval(accuracy, kappa);
  printf("accuracy = %f, kappa = %f\n", accuracy, kappa);
}

void evaluate(string reviewFile, string widFile, string gtFile, string segFile, string senFile)
{
  corpus C(reviewFile, widFile, gtFile, 0);
  unsigned int tseed = 0;
  model M(&C, &tseed, false);
  M.loadModel(segFile, senFile);
  M.updateTopics();
  double accuracy, kappa;
  M.eval(accuracy, kappa);
  printf("%s/%s: segmentation accuracy  = %f, kappa = %f\n", reviewFile.c_str(), segFile.c_str(), accuracy, kappa);
  M.summarizeAcc(accuracy, kappa);
  printf("%s/%s: summarization accuracy = %f, kappa = %f\n", reviewFile.c_str(), segFile.c_str(), accuracy, kappa);
}

int main(int argc, char** argv)
{
  unsupervised("reviews/reviewsBA.txt", // The review corpus
               "wordids/wordidsBA.txt", // Word ids
               "groundtruth/groundtruthBA1.txt", // Labeled groundtruth data
               "models/modelSegBA.out", // Where to save the segmentation output
               "models/modelSenBA.out"); // Where to save the sentiment output

  evaluate("reviews/reviewsBA.txt", "wordids/wordidsBA.txt", "groundtruth/groundtruthBA1.txt", "models/modelSegBA.out", "models/modelSenBA.out");
  printf("Finished!");
  return 0;
}
