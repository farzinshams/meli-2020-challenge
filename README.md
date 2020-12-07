# meli-2020-challenge
5th place solution to the Mercado Livre product recommendation challenge



The solution consists of three parts:
1) Get each datapoint's most likely domain using ComplementNB, on viewed items only.
2) Get each datapoint's most likely domain using bag of words of queries, also using ComplementNB.
3) Create a scoring dataset and classify items with lightgbm. To do this, iterate over datapoints and select most relevant items for each datapoint. Most relevant items are:
viewed items, most bought items from most likely domain (defined by the two previous steps). For each candidate solution, extract relevant information, like:
 - how many times item was viewed (int)
 - item was first viewed (binary)
 - item was last viewed (binary)
 - number of domains viewed
 - number of recurrences to item
 - probability of its viewed domain
 - probability of its queried domain
 - probability of item in its domain
 - etc.
 
 For prediction:
 1) Iterate over datapoints, extract candidate solutions, get score from lightgbm and sort top 10 in descending order.
