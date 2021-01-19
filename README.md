# search engine scoring
Scores the dataset on the basis of quiries present in the XML file.
Uses the following measure to perform the scoring:
1. Okapi TF
2. TF-IDF
3. Okapi-BM125
4. Jelinek Mercer Smoothing

Saves in the relevant txt files corresponding to the scoreing measure. Average precision is calclated using the gap.py file.
Command line Syntax --> ./gap.py corpus.qrel BM25.txt -v
