# rank orders a list of commercially available compounds
# using a fusion model
from __future__ import print_function

try:
    import cPickle as pickle
except ImportError:
    import pickle
import gzip
import os
import sys

from optparse import OptionParser

from joblib import Parallel, delayed
import itertools
import time

# common functions
sys.path.insert(0, os.getcwd()+'/../')
import common_functions as cf

path = os.getcwd() + '/'
inpath = path + '../data/'

# prepare command-line option parser
usage = "usage: %prog [options] arg"
parser = OptionParser(usage)
parser.add_option("--f4096", action="store_true", dest="f4096",
                  help="use fingerprint size of 4096")
parser.add_option("--top", type=int, default=10000, dest="top",
                  help="number of top compounds to save")
parser.add_option("--jobs", type=int, default=4, dest="jobs",
                  help="number of processes to use, joblib semantics")
parser.add_option("--chunksize", type=int, default=1000, dest="chunk_size",
                  help="size of each chunk for parallel processing")
# read in command line options
(options, args) = parser.parse_args()

# load the ML models
if options.f4096:
    lr_rdk5 = pickle.load(gzip.open(path+'../final_models/lr_rdk5-4096_model.pkl.gz', 'rb'))
    rf_rdk5 = pickle.load(gzip.open(path+'../final_models/rf_rdk5-4096_model.pkl.gz', 'rb'))
    rf_morgan2 = pickle.load(gzip.open(path+'../final_models/rf_morgan2-4096_model.pkl.gz', 'rb'))
else:
    lr_rdk5 = pickle.load(gzip.open(path+'../final_models/lr_rdk5_model.pkl.gz', 'rb'))
    rf_rdk5 = pickle.load(gzip.open(path+'../final_models/rf_rdk5_model.pkl.gz', 'rb'))
    rf_morgan2 = pickle.load(gzip.open(path+'../final_models/rf_morgan2_model.pkl.gz', 'rb'))
print("rf models loaded")


# loop over commercial products
def score(lines, lr_rdk5, rf_rdk5, rf_morgan2):
    proba_lr_rdk5 = []
    proba_rf_rdk5 = []
    proba_rf_morgan2 = []
    mols = []
    start = time.time()
    for i, line in enumerate(lines):
        if i > 0 and i % 500 == 0:
            taken = time.time() - start
            print('Ranked %d compounds in %.2f seconds (%.2f cpds/s)' % (i, taken, i / taken))
        if line[0] == "#":
            continue
        line = line.rstrip().split()
        # contains: [smiles, identifier]
        # RDK5
        fp = cf.getNumpyFP(line[0], 'rdk5' if not options.f4096 else 'rdk5-4096', 'float')
        proba_lr_rdk5.append(lr_rdk5.predict_proba(fp.reshape(1, -1))[0][1])
        proba_rf_rdk5.append(rf_rdk5.predict_proba(fp.reshape(1, -1))[0][1])
        fp = cf.getNumpyFP(line[0], 'morgan2' if not options.f4096 else 'morgan2-4096', 'float')
        proba_rf_morgan2.append(rf_morgan2.predict_proba(fp.reshape(1, -1))[0][1])
        mols.append((line[1], line[0]))
    return proba_lr_rdk5, proba_rf_rdk5, proba_rf_morgan2, mols


def chunker(chunksize, iterable):
    it = iter(iterable)
    while True:
       chunk = tuple(itertools.islice(it, chunksize))
       if not chunk:
           return
       yield chunk

with gzip.open(path+'commercial_cmps_cleaned.dat.gz', 'rt') as lines:
    results = Parallel(pre_dispatch='4 * n_jobs', n_jobs=options.jobs)(
        delayed(score)(lines,
                       lr_rdk5,
                       rf_rdk5,
                       rf_morgan2)
        for lines in chunker(options.chunk_size, lines)
    )

proba_lr_rdk5 = []
proba_rf_rdk5 = []
proba_rf_morgan2 = []
mols = []
for plr, prr, prm, m in results:
    proba_lr_rdk5 += plr
    proba_rf_rdk5 += prr
    proba_rf_morgan2 += prm
    mols += m
print("probabilities calculated")

# load similarities
if options.f4096:
    scores_rdk5 = pickle.load(gzip.open(path+'scores_rdk5-4096.pkl.gz', 'rb'))
    scores_morgan2 = pickle.load(gzip.open(path+'scores_morgan2-4096.pkl.gz', 'rb'))
else:
    scores_rdk5 = pickle.load(gzip.open(path+'scores_rdk5.pkl.gz', 'rb'))
    scores_morgan2 = pickle.load(gzip.open(path+'scores_morgan2.pkl.gz', 'rb'))
"similarities loaded"

# assign ranks
scores_lr_rdk5 = cf.assignRanks(proba_lr_rdk5, scores_rdk5)
scores_rf_rdk5 = cf.assignRanks(proba_rf_rdk5, scores_rdk5)
scores_rf_morgan2 = cf.assignRanks(proba_rf_morgan2, scores_morgan2)
print("ranks assigned")

# fusion
fusion_scores = []
for m1, m2, m3, m in zip(scores_lr_rdk5, scores_rf_rdk5, scores_rf_morgan2, mols):
    rank = max([m1[0], m2[0], m3[0]])  # maximum rank
    pp = max([m1[1], m2[1], m3[1]])    # maximum probability
    # store: [max rank, max proba, similarity, identifier, smiles]
    fusion_scores.append([rank, pp, m1[2], m[0], m[1]])
# sort by descending rank
fusion_scores.sort(reverse=True)
print("fusion done")

# write out
fn = 'ranked_list_top{top}_commercial_cmps-{fpsize}.dat.gz'.format(
    top=options.top,
    fpsize='fpsize=4096' if options.f4096 else 'fpsize=1024-2048'
)
outfile = gzip.open(path+fn, 'wt')
outfile.write("#Identifier\tSMILES\tMax_Rank\tMax_Proba\tSimilarity\n")
for r, pp, s, idx, smiles in fusion_scores[:options.top]:
    outfile.write("%s\t%s\t%i\t%.5f\t%.5f\n" % (idx, smiles, r, pp, s))
outfile.close()
print("list written")
