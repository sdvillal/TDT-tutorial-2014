# trains and evaluates a LR model with a given fingerprint
# and does the prediction for the test molecules

from __future__ import print_function

try:
    import cPickle as pickle
except ImportError:
    import pickle
import gzip
import os
import sys

from optparse import OptionParser

from sklearn.linear_model import LogisticRegression

from rdkit.ML.Scoring import Scoring

# common functions
sys.path.insert(0, os.getcwd()+'/')
sys.path.insert(0, os.getcwd()+'/../')
import common_functions as cf

path = os.getcwd() + '/'
inpath = path + '../data/'

# prepare command-line option parser
usage = "usage: %prog [options] arg"
parser = OptionParser(usage)
parser.add_option("-f", "--fingerprint", dest="fp",
                  help="fingerprint name (options are: %s)" % ', '.join(sorted(cf.fp_dict)))

# read in command line options
(options, args) = parser.parse_args()
# required arguments
if options.fp:
    fpname = options.fp
else:
    raise RuntimeError('fingerprint name missing')
print("ML model is trained with", fpname)

# read the actives
fps_act = []
for line in open(inpath+'training_actives_cleaned.dat', 'rt'):
    line = line.strip().split()
    # contains: [sample_id, hit, pec50, smiles]
    fp = cf.getNumpyFP(line[3], fpname, 'float')
    if fp is not None:
        fps_act.append(fp)
num_actives = len(fps_act)
print("actives read and fingerprints calculated:", num_actives)

# read the inactives
fps_inact = []
for line in open(inpath+'training_inactives_cleaned.dat', 'rt'):
    line = line.strip().split()
    # contains: [sample_id, hit, pec50, smiles]
    fp = cf.getNumpyFP(line[3], fpname, 'float')
    if fp is not None:
        fps_inact.append(fp)
num_inactives = len(fps_inact)
print("inactives read and fingerprints calculated:", num_inactives)

# test lists
test_indices_act = []
for line in gzip.open(path+'../test_lists/test_lists_10percent_actives.dat.gz', 'rt'):
    line = line.strip().split()
    test_indices_act.append(set([int(l) for l in line]))
test_indices_inact = []
for line in gzip.open(path+'../test_lists/test_lists_10percent_inactives.dat.gz', 'rt'):
    line = line.strip().split()
    test_indices_inact.append(set([int(l) for l in line]))
print("test lists read in:", len(test_indices_act), len(test_indices_inact))

# loop over the repetitions
infile = gzip.open(path+'scores/lists_'+fpname+'.pkl.gz', 'rb')
rankfile = gzip.open(path+'scores/ranks_actives_lr_'+fpname+'.pkl.gz', 'wb')
outfile = open(path+'analysis/output_analysis_lr_'+fpname+'.dat', 'wt')
for i in range(cf.num_rep):
    print("repetition", i)

    # training 
    train_indices_act = set(range(num_actives)).difference(test_indices_act[i])
    train_indices_inact = set(range(num_inactives)).difference(test_indices_inact[i])
    train_fps = [fps_act[j] for j in train_indices_act]
    train_fps += [fps_inact[j] for j in train_indices_inact]
    ys_fit = [1]*len(train_indices_act) + [0]*len(train_indices_inact)
    # train the model
    ml = LogisticRegression()
    ml.fit(train_fps, ys_fit)

    # chemical similarity
    simil = pickle.load(infile)

    # ranking
    test_fps = [fps_act[j] for j in test_indices_act[i]]
    test_fps += [fps_inact[j] for j in test_indices_inact[i]]
    scores = [[pp[1], s[0], s[1]] for pp, s in zip(ml.predict_proba(test_fps), simil)]

    # write ranks for actives
    cf.writeActiveRanks(scores, rankfile, num_actives)

    scores.sort(reverse=True)

    # evaluation
    auc = Scoring.CalcAUC(scores, -1)
    ef = Scoring.CalcEnrichment(scores, -1, [0.05])

    # write out
    outfile.write("%i\t%.10f\t%.10f\n" % (i, auc, ef[0]))

infile.close()
rankfile.close()
outfile.close()
