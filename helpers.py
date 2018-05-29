import simple_id
import torch
import os
import time
import gensim
import sys
import seaborn
import warnings
warnings.filterwarnings('ignore')

modelFname = 'data/BiRNN-2-128-260.pt'
fullFname = 'data/full_ss_500.tsv'
tokens = 'data/tokens.csv'



seaborn.set_context("paper", rc={
                                 "lines.linewidth": 2,
                                })
seaborn.set(rc = {
                 "font.size": 30,
                 'axes.titlesize' : 30,
                 'axes.labelsize' : 25,
                 'xtick.labelsize': 15,
                 'ytick.labelsize': 15,
                 'legend.fontsize': 15,
                 'axes.facecolor' : '#f5f5f5',
                 'lines.solid_capstyle': 'projecting',
                 'font.family': ['sans-serif'],
                 'font.sans-serif' :['Tahoma', 'DejaVu Sans', 'Lucida Grande', 'Verdana'],
                 })
seaborn.set_palette(seaborn.color_palette("Set2", 10))

seaborn.axes_style()

def get_classes(s):
    try:
        s = s.upper()
    except AttributeError:
        return set()
    vals = set()
    for k in socialSSs.keys():
            if k in s:
                vals.add(socialSSs[k])
    return sorted(list(vals))

def loadSubject(targetSubject):
    #load full datasets
    df_full = pandas.read_csv(fullFname, sep = '\t', error_bad_lines = False)
    df_cs = pandas.read_csv(csJournsFname, sep = '\t', error_bad_lines = False)
    wos_classes = simple_id.read_WOS_CLasses(wos_classifications)

    #Label as CS
    CSs = {r['WoS_Description']: r['Description'] for i, r  in wos_classes.iterrows() if r['Description'] == 'Computer and information sciences' and r['sub_class'] > 0}
    socialSSs = {r['WoS_Description']: r['Description'] for i, r  in wos_classes.iterrows() if r['main_class'] == 'SOCIAL SCIENCES' and r['sub_class'] > 0}
    comp_sources = set(df_cs['source'])
    soc_sources = set(df_full['source'])
    df_full['is_comp'] = df_full['source'].apply(lambda x: True if x in comp_sources else False)

    #Get subject labels
    df_ssJourns = df_full[['source', 'subject_con']].groupby('source').max()
    df_ssJourns['subjects'] = df_ssJourns['subject_con'].apply(get_classes)
    journToSubjects = {i : r['subjects'] for i, r in df_ssJourns.iterrows()}
    df_full[targetSubject] = df_full['source'].apply(lambda x: True if targetSubject in journToSubjects[x] else False)
    return df_full[df_full[targetSubject]]

class TokensLookup(object):
    def __init__(self, fname):
        self.fname = fname
        self._fetched = {}
        with open(fname) as f:
            self.mapping = {s[0] : ',"'.join(s[1:]) for s in (l.split(',"') for l in f)}
    def __getitem__(self, key):
        if key not in self._fetched:
            val = self.mapping[key][:-2].split('","')
            self._fetched[key] = (val[0].split(' '), [s.split(' ') for s in val[1].split('|')])
        return self._fetched[key]

Tokens = TokensLookup(tokens)

def makeVaryingArray(row_dict, Net, w2v):

    preds = []
    for i in range(len(row_dict['title_tokens'])):
        predT = []
        for j in range(len(row_dict['abstract_tokens'])):
            newDict = {
                'abstract_tokens' : row_dict['abstract_tokens'][:j+1],
                'title_tokens' : row_dict['title_tokens'][:i + 1],
                }
            pred = Net.predictRow(newDict, w2v=w2v)
            predT.append(float(pred['probPos']))
        preds.append(predT)
    return preds
