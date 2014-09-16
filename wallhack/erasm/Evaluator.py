"""
Report the proportion of predicted similar authors
who are existing contacts and have interests in common.

Other possibilities:
* groups in common
* high-level discipline in common
"""

from collections import defaultdict
from operator import itemgetter
from optparse import OptionParser
import numpy as np
from nltk.stem import snowball

def evaluate_against_contacts(sims, contacts, min_contacts):
    stats_names = ['# predicted','# actual','prec @ 1','prec @ 3','prec @ 5','prec @ 10','prec','recall @ 1','recall @ 3','recall @ 5','recall @ 10','recall','f1']
    stats = []
    too_few_contacts = 0
    no_contacts = 0
    contacts_match = 0
    for author,predicted in sims.iteritems():
        if author not in contacts:
            no_contacts += 1
            continue
        actual = contacts[author]
        if len(actual) < min_contacts:
            if len(actual) > 0:
                too_few_contacts += 1
            continue
        predicted.sort(key=itemgetter(1),reverse=True)  # sort by similarity score
        correct = sum(1 for sim,_ in predicted if sim in actual)
        if correct > 0:
            contacts_match += 1
        prec = float(correct)/len(predicted)
        recall = float(correct)/len(actual)
        if prec == 0 or recall == 0:
            f1 = 0
        else:
            f1 = 2*prec*recall/(prec+recall)
        prec_1 = sum(1.0 for sim,_ in predicted[:1] if sim in actual)/len(predicted[:1])
        prec_3 = sum(1.0 for sim,_ in predicted[:3] if sim in actual)/len(predicted[:3])
        prec_5 = sum(1.0 for sim,_ in predicted[:5] if sim in actual)/len(predicted[:5])
        prec_10 = sum(1.0 for sim,_ in predicted[:10] if sim in actual)/len(predicted[:10])

        recall_1 = sum(1.0 for sim,_ in predicted[:1] if sim in actual)/len(actual)
        recall_3 = sum(1.0 for sim,_ in predicted[:3] if sim in actual)/len(actual)
        recall_5 = sum(1.0 for sim,_ in predicted[:5] if sim in actual)/len(actual)
        recall_10 = sum(1.0 for sim,_ in predicted[:10] if sim in actual)/len(actual)

        stats.append((len(predicted),len(actual),prec_1,prec_3,prec_5,prec_10,prec,recall_1,recall_3,recall_5,recall_10,recall,f1))

    print 'predicted similar authors to {0} authors with at least {1} contacts'.format(len(stats), min_contacts)
    print 'authors with no contacts {0}'.format(no_contacts)
    print 'authors with less than {0} contacts {1}'.format(min_contacts, too_few_contacts)
    print 'authors with at least 1 contact match {0}'.format(contacts_match)
    for i,stat in enumerate(stats_names):
        print 'mean {0} = {1:.2g} '.format(stat,np.mean([s[i] for s in stats]))
        
    precisions = np.array([prec_1, prec_3, prec_5, prec_10])
    recalls = np.array([recall_1, recall_3, recall_5, recall_10])
    
    return precisions, recalls, f1 

def evaluate_against_research_interests(sims, research_interests, min_acceptable_sims):
    stats_names = ['# predicted having interests','prec @ 1','prec @ 3','prec @ 5','prec @ 10','prec','jacc @ 10','jacc']
    stats = []
    too_few_sims = 0
    sims_no_interest = 0
    sims_with_interest_in_common = 0
    for author,predicted in sims.iteritems():
        if len(predicted) < min_acceptable_sims:
            too_few_sims += 1
        if author not in research_interests:
            continue
        predicted.sort(key=itemgetter(1),reverse=True)  # sort by similarity score
        outcomes = []
        dists = []
        for sim,_ in predicted:
            if sim not in research_interests:
                continue
            joint_interests = research_interests[author].intersection(research_interests[sim])
            outcomes.append(int(bool(joint_interests)))
            num_shared = len(joint_interests)
            num_overall = len(research_interests[author].union(research_interests[sim]))
            dists.append(float(num_shared)/num_overall)
        if not outcomes:
            sims_no_interest += 1
            continue
        else:
            if sum(outcomes) > 0:
                sims_with_interest_in_common += 1

        # need to know the number of potential matching authors to give recall stats
        # just give precision stats for now
        prec = float(sum(outcomes))/len(outcomes)
        prec_1 = float(sum(outcomes[:1]))/len(outcomes[:1])
        prec_3 = float(sum(outcomes[:3]))/len(outcomes[:3])
        prec_5 = float(sum(outcomes[:5]))/len(outcomes[:5])
        prec_10 = float(sum(outcomes[:10]))/len(outcomes[:10])
        jaccard = sum(dists)/len(dists)
        jaccard_10 = sum(dists[:10])/len(dists[:10])
        s = (len(outcomes),prec_1,prec_3,prec_5,prec_10,prec,jaccard_10,jaccard)
        stats.append(s)

    print 'authors with less than {0} sims = {1} ({2:.1f}%)'.format(min_acceptable_sims,too_few_sims,float(too_few_sims)/len(sims))

    print 'predicted similar authors to {0} authors with research interests'.format(len(stats))
    print 'authors with no similar authors with interest {0}'.format(sims_no_interest)
    print 'authors with at least 1 similar author with interest in common {0}'.format(sims_with_interest_in_common)
    for i,stat in enumerate(stats_names):
        print 'mean {0} = {1:.2g} '.format(stat,np.mean([s[i] for s in stats]))
        
    precisions = np.array([prec_1, prec_3, prec_5, prec_10])
    return precisions 

def read_contacts(filename): 
    # load profile_id->contacts groundtruth: consider sim correct if it's one of these
    print 'loading contacts...'
    contacts = defaultdict(set)
    for line in open(filename):
        a,b = map(int,line.strip().split('\t')[:2])
        # contacts are not symmetrical
        contacts[a].add(b)
    
    return contacts 


def read_interests(filename): 
    # load profile_id->research interests groundtruth: hit if (hopefully significant) term in common
    print 'loading research interests...'
    research_interests = defaultdict(set)
    for line in open(filename):
        profile_id,interest = map(int,line.strip().split('\t'))
        research_interests[int(profile_id)].add(interest)
        
    return research_interests

def read_similar_authors(filename, min_score): 
    # TODO: in practice we'll enforce a genuine name check on similar authors
    # to be sure they really are authors
    # -- this should be a restriction on including users in the sims dataset

    print 'loading similar authors with score >= {0}...'.format(min_score)
    sims = defaultdict(list)
    for line in open(filename):
        author,sim,score = line.strip().split('\t')
        author = int(author)
        sim = int(sim)
        
        score = float(score)

        if score >= min_score:
            sims[author].append((sim,score))

    return sims 

if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option('-c','--contacts',dest='contacts',help='contacts tsv file')
    parser.add_option('-r','--research_interests',dest='research_interests',help='research interests tsv file')
    parser.add_option('-s','--sims',dest='sims',help='similar authors tsv file, each line contains author,similar_author,score')
    parser.add_option('--min_score',dest='min_score',type='float',default=2,help='ignore sims below this score (default: %default)')
    parser.add_option('--min_contacts',dest='min_contacts',type='int',default=3,help='evaluate only on authors with this many contacts (default: %default)')
    parser.add_option('--min_acceptable_sims',dest='min_acceptable_sims',type='int',default=3,help='show how many users get less sims than this (default: %default)')

    (opts,args) = parser.parse_args()
    if not opts.contacts or not opts.sims or not opts.research_interests:
        parser.print_help()
        raise SystemExit(1)

    contacts = read_contacts(opts.contacts)
    research_interests = read_interests(opts.interests)
    sims = read_similar_authors(opts.sims, opts.min_score)

    print 'evaluating against contacts...'
    evaluate_against_contacts(sims, contacts, opts.min_contacts)

    # just exclude from results where we have no groundtruth, so need to
    # quantify what proportion of results are actually evaluated
    print 'evaluating against research interests...'
    evaluate_against_research_interests(sims, research_interests, opts.min_acceptable_sims)
