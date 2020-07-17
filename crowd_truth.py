import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from collections import Counter

class CrowdTruth:

    def __init__(self, csv_file):
        self.df, self.clarity_df = self.load_for_crowdtruth(pd.read_csv(csv_file))

    def load_for_crowdtruth(self, df):
        # get all labels
        labels_counter = Counter()

        for ind, answer in df['Answer.votes'].iteritems():
            answers = answer.split('|')
            labels_counter.update(answers)

        # reformat
        reformat_rows = []
        for ind, row in df.iterrows():
            answers = row['Answer.votes'].split("|")
            reformat_row = {'WorkerId':row['WorkerId'],
                             'HITId': row['HITId'],
                             'WorkTimeInSeconds':row['WorkTimeInSeconds']}
            for answer in answers:
                reformat_row[answer] = 1
            reformat_rows.append(reformat_row)

        reformat_df = pd.DataFrame(reformat_rows).replace(np.nan, 0)
        clarity_df = reformat_df.groupby('HITId').agg({label:sum for label in labels_counter.keys()})

        return reformat_df, clarity_df

    def get_item_relation_scores(self):
        return self.clarity_df.div(self.clarity_df.sum(axis=1),axis=0)

    def get_item_clarity(self, item_ind):
        item_rel_df = self.get_item_relation_scores()
        return item_rel_df.max(axis=1).loc[item_ind]

    def relation_clarity_scores(self):
        item_rel_df = self.get_item_relation_scores()
        rel_clarity_df = pd.DataFrame({'rel_clarity': item_rel_df.max(axis=0), 'num_annotations':self.clarity_df.sum(axis=0)})
        return rel_clarity_df.sort_values(by='num_annotations', ascending=False)

    def get_all_worker_ids(self):
        return self.df['WorkerId'].unique()

    def get_worker_df(self, worker_id, include_worktime=False):
        w_df = pd.DataFrame(np.zeros(self.clarity_df.shape), index=self.clarity_df.index, columns=self.clarity_df.columns)

        df = self.df.groupby('WorkerId').get_group(worker_id).drop(columns='WorkerId').set_index('HITId')

        for ind, row in df.iterrows():
            w_df.loc[ind] = row

        if ('WorkTimeInSeconds' in w_df.columns) and not include_worktime:
            return w_df.drop(columns=['WorkTimeInSeconds'])
        else:
            return w_df

    def avg_annotations_per_item(self,df):
        return df[df.sum(axis=1) > 1].sum(axis=1).mean()

    def asym_worker_agreement(self, w_df1, w_df2, minimum_common=1):
        relations_in_common = 0
        num_annotations_w1 = 0

        common_indeces = list(set(w_df1[(w_df1.sum(axis=1) > 0)].index).intersection(set(w_df2[(w_df2.sum(axis=1) > 0)].index)))

        if len(common_indeces) < minimum_common:
            return np.nan

        for ind in common_indeces:
            labels_1 = list(w_df1.loc[ind][w_df1.loc[ind] == 1.0].index)
            labels_2 = list(w_df2.loc[ind][w_df2.loc[ind] == 1.0].index)

            relations_in_common += len(set(labels_1).intersection(set(labels_2)))
            num_annotations_w1 += len(labels_1)

        return relations_in_common / num_annotations_w1

    def avg_asym_worker_agreement(self, w_id1, full_series=False):
        w_df1 = self.get_worker_df(w_id1)
        agreements = []

        for w_id2 in self.df['WorkerId'].unique():
            agr = self.asym_worker_agreement(w_df1, self.get_worker_df(w_id2))
            agreements.append(agr)

        agreements = pd.Series(agreements)
        if not full_series:
            return agreements.mean()
        return pd.Series(agreements)

    def worker_item_sim(self, w_df, item_ind):
        w_vec = w_df.loc[item_ind]
        all_vec = self.clarity_df.loc[item_ind]
        all_minus_w_vec = all_vec - w_vec

        return cosine_sim(w_vec, all_minus_w_vec)

    def worker_item_disagreement(self, w_df, item_ind):
        item_clarity = self.get_item_clarity(item_ind)
        return item_clarity - self.worker_item_sim(w_df, item_ind)

    def avg_worker_item_disagreement(self, w_df):
        w_inds = w_df[w_df.sum(axis=1) > 0].index

        disagr_scores = pd.Series([self.worker_item_disagreement(w_df, ind) for ind in w_inds])

        return disagr_scores.mean()

    def items_in_common(self, w_df1, w_df2):
        return len(set(w_df1[w_df1.sum(axis=1) > 0].index).intersection(set(w_df2[w_df2.sum(axis=1) > 0].index)))

def cosine_sim(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)
