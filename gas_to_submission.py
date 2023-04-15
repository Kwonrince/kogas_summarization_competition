epoch03_law_df = pd.read_csv('data/epoch03_law_df.csv')

submission = pd.read_csv('data/1st_submission_p.csv')

for id_, summary_ in zip(epoch03_law_df['id'], epoch03_law_df['summary']):
    idx = submission[submission['id']==int(id_)].index
    summary_ = ' '.join(re.sub('\n','',summary_).split())
    submission.loc[idx, 'summary'] = summary_

submission.to_csv('./data/1st_submission_p_epoch03law.csv', index=False)