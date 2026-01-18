import pandas as pd

df = pd.read_csv('guardian_news_std.csv')
keywords = ['ukraine', 'russia', 'war']
mask = df['title'].str.lower().str.contains('|'.join(keywords)) | df['content'].str.lower().str.contains('|'.join(keywords))
df_kw = df[mask]
# 先采样所有相关事件新闻
df_sample_kw = df_kw.sample(n=min(100, len(df_kw)), random_state=42)
# 再补充其他新闻
df_sample_other = df[~mask].sample(n=200, random_state=42)
df_sample = pd.concat([df_sample_kw, df_sample_other]).reset_index(drop=True)
df_sample.to_csv('guardian_news_std_sample.csv', index=False)