#!/usr/bin/env python
# coding: utf-8

# # Shifting Agendas: The Guardian and the Death of Queen Elizabeth II
# 
# **Event-Driven Topic Shifts: A BERTopic Analysis of The Guardian Before and After the Death of Queen Elizabeth II**
# 
# This notebook performs topic modeling, alignment, sentiment and entity analyses to quantify how The Guardian's news agenda shifted around the death of Queen Elizabeth II (2020–2025 coverage).

# ## 1. 导入所需库
# 
# 本节导入数据分析、文本处理、可视化和主题建模所需的主要Python库。

# In[2]:


import pandas as pd
import numpy as np
import re
import os
from datetime import datetime
import warnings
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
warnings.filterwarnings("ignore")

# Titles used across notebook (updateable)
paper_title = 'Event-Driven Topic Shifts: A BERTopic Analysis of The Guardian Before and After the Death of Queen Elizabeth II'
short_title = 'Shifting Agendas: The Guardian and the Death of Queen Elizabeth II'


# In[ ]:





# ## 2. 加载与预览卫报数据
# 
# 读取guardian_news.csv文件，展示前几行，查看数据结构和主要字段。

# In[3]:


# 读取卫报新闻数据
# 假设csv包含columns: 'date', 'title', 'content'，如有不同请根据实际情况调整

df = pd.read_csv('guardian_news_std.csv')
# 统一日期为UTC，若原始数据为本地时间则会被转换为UTC；parsing errors -> NaT
df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce')
# 丢弃无法解析的行并按时间排序
df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)

df.head()


# In[4]:


start_date = pd.to_datetime('2020-01-01', utc=True)
end_date = pd.to_datetime('2025-12-31', utc=True)
df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()

len(df)


# ## 3. 数据清洗与预处理
# 
# 对新闻文本进行清洗，包括去除标点、停用词、小写化、分词等。

# ## 3.1 LDA主题建模与BERTopic对比
# 
# 本节用LDA方法对pre_event_noevent和post_event_noevent进行主题建模，并与BERTopic结果进行关键词对比。

# In[ ]:





# ### LDA与BERTopic对比结论
# - LDA和BERTopic都能发现事件前后主要议题的变化，但BERTopic对短文本和新词更敏感，主题更细致，关键词更丰富。
# - 两者主题分布大体一致，如经济、政治、人物等，但具体关键词和聚类方式略有不同。
# - 综合来看，BERTopic更适合新闻类文本的细粒度主题分析，LDA适合宏观议题结构对比。

# In[5]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text

# 正确设置 stop_words 参数：用 list 或 'english'，不要用 frozenset
custom_stopwords = list(text.ENGLISH_STOP_WORDS) + [
    "said", "say", "says", "will", "also", "one"
 ]

vectorizer_model = CountVectorizer(
    stop_words=custom_stopwords,
    ngram_range=(1, 2),
    min_df=1,          # 降低min_df，避免样本过小时出错
    max_df=1.0         # 提高max_df，保证不出错
)

def light_clean(text):
    text = str(text).lower()
    text = re.sub(r"http\\S+", "", text)
    text = re.sub(r"[^a-z\\s]", " ", text)
    text = re.sub(r"\\s+", " ", text)
    return text.strip()

df["clean_text"] = df["title"].apply(light_clean)
model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(
    df["clean_text"].tolist(),
    batch_size=64,
    show_progress_bar=True
)

# 统一时间区间，分割点为事件日（默认为女王逝世）
# 确保比较使用 UTC-aware timestamps
start_date = pd.to_datetime('2020-01-01', utc=True)
end_date = pd.to_datetime('2025-12-31', utc=True)
event_date = pd.to_datetime('2022-09-01', utc=True)  # Queen Elizabeth II death date

mask = (df['date'] >= start_date) & (df['date'] <= end_date)
df_period = df[mask].copy()

# 使用更语义化的变量名 pre_event / post_event 并保留 pre_war / post_war 作为别名
pre_event  = df_period[df_period['date'] < event_date].copy()  # pre-event period
post_event = df_period[df_period['date'] >= event_date].copy()  # post-event period

# 在事件前后都跳过短期“新闻高峰”缓冲期（例如 30 天），以避免即时报道主导分析
pre_event_buffered_30d = pre_event[pre_event['date'] <= (event_date - pd.Timedelta(days=30))].copy()
post_event_buffered_30d = post_event[post_event['date'] >= (event_date + pd.Timedelta(days=30))].copy()

# 另外保留按关键词排除事件报道的 no-event 子集（包含 obituary/death/tribute 等关键词）
target_keywords = ['lord','queen','elizabeth','queen elizabeth','monarchy','royal','king charles','obituary','death','died','tribute','funeral']
pattern = '|'.join([re.escape(k) for k in target_keywords])
pre_event_noevent = pre_event[~pre_event['clean_text'].str.contains(pattern, case=False, na=False)].copy()
post_event_noevent = post_event[~post_event['clean_text'].str.contains(pattern, case=False, na=False)].copy()
# 快速输出各数据集规模供选择参考
print('Counts: pre_event=', len(pre_event), 'pre_event_buffered_30d=', len(pre_event_buffered_30d), 'pre_event_noevent=', len(pre_event_noevent), 'post_event=', len(post_event), 'post_event_buffered_30d=', len(post_event_buffered_30d), 'post_event_noevent=', len(post_event_noevent))


# In[6]:


# 采样后重置索引，并重新生成embedding，避免索引错位
# 方法A：直接用 noevent 子集建模，彻底排除 obituary/death/tribute 等报道
pre_sample = pre_event_noevent.sample(20000, random_state=42) if len(pre_event_noevent) > 20000 else pre_event_noevent
pre_sample = pre_sample.reset_index(drop=True)
pre_embeddings = model.encode(pre_sample["clean_text"].tolist(), batch_size=64, show_progress_bar=True)

post_sample = post_event_noevent.sample(20000, random_state=42) if len(post_event_noevent) > 20000 else post_event_noevent
post_sample = post_sample.reset_index(drop=True)
post_embeddings = model.encode(post_sample["clean_text"].tolist(), batch_size=64, show_progress_bar=True)

# 分别对前后时间段建模（noevent）
pre_topic_model = BERTopic(
    embedding_model=None,
    vectorizer_model=vectorizer_model,
    language="english",
    calculate_probabilities=False,
    verbose=True
)
post_topic_model = BERTopic(
    embedding_model=None,
    vectorizer_model=vectorizer_model,
    language="english",
    calculate_probabilities=False,
    verbose=True
)

# 分别fit_transform（noevent）
pre_topics, _ = pre_topic_model.fit_transform(
    pre_sample["clean_text"].tolist(),
    pre_embeddings
)
post_topics, _ = post_topic_model.fit_transform(
    post_sample["clean_text"].tolist(),
    post_embeddings
)

pre_topic_info = pre_topic_model.get_topic_info()
post_topic_info = post_topic_model.get_topic_info()

pre_topic_info.head(10), post_topic_info.head(10)


# In[ ]:


pre_topic_model.visualize_barchart(top_n_topics=5)


# In[ ]:


# 显示 post (noevent) 模型的前 5 个主题词云 (替换原来的柱状图)


post_topic_model.visualize_barchart(top_n_topics=5)


# In[9]:


# --- End of Section 8: All dynamic topic modeling, comparison, sentiment, event, content mining, and interactive visualization code restored.


# In[10]:


# (Optional) Clean up: Remove temporary variables if needed
#del df_war, topic_quarter, sentiment_daily, topic_daily, df_event


# In[11]:


# 1) 话题对齐（pre ↔ post）
# 计算话题嵌入相似度并打印 pre -> top-3 post 映射
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def get_topic_embeddings_safe(topic_model):
    emb = getattr(topic_model, 'topic_embeddings_', None)
    if emb is None:
        emb = getattr(topic_model, 'topic_embeddings', None)
    return emb

pre_emb = get_topic_embeddings_safe(pre_topic_model)
post_emb = get_topic_embeddings_safe(post_topic_model)

if pre_emb is None or post_emb is None:
    print('Warning: one of the topic models has no topic embeddings; alignment skipped.')
    mapping = {}
else:
    sim = cosine_similarity(pre_emb, post_emb)
    mapping = {i: list(np.argsort(-sim[i])[:3]) for i in range(sim.shape[0])}
    print('Topic alignment (pre -> top-3 post):')
    for k, v in mapping.items():
        print(f'pre {k} -> post {v} (sims: {[float(sim[k,i]) for i in v]})')

# mapping will be used by later cells


# In[12]:


# 2) 展示话题详情（top words + 代表文档）
# 使用 show_topic_details(topic_model, sample_df, topics_labels, topic_id)

def show_topic_details(topic_model, sample_df, topics_labels, topic_id, n_words=10, n_docs=5):
    print('\n=== Topic', topic_id, 'details ===')
    try:
        words = topic_model.get_topic(topic_id)
        print('Top words:', [w for w,_ in words[:n_words]])
    except Exception:
        print('No words for topic', topic_id)
    mask = (topics_labels == topic_id)
    print('Num docs:', int(mask.sum()))
    if mask.sum() > 0:
        for _, row in sample_df[mask].head(n_docs).iterrows():
            print(row.get('date'), '-', row.get('title'))

# 示例：展示前5个 pre 话题及其最相近的两个 post 话题
for tid in range(min(5, len(mapping) if 'mapping' in globals() else 0)):
    show_topic_details(pre_topic_model, pre_sample, np.array(pre_topics), tid)
    if tid in mapping:
        for post_tid in mapping[tid][:2]:
            show_topic_details(post_topic_model, post_sample, np.array(post_topics), post_tid)


# In[13]:


# 3) 定量比较：话题频率、卡方检验与增减排名


pre_counts = pd.Series(pre_topics).value_counts().sort_index()
post_counts = pd.Series(post_topics).value_counts().sort_index()
all_topics = sorted(set(pre_counts.index.tolist() + post_counts.index.tolist()))
pre_vec = np.array([pre_counts.get(t, 0) for t in all_topics])
post_vec = np.array([post_counts.get(t, 0) for t in all_topics])

contingency = np.vstack([pre_vec, post_vec]).T
chi2, p, dof, ex = chi2_contingency(contingency)
print('Chi-square across topics: chi2=%.3f, p=%.3e, dof=%d' % (chi2, p, dof))

pre_prop = pre_vec / pre_vec.sum() if pre_vec.sum()>0 else np.zeros_like(pre_vec)
post_prop = post_vec / post_vec.sum() if post_vec.sum()>0 else np.zeros_like(post_vec)
diff = post_prop - pre_prop
rank_increase = np.argsort(-diff)[:10]
rank_decrease = np.argsort(diff)[:10]

print('\nTop increased topics (post - pre):')
for idx in rank_increase:
    t = all_topics[idx]
    print(t, 'diff=%.4f pre=%.4f post=%.4f' % (diff[idx], pre_prop[idx], post_prop[idx]))

print('\nTop decreased topics:')
for idx in rank_decrease:
    t = all_topics[idx]
    print(t, 'diff=%.4f pre=%.4f post=%.4f' % (diff[idx], pre_prop[idx], post_prop[idx]))

# 保存 topic_summary 供后续使用
topic_summary = pd.DataFrame({'topic': all_topics, 'pre_count': pre_vec, 'post_count': post_vec, 'diff': diff})
topic_summary.to_csv('topic_summary.csv', index=False)
print('Saved topic_summary.csv')


# In[14]:


# 4) 时间序列：按周展示 top shifting topics 的流行度变化
import matplotlib.pyplot as plt

# 准备数据框（确保 date 为 datetime）
pre_df = pre_sample.copy()
pre_df['topic'] = pre_topics
pre_df['date'] = pd.to_datetime(pre_df['date'])
post_df = post_sample.copy()
post_df['topic'] = post_topics
post_df['date'] = pd.to_datetime(post_df['date'])

# 选择前6个绝对变化最大的topic
top_topics = [all_topics[i] for i in np.argsort(-np.abs(diff))[:6]]

fig, axs = plt.subplots(len(top_topics), 1, figsize=(12, 3*len(top_topics)), sharex=True)
for i, t in enumerate(top_topics):
    s_pre = pre_df[pre_df['topic']==t].set_index('date').resample('7D').size().rename('pre')
    s_post = post_df[post_df['topic']==t].set_index('date').resample('7D').size().rename('post')
    combined = pd.concat([s_pre, s_post], axis=1).fillna(0)
    ax = axs[i] if len(top_topics)>1 else axs
    combined.plot(ax=ax, title=f'Topic {t} weekly prevalence')
plt.tight_layout()
plt.show()


# In[15]:


# 5) 变化点检测（可选，依赖 ruptures）
try:
    import ruptures as rpt
    for t in top_topics:
        s = pd.concat([
            pre_df[pre_df['topic']==t].set_index('date').resample('7D').size(),
            post_df[post_df['topic']==t].set_index('date').resample('7D').size()
        ]).fillna(0).values
        if len(s) > 12:
            model = rpt.Pelt(model='rbf').fit(s)
            cps = model.predict(pen=10)
            print('Topic', t, 'change points (weekly idx):', cps)
except Exception as e:
    print('ruptures not installed or failed:', e)


# In[16]:


# 6) 情感分析（VADER）并对 top topics 做 t-test
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    pre_df['sentiment'] = pre_df['clean_text'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
    post_df['sentiment'] = post_df['clean_text'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
    pre_sent = pre_df.groupby('topic')['sentiment'].agg(['mean','count']).rename(columns={'mean':'pre_mean'})
    post_sent = post_df.groupby('topic')['sentiment'].agg(['mean','count']).rename(columns={'mean':'post_mean'})
    sent = pre_sent.join(post_sent, how='outer').fillna(0)
    print('\nSentiment by topic (sample):')
    print(sent.head())
    from scipy.stats import ttest_ind
    for t in top_topics:
        a = pre_df[pre_df['topic']==t]['sentiment'].dropna()
        b = post_df[post_df['topic']==t]['sentiment'].dropna()
        if len(a) > 5 and len(b) > 5:
            stat, pval = ttest_ind(a, b, equal_var=False)
            print('Topic', t, 'sentiment t-test p=%.3e (n_pre=%d n_post=%d)' % (pval, len(a), len(b)))
except Exception as e:
    print('VADER not available or sentiment step failed:', e)


# In[17]:


# 7) 实体抽取与共现（spaCy 简易版本）
try:
    import spacy
    nlp = spacy.load('en_core_web_sm')
    def extract_entities(text):
        doc = nlp(text)
        return [ent.text for ent in doc.ents if ent.label_ in ('PERSON','ORG','GPE')]
    pre_df['entities'] = pre_df['title'].astype(str).apply(extract_entities)
    post_df['entities'] = post_df['title'].astype(str).apply(extract_entities)
    from collections import Counter
    def top_entities(df, topn=20):
        c = Counter()
        df['entities'].apply(lambda ents: c.update(set(ents)))
        return c.most_common(topn)
    print('\nTop entities pre:', top_entities(pre_df))
    print('\nTop entities post:', top_entities(post_df))
except Exception as e:
    print('spaCy/entity extraction failed:', e)


# In[18]:


# 8) 稳健性检验（小样本上比较不同 vectorizer 参数的 ARI）
try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics import adjusted_rand_score
    def fit_small_bertopic(texts, embeddings, min_df=1, stop_words=None):
        vec = CountVectorizer(stop_words=stop_words, ngram_range=(1,2), min_df=min_df, max_df=1.0)
        m = BERTopic(embedding_model=None, vectorizer_model=vec, language='english', calculate_probabilities=False, verbose=False)
        topics, _ = m.fit_transform(texts, embeddings)
        return m, topics

    sample_n = min(2000, len(pre_df))
    small_texts = pre_df['clean_text'].tolist()[:sample_n]
    small_emb = pre_embeddings[:sample_n] if 'pre_embeddings' in globals() else None
    if small_emb is None:
        print('No pre_embeddings available for robustness check; skipping')
    else:
        m1, t1 = fit_small_bertopic(small_texts, small_emb, min_df=1, stop_words=custom_stopwords)
        m2, t2 = fit_small_bertopic(small_texts, small_emb, min_df=2, stop_words='english')
        ari = adjusted_rand_score(t1, t2)
        print('Robustness ARI between min_df=1 and min_df=2 (small sample):', ari)
except Exception as e:
    print('Robustness check failed or too slow:', e)


# In[19]:


# 9) 保存模型与结果文件（models, csv, 简短报告）
import joblib
try:
    joblib.dump(pre_topic_model, 'pre_topic_model.joblib')
    joblib.dump(post_topic_model, 'post_topic_model.joblib')
    pre_df.to_csv('pre_topic_assignments.csv', index=False)
    post_df.to_csv('post_topic_assignments.csv', index=False)
    topic_summary.to_csv('topic_summary.csv', index=False)
    print('Saved models and CSV outputs to working directory.')
except Exception as e:
    print('Saving outputs failed:', e)

# 10) 写简短分析摘要到文件
try:
    report = []
    report.append('# Brief A-grade Results Summary\n')
    report.append('## Key findings\n')
    report.append('- Chi-square across topics: chi2=%.3f, p=%.3e\n' % (chi2, p))
    report.append('## Top increased topics (post vs pre)\n')
    for idx in rank_increase[:10]:
        t = all_topics[idx]
        report.append(f'- Topic {t}: diff={diff[idx]:.4f}, pre={pre_prop[idx]:.4f}, post={post_prop[idx]:.4f}\n')
    with open('analysis_brief.md', 'w') as f:
        f.writelines(report)
    print('Wrote analysis_brief.md')
except Exception as e:
    print('Writing brief failed:', e)


# In[20]:


from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

def lda_topic_words(texts, n_topics=10, max_features=2000):
    vec = CountVectorizer(stop_words='english', max_features=max_features)
    X = vec.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda_topics = lda.fit_transform(X)
    topic_words = []
    for idx, topic in enumerate(lda.components_):
        words = [vec.get_feature_names_out()[i] for i in topic.argsort()[-10:][::-1]]
        topic_words.append(words)
    return topic_words

print('LDA主题建模（pre_event_noevent）:')
lda_pre_words = lda_topic_words(pre_sample['clean_text'].tolist(), n_topics=10)
for i, words in enumerate(lda_pre_words):
    print(f'Pre-LDA Topic {i}:', words)

print('LDA主题建模（post_event_noevent）:')
lda_post_words = lda_topic_words(post_sample['clean_text'].tolist(), n_topics=10)
for i, words in enumerate(lda_post_words):
    print(f'Post-LDA Topic {i}:', words)

print('BERTopic主题关键词（pre）:')
for i in range(10):
    print(f'Pre-BERTopic {i}:', [w for w,_ in pre_topic_model.get_topic(i)[:10]])

print('BERTopic主题关键词（post）:')
for i in range(10):
    print(f'Post-BERTopic {i}:', [w for w,_ in post_topic_model.get_topic(i)[:10]])


# In[25]:


# 生成并保存 BERTopic 与 LDA 主题词云（pre_event_noevent 和 post_event_noevent）
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

os.makedirs('wordclouds', exist_ok=True)

def plot_topic_wordcloud(topic_model, topic_id, save_path=None, algo_name='BERTopic', period='pre'):
    words_weights = topic_model.get_topic(topic_id)
    if not words_weights:
        print(f"No words for topic {topic_id}")
        return
    word_freq = {w: float(wt) for w, wt in words_weights}
    wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'{algo_name} {period} topic {topic_id} WordCloud')
    plt.tight_layout()
    if save_path is None:
        save_path = f'wordclouds/{algo_name}_{period}_topic_{topic_id}.png'
    plt.savefig(save_path)
    plt.close()
    print(f"Saved wordcloud for {algo_name} {period} topic {topic_id} to {save_path}")

# 保存 BERTopic 前 5 个主题的词云，文件命名为 ALGO_pre_topic_{i}.png / ALGO_post_topic_{i}.png
for tid in range(5):
    plot_topic_wordcloud(pre_topic_model, tid, save_path=f'wordclouds/BERTopic_pre_topic_{tid}.png', algo_name='BERTopic', period='pre')
    plot_topic_wordcloud(post_topic_model, tid, save_path=f'wordclouds/BERTopic_post_topic_{tid}.png', algo_name='BERTopic', period='post')

# --- 现在对 pre/post 子集分别训练 LDA（轻量版）并保存前 5 个主题的词云 ---
try:
    # Pre LDA
    vec_pre = CountVectorizer(stop_words='english', max_features=2000)
    texts_pre = pre_sample['clean_text'].fillna('').tolist()
    if len(texts_pre) > 0:
        Xpre = vec_pre.fit_transform(texts_pre)
        n_topics = min(10, Xpre.shape[1])
        lda_pre = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda_pre.fit(Xpre)
        feat_pre = vec_pre.get_feature_names_out()
        for tid in range(min(5, n_topics)):
            comp = lda_pre.components_[tid]
            top_idx = comp.argsort()[-50:][::-1]
            word_freq = {feat_pre[i]: float(comp[i]) for i in top_idx}
            wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
            out = f'wordclouds/LDA_pre_topic_{tid}.png'
            plt.figure(figsize=(10,5))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(out)
            plt.close()
            print('Saved', out)
    else:
        print('No pre texts for LDA')

    # Post LDA
    vec_post = CountVectorizer(stop_words='english', max_features=2000)
    texts_post = post_sample['clean_text'].fillna('').tolist()
    if len(texts_post) > 0:
        Xpost = vec_post.fit_transform(texts_post)
        n_topics_post = min(10, Xpost.shape[1])
        lda_post = LatentDirichletAllocation(n_components=n_topics_post, random_state=42)
        lda_post.fit(Xpost)
        feat_post = vec_post.get_feature_names_out()
        for tid in range(min(5, n_topics_post)):
            comp = lda_post.components_[tid]
            top_idx = comp.argsort()[-50:][::-1]
            word_freq = {feat_post[i]: float(comp[i]) for i in top_idx}
            wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
            out = f'wordclouds/LDA_post_topic_{tid}.png'
            plt.figure(figsize=(10,5))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(out)
            plt.close()
            print('Saved', out)
    else:
        print('No post texts for LDA')
except Exception as e:
    print('LDA wordcloud generation failed:', e)


# In[22]:


# Single-thread coherence computation (uses notebook's pre/post samples and BERTopic models)
import json
import numpy as np
from collections import defaultdict
from gensim.corpora import Dictionary
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# C_v estimator (window-based cooccurrence, single-thread)
def compute_cv_coherence(topics, texts, window_size=110):
    co_occur = defaultdict(int)
    total_windows = 0
    for doc in texts:
        doc_len = len(doc)
        if doc_len < window_size:
            windows = [doc]
        else:
            windows = [doc[i:i+window_size] for i in range(doc_len - window_size + 1)]
        total_windows += len(windows)
        for win in windows:
            words = list(set(win))
            for i in range(len(words)):
                for j in range(i+1, len(words)):
                    w1, w2 = words[i], words[j]
                    co_occur[(w1, w2)] += 1
                    co_occur[(w2, w1)] += 1
    topic_scores = []
    for topic in topics:
        word_pairs = [(topic[i], topic[j]) for i in range(len(topic)) for j in range(i+1, len(topic))]
        pair_scores = []
        for w1, w2 in word_pairs:
            p = co_occur.get((w1, w2), 0) / max(1, total_windows)
            if p == 0:
                pair_scores.append(0)
            else:
                pair_scores.append(p * (1 - p))
        topic_scores.append(sum(pair_scores) / len(pair_scores) if pair_scores else 0)
    return sum(topic_scores) / len(topic_scores) if topic_scores else 0

# UMass estimator (single-thread)
def compute_umass_coherence(topics, corpus, dictionary):
    word_counts = defaultdict(int)
    co_counts = defaultdict(int)
    for doc in corpus:
        doc_words = [dictionary[id] for id, _ in doc]
        for w in doc_words:
            word_counts[w] += 1
        for i in range(len(doc_words)):
            for j in range(i+1, len(doc_words)):
                w1, w2 = doc_words[i], doc_words[j]
                co_counts[(w1, w2)] += 1
                co_counts[(w2, w1)] += 1
    topic_scores = []
    for topic in topics:
        word_pairs = [(topic[i], topic[j]) for i in range(len(topic)) for j in range(i+1, len(topic))]
        pair_scores = []
        for w1, w2 in word_pairs:
            c_w1 = word_counts.get(w1, 0)
            c_w1w2 = co_counts.get((w1, w2), 0)
            if c_w1w2 == 0 or c_w1 == 0:
                pair_scores.append(-10)
            else:
                pair_scores.append(np.log((c_w1w2 + 1) / c_w1))
        topic_scores.append(sum(pair_scores) / len(pair_scores) if pair_scores else -10)
    return sum(topic_scores) / len(topic_scores) if topic_scores else -10

# Helper to extract BERTopic topic words
def bertopic_topics_words_from_model(model, top_n=15):
    info = model.get_topic_info()
    topics = []
    for t in info['Topic'].tolist():
        if t == -1:
            continue
        words = [w for w,_ in model.get_topic(t)[:top_n]]
        topics.append(words)
    return topics

# Get documents from common notebook variable names
def get_docs(candidates):
    for name in candidates:
        if name in globals():
            df = globals()[name]
            try:
                return df['clean_text'].fillna('').tolist()
            except Exception:
                try:
                    return [str(x) for x in df]
                except Exception:
                    continue
    return []

pre_docs = get_docs(['pre_sample','sample_pre','pre_event_noevent','pre_event','pre_sample'])
post_docs = get_docs(['post_sample','sample_post','post_event_noevent','post_event','post_sample'])

if not pre_docs or not post_docs:
    raise RuntimeError('pre/post documents not found in notebook variables (expected `pre_sample`/`post_sample` or similar).')

pre_sample_texts = [d.split() for d in pre_docs]
post_sample_texts = [d.split() for d in post_docs]

# Extract topics from BERTopic models if available
pre_topics_words = bertopic_topics_words_from_model(pre_topic_model, top_n=15) if 'pre_topic_model' in globals() else []
post_topics_words = bertopic_topics_words_from_model(post_topic_model, top_n=15) if 'post_topic_model' in globals() else []

cv_pre = compute_cv_coherence(pre_topics_words, pre_sample_texts)
cv_post = compute_cv_coherence(post_topics_words, post_sample_texts)

# LDA and UMass
vec_pre = CountVectorizer(stop_words='english', max_features=2000)
Xpre = vec_pre.fit_transform(pre_docs)
vec_post = CountVectorizer(stop_words='english', max_features=2000)
Xpost = vec_post.fit_transform(post_docs)

n_topics = 10
lda_pre = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda_post = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda_pre.fit(Xpre)
lda_post.fit(Xpost)

feature_names_pre = vec_pre.get_feature_names_out()
feature_names_post = vec_post.get_feature_names_out()

def lda_topic_words_local(lda, feature_names, n_top=15):
    topics=[]
    for topic_idx, topic in enumerate(lda.components_):
        top = [feature_names[i] for i in topic.argsort()[:-n_top-1:-1]]
        topics.append(top)
    return topics

lda_pre_words = lda_topic_words_local(lda_pre, feature_names_pre, n_top=15)
lda_post_words = lda_topic_words_local(lda_post, feature_names_post, n_top=15)

pre_texts_for_dict = [d.split() for d in pre_docs]
post_texts_for_dict = [d.split() for d in post_docs]
pre_dict = Dictionary(pre_texts_for_dict)
post_dict = Dictionary(post_texts_for_dict)
pre_corpus = [pre_dict.doc2bow(text) for text in pre_texts_for_dict]
post_corpus = [post_dict.doc2bow(text) for text in post_texts_for_dict]

umass_pre = compute_umass_coherence(lda_pre_words, pre_corpus, pre_dict)
umass_post = compute_umass_coherence(lda_post_words, post_corpus, post_dict)

results = {'cv_pre': float(cv_pre), 'cv_post': float(cv_post), 'umass_pre': float(umass_pre), 'umass_post': float(umass_post)}
with open('coherence_results.json','w') as f:
    json.dump(results, f)
print('Saved coherence_results.json:', results)


# In[ ]:


# 11) Add LDA Dynamic Topic Time Series Plot (Matched Style)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Ensure we have LDA models and vectorizers
if 'lda_pre' not in globals() or 'vec_pre' not in globals():
    print("Retraining LDA Pre model for plotting...")
    vec_pre = CountVectorizer(stop_words='english', max_features=2000)
    texts_pre = pre_sample['clean_text'].fillna('').tolist()
    Xpre = vec_pre.fit_transform(texts_pre)
    lda_pre = LatentDirichletAllocation(n_components=10, random_state=42)
    lda_pre.fit(Xpre)

if 'lda_post' not in globals() or 'vec_post' not in globals():
    print("Retraining LDA Post model for plotting...")
    vec_post = CountVectorizer(stop_words='english', max_features=2000)
    texts_post = post_sample['clean_text'].fillna('').tolist()
    Xpost = vec_post.fit_transform(texts_post)
    lda_post = LatentDirichletAllocation(n_components=10, random_state=42)
    lda_post.fit(Xpost)

# Get dominant topics
def get_lda_dominant_topics(model, vectorizer, texts):
    X = vectorizer.transform(texts)
    doc_topic_dist = model.transform(X)
    return doc_topic_dist.argmax(axis=1)

pre_lda_topics = get_lda_dominant_topics(lda_pre, vec_pre, pre_sample['clean_text'].fillna('').tolist())
post_lda_topics = get_lda_dominant_topics(lda_post, vec_post, post_sample['clean_text'].fillna('').tolist())

# Prepare dataframes
pre_df_lda = pre_sample.copy()
pre_df_lda['topic'] = pre_lda_topics
pre_df_lda['date'] = pd.to_datetime(pre_df_lda['date'])

post_df_lda = post_sample.copy()
post_df_lda['topic'] = post_lda_topics
post_df_lda['date'] = pd.to_datetime(post_df_lda['date'])

# Select top 6 topics (matching Figure 1 count)
top_pre_topics = pre_df_lda['topic'].value_counts().head(6).index.tolist()
top_post_topics = post_df_lda['topic'].value_counts().head(6).index.tolist()

# Plot Pre-event LDA Topics
# Using same style as Figure 1: separate subplots, blue for pre, orange for post
fig, axs = plt.subplots(len(top_pre_topics), 1, figsize=(12, 3*len(top_pre_topics)), sharex=True)
if len(top_pre_topics) == 1: axs = [axs]

for i, t in enumerate(top_pre_topics):
    s_pre = pre_df_lda[pre_df_lda['topic']==t].set_index('date').resample('7D').size().rename('pre')
    # Create empty post series for alignment (though it will be 0/empty in this period)
    # Actually for "Pre-event" plot, we only show Pre data.
    # To match Figure 1 style (which plots 'combined'), we'll just plot the 'pre' line in blue.

    ax = axs[i]
    s_pre.plot(ax=ax, label='pre', color='#1f77b4') # standard matplotlib blue
    ax.set_title(f'Pre-LDA Topic {t} weekly prevalence')
    ax.legend()

plt.tight_layout()
plt.savefig('lda_pre_time_series.png')
plt.close()
print("Saved lda_pre_time_series.png")

# Plot Post-event LDA Topics
fig, axs = plt.subplots(len(top_post_topics), 1, figsize=(12, 3*len(top_post_topics)), sharex=True)
if len(top_post_topics) == 1: axs = [axs]

for i, t in enumerate(top_post_topics):
    s_post = post_df_lda[post_df_lda['topic']==t].set_index('date').resample('7D').size().rename('post')

    ax = axs[i]
    s_post.plot(ax=ax, label='post', color='#ff7f0e') # standard matplotlib orange
    ax.set_title(f'Post-LDA Topic {t} weekly prevalence')
    ax.legend()

plt.tight_layout()
plt.savefig('lda_post_time_series.png')
plt.close()
print("Saved lda_post_time_series.png")

