import pandas as pd
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('D:\\工人绩效预测\\Task1_W_Zone1.csv')
categorical_cols = ['jobTitle', 'gender', 'edu', 'dept']
encoders = {}
df = pd.DataFrame(df)
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le  # 保存编码器
# 定义分箱规则
bins = [1000, 4000, 7000, 10000, 13000]
labels = [0, 1, 2, 3]
# 生成分类列
df['bonus_category'] = pd.cut(df['bonus'],
                              bins=bins,
                              labels=labels,
                              right=False)  # 包含左边界，不包含右边界
# 覆盖并删除列
df['bonus'] = df['bonus_category']  # 赋值到原列
df.drop(columns='bonus_category', inplace=True)  # 删除分类列
# 验证分布
print(df.head(15))