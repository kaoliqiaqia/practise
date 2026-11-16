# 导入库
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

# 页面设置
st.set_page_config(page_title='临床预测模型', page_icon='🩺')
st.title('🩺 临床机器学习预测演示')

# 加载模型
BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR / 'model_results' / 'LR_model.pkl'

if not model_path.is_file():
    st.error(f'未找到模型：{model_path}')
    st.stop()
model = joblib.load(model_path)

# 从 Pipeline 反推变量清单
encoder = model.named_steps['encoder']
features = list(encoder.feature_names_in_)  # 训练时的特征顺序
numeric = list(encoder.transformers_[0][2])       # 训练时 num 分支
categorical = list(encoder.transformers_[1][2])   # 训练时 cat 分支
categories = encoder.named_transformers_['cat'].categories_
category_options = dict(zip(categorical, categories))

# 显示输入表单
st.subheader('输入临床指标')  # 二级标题
st.write(f'当前模型共需要 {len(features)} 项变量。')
values = {}
with st.form('prediction'):
    for name in features:
        if name in numeric:
            values[name] = st.number_input(name, value=0.0, format='%.2f')
        else:
            values[name] = st.selectbox(name, list(category_options[name]))
    submitted = st.form_submit_button('开始预测', use_container_width=True)

#  调用 Pipeline，显示预测结果
if submitted:
    patient = pd.DataFrame([values], columns=features)
    positive_index = list(model.classes_).index(1)
    probability = float(model.predict_proba(patient)[0, positive_index])
    result = '阳性' if probability >= 0.5 else '阴性'
    st.subheader('预测结果')
    st.metric('阳性预测概率', f'{probability:.2%}')
    st.write(f'预测类别：**{result}**')

