import os
import pandas as pd
import streamlit as st
from ydata_profiling import ProfileReport


# @st.cache
def data_preview(file_path: str):
    st.write('**Data Name:**', file_path.split('\\')[-1])
    st.write('**Data Path:**', file_path)
    df_data = pd.read_csv(file_path)
    st.checkbox("Use container width", value=True, key="use_container_width")
    
    # Same as st.write(df)
    # return st.dataframe(df_data)
    return st.dataframe(df_data, use_container_width=st.session_state.use_container_width)


def data_metadata(file_path: str):
    df_data = pd.read_csv(file_path)
    st.write('**Data Name:**', file_path.split('\\')[-1])
    st.write('**Data Path:**', file_path)
    csv = df_data.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name="BodyPerformans.csv",
        mime="text/csv",
        )        
    st.write('**Data Shape:**', df_data.shape)
    # st.write('**Data Columns Count:{} & List:{}**'.format(len(df_data.columns), list(df_data.columns)))
    st.write(f'**Data Columns Count:** {len(df_data.columns)}')
    st.write(f'**Data Columns:** {list(df_data.columns)}')
    st.write('**Data Types:**', df_data.dtypes)
    st.write('**Data Describe:**', df_data.describe().T)
    st.write('**Data Info:**', df_data.info())
    st.write('**Data Head:**', df_data.head())
    st.write('**Data Tail:**', df_data.tail())


# @st.cache
def data_profiling(file_path: str, report_path: str):
    report_files = ['ProfilingReport.html']
    report_files_exist = all(os.path.isfile(os.path.join(report_path, file)) for file in report_files)
    if not report_files_exist:
        df_data = pd.read_csv(file_path)
        profile = ProfileReport(df=df_data, title="Profiling Report", minimal=True)
        profile.to_file(os.path.join(report_path, "ProfilingReport.html"))

    # return profile.to_widgets()
    # return profile.profile.to_notebook_iframe()
