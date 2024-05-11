import pandas as pd
import streamlit as st

# def data_preview(df_data):
#     """ Data Preview """
#     st.title('Data Preview')
#     df = pd.DataFrame(df_data)

#     return st.dataframe(df)


# def data_preview():
#     df = pd.DataFrame(np.random.randn(50, 20), columns=("col %d" % i for i in range(20)))
#     return st.dataframe(df)
#     # Same as st.write(df)

# @st.cache_data
def data_preview(file_path: str):
    st.write('Data Name:', file_path.split('\\')[-1])
    st.write('Data Path:', file_path)
    df_data = pd.read_csv(file_path)
    st.checkbox("Use container width", value=True, key="use_container_width")
    # return st.dataframe(df_data)
    return st.dataframe(df_data, use_container_width=st.session_state.use_container_width)
    # Same as st.write(df)


def data_metadata(file_path: str):
    df_data = pd.read_csv(file_path)
    st.write('Data Name:', file_path.split('\\')[-1])
    st.write('Data Path:', file_path)
    csv = df_data.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name="BodyPerformans.csv",
        mime="text/csv",
        )        
    st.write('Data Shape:', df_data.shape)
    st.write(f'Data Columns Count:{len(df_data.columns)} & List:{list(df_data.columns)}')
    st.write(f'Data Columns:{list(df_data.columns)}')
    st.write(f'Data Columns Count:{len(df_data.columns)}')
    st.write('Data Types:', df_data.dtypes)
    st.write('Data Describe:', df_data.describe().T)
    st.write('Data Info:', df_data.info())
    st.write('Data Head:', df_data.head())
    st.write('Data Tail:', df_data.tail())
    
