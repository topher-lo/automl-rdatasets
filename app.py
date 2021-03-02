import streamlit as st
import spacy
import pandas as pd
import numpy as np
import missingno as msno

from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

from src.utils import load_data
from src.search import spacy_preprocessor
from src.search import spacy_similarity


# Load a blank spaCy model
# Used to access spaCy dtypes as inputs to hash_func in st.cache(...)
spacy.blank('en')


@st.cache(allow_output_mutation=True,
          suppress_st_warning=True,
          show_spinner=False)
def load_model(model: str) -> spacy.language.Language:
    """Load spaCy model.
    """
    return spacy.load(model)


@st.cache(hash_funcs={spacy.vocab.Vocab: id,
                      spacy.lang.en.English: id},
          allow_output_mutation=True,
          show_spinner=False)
def get_docs(*args, **kwargs):
    """Returns list of processed spaCy Doc objects.
    """
    docs = spacy_preprocessor(*args, **kwargs)
    return docs


@st.cache(hash_funcs={spacy.tokens.doc.Doc: id,
                      spacy.vocab.Vocab: id,
                      spacy.lang.en.English: id},
          allow_output_mutation=True,
          show_spinner=False)
def get_matches(*args, **kwargs):
    """Returns list of cosine similarity scores.
    """
    return spacy_similarity(*args, **kwargs)


def sidebar(data=None):
    # AutoML
    st.sidebar.header('AutoML settings')
    # Select missing value strategy
    na_strategy = st.sidebar.selectbox('How should missing values be handled?',
                                       ('Missing indicator',
                                        'MICE',
                                        'Complete case'))
    # Choose supervised ML task type
    ml_task = st.sidebar.selectbox('Is AutoML being used for a supervised'
                                   ' classification or regression problem?',
                                   ('Classification', 'Regression'))
    # Select categorical variables
    if data:
        columns = data.columns.tolist()
    else:
        columns = []
    is_factor = st.sidebar.multiselect('Are there any categorical variables?',
                                       options=columns)
    # Select outcome variable
    outcome = st.sidebar.selectbox('What is the outcome variable?',
                                   options=columns)
    # Contextual search
    st.sidebar.header('Search options')
    # Cutoff similarity score
    cutoff_similarity = st.sidebar.number_input('Minimum cutoff for cosine'
                                                ' similarity between query'
                                                ' and R dataset description',
                                                min_value=0.00,
                                                value=0.50,
                                                step=0.01)
    # Maximum number of matches
    max_num_matches = st.sidebar.number_input('Maximum number of'
                                              ' matches shown',
                                              min_value=1,
                                              value=10,
                                              step=1)
    return {'na_strategy': na_strategy,
            'ml_task': ml_task,
            'is_factor': is_factor,
            'outcome': outcome,
            'cutoff_similarity': cutoff_similarity,
            'max_num_matches': max_num_matches}


def main():
    # Configures the default settings
    st.set_page_config(page_title='automl-rdatasets',
                       page_icon='🔎')
    st.title('🔎 AutoML on R datasets 🧙')
    st.subheader('MIT License')
    st.markdown(
      """
      ---
      🚀 Using the app:\n
      1. Find relevant R datasets using the searchbar
      2. Select a R dataset
      3. Press the "Data profiling report" button or "Missing value plots"
      to perform EDA
      4. Select an outcome variable in the chosen dataset
      5. Choose a supervised ML task to perform
      (i.e. regression or classification)
      6. Press the "Run AutoML" button to perform AutoML and generate Python
      code for the best ML pipeline
      """
    )

    # Sidebar
    data = None
    options = sidebar(data)

    # Pretrained NLP model
    model = 'en_core_web_md'
    nlp = load_model(model)

    # R datasets search bar
    rdatasets = load_data('https://raw.githubusercontent.com/'
                          'vincentarelbundock/Rdatasets/'
                          'master/datasets.csv')

    # Run spaCy processing pipeline on R datasets meta-data
    with st.spinner('Processing R datasets table...'):
        docs = get_docs(rdatasets['Title'].tolist(), nlp)

    # Search bar
    search = st.text_input('Find relevant R datasets ordered by'
                           ' cosine similarity')
    if not(search):
        st.stop()

    # Get matches
    matches = get_matches(docs, search, nlp=nlp)
    matches = pd.Series(matches)
    # Get matches above cutoff
    relevant_matches = (matches.loc[matches > options.get('cutoff_similarity')]
                               .nlargest(options.get('max_num_matches'))
                               .index)
    relevant_cols = ['Package', 'Item', 'Title', 'Rows', 'Cols']
    rdatasets_matches = (rdatasets.loc[rdatasets.index.isin(relevant_matches),
                                       relevant_cols]
                                  .reindex(relevant_matches))
    if len(rdatasets_matches) > 0:
        st.table(rdatasets_matches)
    else:
        st.warning('No relevant datasets')
        st.stop()

    # Select dataset
    selected_dataset_idx = st.selectbox('Select a dataset by its '
                                        'index in the table above',
                                        options=[None] +
                                        relevant_matches.tolist())
    selected_dataset = rdatasets[rdatasets.index == selected_dataset_idx]
    if not(selected_dataset_idx):
        st.stop()

    # Load data from selected url
    with st.spinner('Loading data...'):
        url = selected_dataset['CSV'].tolist()[0]
        documentation = selected_dataset['Doc'].tolist()[0]
        data = load_data(url=url, index_col=0).reset_index(drop=True)
        st.info('Data loaded with success!')
        st.success(f'Documentation found [here]({documentation}).')
        st.write('---')
        # Data head and tail
        title = selected_dataset.at[selected_dataset_idx, 'Title']
        st.subheader(title)
        st.text('First and last 5 rows:')
        st.table(data.iloc[np.r_[0:4, -4:0]])

    # Column containers for buttons
    col1, col2, col3 = st.beta_columns(3)
    # Data profiling
    if col1.button('🔬 Data profiling report'):
        profile_report = ProfileReport(data, explorative=True)
        st_profile_report(profile_report)
    # Missing value analysis
    if col2.button('🔎 Missing value plots'):
        # Check if there are any missing values
        if pd.notna(data).all().all():
            st.warning('No missing values in dataset')
        else:
            fig1 = msno.matrix(data).get_figure()
            st.pyplot(fig1)
            fig2 = msno.heatmap(data).get_figure()
            st.pyplot(fig2)
            fig3 = msno.dendrogram(data).get_figure()
            st.pyplot(fig3)
    # Run data workflow
    if col3.button('✨ Run AutoML!'):
        cat_variables = options.get('is_factor')
        if cat_variables:
            num_cat = len(cat_variables)
            st.info(f'Since you specified {num_cat} categorical variables,'
                    ' please answer the following questions:')
            for cat in cat_variables:
                st.markdown(cat)
                st.radio('Is the variable ordered?', ('Yes', 'No'))
                st.multiselect('What are the variable\'s categories*?')
                st.text('Note 1. If the ')
                st.text('Note 2. If the variable is ordered, please select'
                        ' each category in ascending order from left to right.')


if __name__ == "__main__":
    main()
