import collections
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
                                              value=5,
                                              step=1)
    return {'na_strategy': na_strategy,
            'ml_task': ml_task,
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
    options = sidebar()

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
        st.success('Data loaded with success!')
        st.info(f'Documentation found [here]({documentation}).')
        st.write('---')
        # Data head and tail
        title = selected_dataset.at[selected_dataset_idx, 'Title']
        st.subheader(title)
        st.text('First and last 5 rows:')
        st.dataframe(data.iloc[np.r_[0:4, -4:0]])
        st.write('')  # Blank line
        # Select categorical variables
        cat_variables = st.multiselect('Are there any categorical variables?',
                                       options=data.columns)
        # Select outcome variable
        outcome = st.selectbox('What is the outcome variable?',
                               options=data.columns)

        # Set categorical configs
        cat_configs = {}
        # Categorical variables config
        if cat_variables:
            st.write('---')
            num_cat = len(cat_variables)
            st.info(f'Since you specified {num_cat} categorical variables,'
                    ' please answer the following questions:')
            st.markdown(
                """
                #### Note 1.\n
                If the variable is ordered, please select all valid* categories
                in ascending order from left to right.

                #### Note 2.\n
                *Any values **not** included in the "categories" widget will
                be considered as a missing value, which is represented as
                `pd.NA` in the dataframe.
                """
            )
            for i in range(len(cat_variables)):
                st.write('')  # Insert blank line
                cat = cat_variables[i]
                st.subheader('{}. {}'.format(i+1, cat))
                is_cat_ordered = st.radio('Is this variable ordered?',
                                          ('Yes', 'No'),
                                          index=1,
                                          key=cat)
                cats = st.multiselect('What are the variable\'s categories?',
                                      data.loc[:, cat]
                                          .unique(),
                                      key=cat)
                cat_configs[cat] = (is_cat_ordered, cats)
                st.write('')  # Insert blank line

    # Column containers for buttons
    st.write('---')
    col1, col2, col3 = st.beta_columns(3)
    # Buttons
    run_profiling = col1.button('🔬 Data profiling report')
    run_na_report = col2.button('🔎 Missing value plots')
    run_automl = col3.button('✨ Run AutoML!')
    st.write('---')
    # Data profiling
    if run_profiling:
        profile_report = ProfileReport(data, explorative=True)
        st_profile_report(profile_report)
    # Missing value analysis
    if run_na_report:
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
    # Initialise categorical variables arguments
    if run_automl:
        pass


if __name__ == "__main__":
    main()
