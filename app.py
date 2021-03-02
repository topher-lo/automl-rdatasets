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
from src.automl import clean_data
from src.automl import encode_data


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
    # Choose supervised ML task type
    ml_task = st.sidebar.selectbox('Is AutoML being used for a supervised'
                                   ' classification or regression problem?',
                                   ('Classification', 'Regression'))
    scoring = st.sidebar.selectbox('Which metric is used to evaluate models?',
                                   ('Accuracy rate',
                                    'Area under ROC curve',
                                    'Mean squared error',
                                    'F1 score'))
    scoring_name_code = {
        'Accuracy rate': 'accuracy',
        'Area under ROC curve': 'roc_auc',
        'Mean squared error': 'neg_mean_squared_error',
        'F1 score': 'f1'
    }
    scoring_code = scoring_name_code[scoring]
    test_size = st.sidebar.slider('What percentage of the data'
                                  ' is in the test set?',
                                  min_value=0.0,
                                  max_value=1.0,
                                  value=0.25,
                                  step=0.01)
    train_size = st.sidebar.slider('What percentage of the data'
                                   ' is in the training set?',
                                   min_value=0.0,
                                   max_value=1.0,
                                   value=0.75,
                                   step=0.01)
    generations = st.sidebar.slider('How many iterations should the'
                                    ' pipeline optimisation process'
                                    ' run for?',
                                    min_value=1,
                                    max_value=20,
                                    value=5,
                                    step=1)
    max_time = st.sidebar.slider('Maximum running time'
                                 ' (in minutes) to'
                                 ' optimise pipeline',
                                 min_value=1,
                                 max_value=20,
                                 value=5,
                                 step=1)
    max_eval_time = st.sidebar.slider('Maximum running time to'
                                      ' (in minutes) to evaluate each pipeline',
                                      min_value=1,
                                      max_value=10,
                                      value=2,
                                      step=1)
    pop_size = st.sidebar.number_input('How many observations should be'
                                       ' retained in the genetic programming'
                                       ' population for each iteration?',
                                       min_value=0,
                                       value=50,
                                       step=1)
    automl_config = {
        'generations': generations,
        'population_size': pop_size,
        'scoring': scoring_code,
        'max_time_mins': max_time,
        'max_eval_time_mins': max_eval_time
    }
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
    return {'ml_task': ml_task,
            'test_size': test_size,
            'train_size': train_size,
            'automl_config': automl_config,
            'cutoff_similarity': cutoff_similarity,
            'max_num_matches': max_num_matches}


def main():
    # Configures the default settings
    st.set_page_config(page_title='automl-rdatasets',
                       page_icon='🔎')

    # Sidebar
    options = sidebar()

    # Title
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
      5. From the sidebar, choose a supervised ML task to perform
      (i.e. regression or classification)
      6. Press the "Run AutoML" button to perform AutoML and generate Python
      code for the best ML pipeline
      """
    )
    st.write('')

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
    st.write('')
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
        col1, col2 = st.beta_columns(2)
        col1.success('Dataset loaded with success!')
        col2.info(f'Documentation found [here]({documentation}).')
        # Data head and tail
        title = selected_dataset.at[selected_dataset_idx, 'Title']
        st.subheader(title)
        st.text('First and last 5 rows:')
        st.dataframe(data.iloc[np.r_[0:4, -4:0]])
        # Model specs
        st.write('---')  # Divider
        st.header('Specify your model')
        st.write('')  # Blank line
        with st.beta_expander('View instructions'):
            st.markdown(
                """
                #### Note 1.\n
                `automl-rdatasets` automatically converts textual columns
                into unordered categorical variables.

                #### Note 2.\n
                You **do not** have to specify the categorical variables if the
                following conditions are met:\n
                1. Each column with `category` dtype contains only strings\*
                or only numeric values\*
                3. If the categorical variable is ordered, the variable's order
                follows the alphanumeric order of the column's values

                #### Note 3.\n
                *`automl-rdatasets` accepts missing values recognised
                by `pandas`.
                """
            )
        st.write('')  # Blank line
        st.write('')  # Blank line
        # Select categorical variables
        cat_variables = st.multiselect('Are there any categorical variables?',
                                       options=data.columns)
        # Select outcome variable
        outcome = st.selectbox('What is the outcome variable?',
                               options=data.columns)

        # Set categorical configs
        cats_config = {}
        # Categorical variables config
        if cat_variables:
            st.write('---')
            num_cat = len(cat_variables)
            st.header('Categorical variables')
            with st.beta_expander('View instructions'):
                st.markdown(
                    """
                    #### Note 1.\n
                    Since you declared 1 or more categorical variables,
                    `automl-rdatasets` needs to know:\n
                    1. Whether the variable is unordered or ordered
                    2. The variable's categories
                    3. (If ordered) the categories' order

                    #### Note 2.\n
                    If the variable is ordered, please select all valid*
                    categories in ascending order from left to right.

                    #### Note 3.\n
                    *Any values **not** included in the "categories" widget
                    will be considered as a missing value, which is
                    represented as `pd.NA` in the dataframe.
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
                cats = st.multiselect('What are this variable\'s categories?',
                                      data.loc[:, cat]
                                          .unique(),
                                      key=cat)
                cats_config[cat] = (is_cat_ordered, cats)

    # Column containers for buttons
    st.write('---')  # Divider
    st.header('Run analysis')
    st.write('')  # Blank line
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
        is_ordered = [k for k, v in cats_config.items() if v[0] == 'Yes']
        categories = {k: v[1] for k, v in cats_config.items()}
        # Clean data
        cleaned_data = clean_data(data,
                                  is_factor=cat_variables,
                                  is_ordered=is_ordered,
                                  categories=categories)
        # Encoded data
        encoded_data = encode_data(cleaned_data)
        # Model data
        ml_task = options.get('ml_task')
        test_size = options.get('test_size')
        train_size = options.get('train_size')
        automl_config = options.get('automl_config')
        automl_code = run_automl(encoded_data,
                                 outcome,
                                 ml_task,
                                 train_size,
                                 test_size,
                                 **automl_config)
        # Display code for best ML pipeline found
        st.markdown(
            """
            ```python
            {}
            ```
            """.format(automl_code)
        )


if __name__ == "__main__":
    main()
