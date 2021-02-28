import streamlit as st
import spacy
import pandas as pd

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


def main():
    # Configures the default settings
    st.set_page_config(page_title='streamlit-rdatasets',
                       page_icon='🔎')
    st.title('🔎🧙')
    st.title('AutoML on R datasets')
    st.subheader('MIT License')
    st.markdown(
      """
      ---
      🚀 Using the app:\n
      1. Find relevant R datasets using the searchbar
      2. Select a R dataset
      3. Press the "Data profiling report" button to perform EDA
      4. From the sidebar: Select an outcome variable in the chosen dataset
      5. From the sidebar: Choose a supervised ML task to perform
      (i.e. regression or classification)
      6. Press the "Run AutoML" button to perform AutoML and generate Python
      code for the best ML pipeline

      🦾 Tech stack:
      - Uses `spaCy`'s pre-trained Word2Vec word embeddings
      and cosine similarity
      to perform contextual search
      - Generates the data profiling report using `pandas-profiling`
      - Performs AutoML using `TPOT`
      """
    )

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
    search = st.text_input('Find relevant R datasets')
    if not(search):
        st.stop()

    # Get matches
    matches = get_matches(docs, search, nlp=nlp)
    cutoff_similarity = 0.5  # Cutoff similarity score
    cutoff_num_matches = 10  # Cutoff number of matches
    matches = pd.Series(matches)
    # Get matches above cutoff
    relevant_matches = (matches.loc[matches > cutoff_similarity]
                               .nlargest(10)
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
                                        options=[None] + relevant_matches.tolist())
    selected_dataset = rdatasets[rdatasets.index == selected_dataset_idx]
    if not(selected_dataset_idx):
        st.stop()


if __name__ == "__main__":
    main()
