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
      4. Select an outcome variable in the chosen dataset
      5. Choose a supervised ML task to perform
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


if __name__ == "__main__":
    main()
