import streamlit as st


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
    


if __name__ == "__main__":
    main()
