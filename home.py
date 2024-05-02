import streamlit as st
import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.tokenize import MWETokenizer

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def do_stuff_on_page_load():
    st.set_page_config(layout="wide")

do_stuff_on_page_load()

st.markdown('''
<style>
.appview-container .main .block-container{{
        padding-top: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;    }}
</style>
''', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('# For each of the NLTK syntactic tags, define your own syntactic tag:')
    nltkTags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', '.', ',', '$']
    corpusTags = ['Con', 'Q', 'D', 'Ex', 'Fw', 'P', 'A', 'A', 'A', 'Ls', 'Mod', 'N', 'N', 'N', 'N', 'D-Q', 'Pos', 'Pron', 'PosPron', 'Adv', 'Adv', 'Adv', 'Part', 'To-Inf', 'Intj', 'V', 'V', 'V', 'V', 'V', 'V', 'WhD', 'WhPron', 'WhPos', 'WhAdv', '.', ',', '$']
    tag_map = {}
    for i in range(len(nltkTags)):
        tag_map[nltkTags[i]] = st.text_input(nltkTags[i], corpusTags[i])

# Dictionary mapping common contractions to their expanded forms
contractions_dict = {
    "I'm": "I am",
    "you're": "you are",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "we're": "we are",
    "they're": "they are",
    "can't": "cannot",
    "don't": "do not",
    "won't": "will not",
    "shouldn't": "should not",
    "haven't": "have not",
    "isn't": "is not",
    "weren't": "were not",
    "didn't": "did not",
    "wouldn't": "would not",
    "'t": "not"  # Treating 't as a negative marker
}

st.title('HASPNeL Syntactic Tagger')
st.markdown('***')

st.markdown("The objective of this web app is to transform utterances in English as a list of strings into strings with the structure: <word>|category for each word of each utterance. The categorization is based on the Python's library, NLTK.")
st.markdown('### Step 1. Open the sidebar on the left to define your own syntactic categories.')
col1, col2 = st.columns([6,2])
with col1:
    col1.markdown('### Step 2. Define your utterances or upload a .csv file with the following format ->')
with col2:
    with open('data/utterances.csv') as f:
        col2.download_button('Download Format CSV', f, 'utterances.csv')

option = st.selectbox(
    '',
    ('Define', 'Upload'))

if option == 'Define':
    if 'data' not in st.session_state:
        data = pd.DataFrame({'utterance':[]})
        st.session_state.data = data

    data = st.session_state.data

    def add_dfForm():
        row = pd.DataFrame({'utterance':[st.session_state.input_colA]})
        st.session_state.data = pd.concat([st.session_state.data, row], ignore_index=True)

    dfForm = st.form(key='dfForm')
    with dfForm:
        dfColumns = st.columns(1)
        with dfColumns[0]:
            st.text_input('Enter utterances to add them in the dataframe. Reload page to reset.', key='input_colA')
        st.form_submit_button(on_click=add_dfForm)
        
    st.dataframe(data)
else:
    uploaded_file = st.file_uploader("Choose a file")
    try:
        data = pd.read_csv(uploaded_file)
        st.dataframe(data)
    except:
        pass

st.markdown('### Step 3. Push the button to see and download results.')

if st.button('Process'):
    utt = data.iloc[:,0].values
    taggedUtt = []
    mwetokenizer = MWETokenizer([('ca', "n't")])  # Define multi-word expression tokenizer
    for u in utt:
        ut = ''
        tokens = word_tokenize(u)
        tokens = mwetokenizer.tokenize(tokens)  # Apply multi-word expression tokenizer
        # Handle contractions
        for i in range(len(tokens)):
            token = tokens[i]
            if token in contractions_dict:
                expanded_form = contractions_dict[token]
                expanded_tokens = word_tokenize(expanded_form)
                tokens = tokens[:i] + expanded_tokens + tokens[i+1:]
        tags = nltk.pos_tag(tokens)
        for p in tags:
            if p[1] in tag_map:  # Check if the tag exists in the tag_map
                ut += p[0] + '|' + tag_map[p[1]] + ' '
            else:
                ut += p[0] + '|' + p[1] + ' '  # If not, use the original tag
        taggedUtt.append(ut)
    dft = pd.DataFrame({'utterance': utt, 'tagged': taggedUtt})
    st.dataframe(dft)

    dft.to_csv('data/utterancesTagged.csv', index=False, encoding='utf-8')  # Specify UTF-8 encoding

    with open('data/utterancesTagged.csv', 'rb') as ff:
        st.download_button('Download CSV', ff, 'utterancesTagged.csv')
