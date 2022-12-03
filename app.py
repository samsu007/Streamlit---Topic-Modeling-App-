import streamlit as st
from top2vec import Top2Vec

path = "model/topic_model.pth"


def get_model(path):
    model = Top2Vec.load(path)
    return model


def get_docs(model, entity):
    docs = []
    scores = []
    documents, document_scores, document_ids = model.search_documents_by_keywords(keywords=[entity], num_docs=3)
    for doc, score, doc_id in zip(documents, document_scores, document_ids):
        docs.append(doc)
        scores.append(score)
    return docs, scores


st.title("Topic Modelling")

entity = st.text_input("search keywords ...")
st.caption("Try with these keywords :")
st.write("btc, bitcoin, xrp, ethereum, alameda, blockfi, ftx, blockchain")
model = get_model(path)

if st.button('search'):
    docs, score = get_docs(model, entity)
    st.subheader("Related Articles :")
    for i, doc in enumerate(docs):
        st.write("Articles : ", doc)
        st.write("Scores : ", score[i])
