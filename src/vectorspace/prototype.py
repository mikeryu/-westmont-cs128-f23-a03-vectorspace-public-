import pickle
from math import sqrt, log10
from nltk.corpus import inaugural, stopwords
from nltk.stem.snowball import SnowballStemmer


def compute_dot_product(v1, v2):  # DONE
    return sum([e1 * e2 for e1, e2 in zip(v1, v2)])


def compute_euclidean_norm(vec):  # DONE
    return sqrt(sum([x ** 2 for x in vec]))


def compute_cosine_similarity(v1, v2):  # DONE
    denominator = (compute_euclidean_norm(v1) * compute_euclidean_norm(v2))
    return (compute_dot_product(v1, v2) / denominator) if denominator else 0.0


def compute_tf_idf(tf, df, corpus_size):  # DONE
    # https://en.wikipedia.org/wiki/Tf%E2%80%93idf
    tf_term = log10(1 + tf)
    idf_term = log10(corpus_size / (1 + df))
    return tf_term * idf_term


def compute_document_vector_tf_idf(terms_dict, document, df_dict, corpus_size):  # DONE
    vec = [0.0] * len(terms_dict)
    for term, index in terms_dict.items():
        tf = compute_term_frequency(term, document)
        tf_idf = compute_tf_idf(tf, df_dict[term], corpus_size)
        vec[index] = tf_idf
    return vec


def compute_document_frequencies(terms, corpus):  # DONE
    # assume the document and term are pre-stemmed
    df_dict = {}

    def check_membership_linear(t, doc):
        for w in doc:
            if t == w:
                return True
        return False

    for term in terms:
        df_dict[term] = sum([1 if check_membership_linear(term, doc) else 0 for doc in corpus])
    return df_dict


def compute_term_frequency(term, document):  # DONE
    # assume the document and term are pre-stemmed
    tf = 0
    for word in document:
        tf += 1 if word == term else 0
    return tf


def get_coincident_elements(v1, v2):  # DONE
    return [(e1, e2) for e1, e2 in zip(v1, v2) if e1 and e2]


def filter_words(words, stopwords_set):  # DONE
    # assume document is a list of words
    return filter(lambda w: w.isalpha() and w not in stopwords_set, words)


def stem_words(words, stemmer):  # DONE
    # assume document is a list of words
    return [stemmer.stem(w) for w in words]


print("GO!")

sb_stemmer = SnowballStemmer('english')
stopwords_set = set(stopwords.words('english'))

inaugural_terms_dict = {}
inaugural_words_corpus = []
inaugural_df_dict = {}
term_document_matrix = []

try:
    with open("../data/inaugural_corpus_data.inst.pkl", 'rb') as pickle_file:
        inaugural_terms_dict, inaugural_words_corpus, inaugural_df_dict, term_document_matrix = pickle.load(pickle_file)
        print("Pickle file found, loaded inaugural_corpus_data.inst.pkl")
except FileNotFoundError:
    print("Pickle file not found, creating ../data/inaugural_corpus_data.inst.pkl")
    inaugural_unique_words = set(stem_words(inaugural.words(), sb_stemmer))
    inaugural_sorted_words = sorted(filter_words(inaugural_unique_words, stopwords_set))

    inaugural_terms_dict = {k: v for (k, v) in zip(inaugural_sorted_words, range(len(inaugural_sorted_words)))}

    inaugural_words_corpus = [stem_words(filter_words(inaugural.words(file_id), stopwords_set), sb_stemmer)
                              for file_id in inaugural.fileids()]

    inaugural_df_dict = compute_document_frequencies(inaugural_terms_dict.keys(), inaugural_words_corpus)

    for index, document in enumerate(inaugural_words_corpus):
        _ = compute_document_vector_tf_idf(inaugural_terms_dict, document,
                                           inaugural_df_dict, len(inaugural_words_corpus))
        term_document_matrix.append(_)
        print(f"Done with document #{index + 1} (out of {len(inaugural_words_corpus)})")

    with open("../data/inaugural_corpus_data.inst.pkl", "wb") as pickle_file:
        data = (inaugural_terms_dict, inaugural_words_corpus, inaugural_df_dict, term_document_matrix)
        pickle.dump(data, pickle_file)

corpus_size = len(inaugural_words_corpus)

query_document = "war and drug and death life terrorism victory glorious"
query_document_words = stem_words(filter_words(query_document.split(), stopwords_set), sb_stemmer)
query_vector = compute_document_vector_tf_idf(inaugural_terms_dict, query_document_words, inaugural_df_dict, corpus_size)

titles = inaugural.fileids()
result = []

for index, doc_vector in enumerate(term_document_matrix):
    result.append((titles[index], compute_cosine_similarity(doc_vector, query_vector)))

result.sort(key=lambda x: x[1], reverse=True)

print(f"Searching in {corpus_size} documents ...")
print(f"Query  : {query_document}")
print(f"Result :\n    " + "\n    ".join(map(str, result[:10])))

