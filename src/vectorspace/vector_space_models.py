"""Abstract data type definitions for vector space model that supports
   cosine similarity queries using TF-IDF matrix built from the corpus.
"""

import sys
import concurrent.futures

from math import sqrt, log10
from typing import Callable, Iterable
from nltk.stem import StemmerI

__author__ = "Boaty McBoatface"
__copyright__ = "Copyright 2023, Westmont College, Mike Ryu"
__credits__ = ["Boaty McBoatface", "Mike Ryu"]
__license__ = "MIT"
__email__ = "mryu@westmont.edu"


class Vector:
    """TODO: Complete the class docstring for Vector. See Assignment 2's `orb_models.py for examples."""
    def __init__(self, elements: list[float] | None = None):
        self._vec = elements if elements else []

    def __getitem__(self, index: int) -> float:
        if index < 0 or index >= len(self._vec):
            raise IndexError(f"Index out of range: {index}")
        else:
            return self._vec[index]

    def __setitem__(self, index: int, element: float) -> None:
        if 0 <= index < len(self._vec):
            self._vec[index] = element
        else:
            raise IndexError(f"Index out of range: {index}")

    def __eq__(self, other) -> bool:
        if other is self:
            return True
        elif other is None or not isinstance(other, Vector):
            return False
        else:
            return self._vec == other.vec

    def __str__(self) -> str:
        return str(self._vec)

    @property
    def vec(self):
        return self._vec

    @staticmethod
    def _get_cannot_compute_msg(computation: str, instance: object):
        return f"Cannot compute {computation} with an instance that is not a DocumentVector: {instance}"

    def norm(self) -> float:
        """TODO: Euclidean norm of the vector."""
        return 0.0

    def dot(self, other: object) -> float:
        """TODO: Dot product of `self` and `other` vectors."""
        if not isinstance(other, Vector):
            raise ValueError(self._get_cannot_compute_msg("dot product", other))
        else:
            return 0.0

    def cossim(self, other: object) -> float:
        """TODO: Cosine similarity of `self` and `other` vectors."""
        if not isinstance(other, Vector):
            raise ValueError(self._get_cannot_compute_msg("cosine similarity", other))
        else:
            return 0.0

    def boolean_intersect(self, other: object) -> list[tuple[float, float]]:
        """Returns a list of tuples of elements where both `self` and `other` had nonzero values."""
        if not isinstance(other, Vector):
            raise ValueError(self._get_cannot_compute_msg("boolean intersection", other))
        else:
            return [(e1, e2) for e1, e2 in zip(self._vec, other._vec) if e1 and e2]


class Document:
    """TODO: Complete he class docstring for Document. See Assignment 2's `orb_models.py for examples.`"""
    _iid = 0

    def __init__(self, title: str = None, words: list[str] = None, processors: tuple[set[str], StemmerI] = None):
        Document._iid += 1
        self._iid = Document._iid
        self._title = title if title else f"(Untitled {self._iid})"
        self._words = list(words) if words else []

        if processors:
            exclude_words = processors[0]
            stemmer = processors[1]
            if not isinstance(exclude_words, set) or not isinstance(stemmer, StemmerI):
                raise ValueError(f"Invalid processor type(s): ({type(exclude_words)}, {type(stemmer)})")
            else:
                self.stem_words(stemmer)
                self.filter_words(exclude_words)

    def __iter__(self):
        return iter(self._words)

    def __eq__(self, other) -> bool:
        if other is self:
            return True
        elif other is None or not isinstance(other, Document):
            return False
        else:
            return self._title == other.title and self._words == other.words

    def __hash__(self) -> int:
        return hash((self._title, tuple(self._words)))

    def __str__(self) -> str:
        words_preview = ["["]
        preview_size = 5
        index = 0

        while index < len(self._words) and index < preview_size:
            words_preview.append(f"{self._words[index]}, ")
            index += 1
        words_preview.append("... ]")

        return "[{i:04d}]: {title} {words}".format(
            i=self._iid,
            title=self._title,
            words="".join(words_preview)
        )

    @property
    def iid(self):
        return self._iid

    @property
    def title(self):
        return self._title

    @property
    def words(self):
        return self._words

    def filter_words(self, exclude_words: set[str]) -> None:
        """TODO: Remove any words from `_words` that appear in `exclude_words` passed in."""
        self._words = self._words

    def stem_words(self, stemmer: StemmerI) -> None:
        """TODO: Stem each word in `_words` using the `stemmer` passed in."""
        self._words = self._words

    def tf(self, term: str) -> int:
        """TODO: Compute and return the term frequency of the `term` passed in among `_words`."""
        return 0


class Corpus:
    """TODO: Complete he class docstring for Document. See Assignment 2's `orb_models.py for examples.`"""
    def __init__(self, documents: list[Document], threads=1, debug=False):
        self._docs: list[Document] = documents

        # Setting flags.
        self._threads: int = threads
        self._debug: bool = debug

        # Bulk of the processing (and runtime) occurs here.
        self._terms = self._compute_terms()
        self._dfs = self._compute_dfs()
        self._tf_idf = self._compute_tf_idf_matrix()

    def __getitem__(self, index) -> Document:
        if 0 <= index < len(self._docs):
            return self._docs[index]
        else:
            raise IndexError(f"Index out of range: {index}")

    def __iter__(self):
        return iter(self._docs)

    def __len__(self):
        return len(self._docs)

    @property
    def docs(self):
        return self._docs

    @property
    def terms(self):
        return self._terms

    @property
    def dfs(self):
        return self._dfs

    @property
    def tf_idf(self):
        return self._tf_idf

    def _compute_terms(self) -> dict[str, int]:
        """TODO: Computes and returns the terms (unique, stemmed, and filtered words) of the corpus."""
        return {}  # HINT: use self._build_index_dict(...)

    def _compute_df(self, term) -> int:
        """Computes and returns the document frequency of the `term` in the context of this corpus (`self`)."""
        if self._debug:
            print(f"Started working on DF for '{term}'")
            sys.stdout.flush()

        def check_membership(t: str, doc: Document) -> bool:
            """TODO: An efficient method to check if the term `t` occurs in a list of words `doc`."""
            return False

        return sum([1 if check_membership(term, doc) else 0 for doc in self._docs])

    def _compute_dfs(self) -> dict[str, int]:
        """Computes document frequencies for each term in this corpus and returns a dictionary of {term: df}s."""
        if self._threads > 1:
            return Corpus._compute_dict_multithread(self._threads, self._compute_df, self._terms.keys())
        else:
            return {}  # HINT: Using dictionary comprehension makes this a single-liner.

    def _compute_tf_idf(self, term, doc=None, index=None):
        """TODO: Computes and returns the TF-IDF score for the term and a given document.

        An arbitrary document may be passed in directly (`doc`) or be passed as an `index` within the corpus.

        """
        dfs = self._dfs
        doc = self._get_doc(doc, index)

        # TODO: WRITE YOUR IMPLEMENTATION HERE.
        # HINT: Use `dfs` declared above to eliminate redundant calculations.

        return 0

    def compute_tf_idf_vector(self, doc=None, index=None) -> Vector:
        """Computes and returns the TF-IDF vector for the given document.

        An arbitrary document may be passed in directly (`doc`) or be passed as an `index` within the corpus.

        """
        doc = self._get_doc(doc, index)

        # TODO: WRITE YOUR IMPLEMENTATION HERE.

        return Vector()

    def _compute_tf_idf_matrix(self) -> dict[str, Vector]:
        """Computes and returns the TF-IDF matrix for the whole corpus.

        The TF-IDF matrix is a dictionary of {document title: TF-IDF vector for the document}.

        """
        def tf_idf(document):
            if self._debug:
                print(f"Processing '{document.title}'")
                sys.stdout.flush()
            vector = self.compute_tf_idf_vector(doc=document)
            return vector

        matrix = {}
        if self._threads > 1:
            matrix = Corpus._compute_dict_multithread(self._threads, tf_idf, self._docs,
                                                      lambda d: d, lambda d: d.title)
        else:
            for doc in self._docs:
                # TODO: COMPLETE THIS LOOP BODY HERE.

                if self._debug:
                    print(f"Done with doc {doc.title}")
        return matrix

    def _get_doc(self, document, index):
        """A helper function to None-guard the `document` argument and fetch documents per `index` argument."""
        if document is not None and index is None:
            return document
        elif index is not None and document is None:
            if 0 <= index < len(self):
                return self._docs[index]
            else:
                raise IndexError(f"Index out of range: {index}")

        elif document is None and index is None:
            raise ValueError("Either document or index is required")
        else:
            raise ValueError("Either document or index must be passed in, not both")

    @staticmethod
    def _compute_dict_multithread(num_threads: int, op: Callable, iterable: Iterable,
                                  op_arg_func= lambda x: x, key_arg_func=lambda x: x) -> dict:
        """Experimental generic multithreading dispatcher and collector to parallelize dictionary construction.

        Args:
            num_threads (int): maximum number of threads (workers) to utilize.
            op: (Callable): operation (function or method) to execute.
            iterable: (Iterable): iterable to call the `op` on each item.
            op_arg_func: a function that maps an item of the `iterable` to an argument for the `op`.
            key_arg_func: a function that maps an item of the `iterable` to the key to use in the resulting dict.

        Returns:
            A dictionary of {key_arg_func(an item of `iterable`): op(p_arg_func(an item of `iterable`))}.

        """
        result = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_keys = {executor.submit(op, op_arg_func(item)): key_arg_func(item) for item in iterable}
            for future in concurrent.futures.as_completed(future_to_keys):
                key = future_to_keys[future]
                try:
                    result[key] = future.result()
                except Exception as e:
                    print(f"Key '{key}' generated exception:", e, file=sys.stderr)
        return result

    @staticmethod
    def _build_index_dict(lst: list) -> dict:
        """Given a list, returns a dictionary of {item from list: index of item}."""
        return {item: index for (index, item) in enumerate(lst)}

