#source : https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/raptor.ipynb
import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from typing import List, Dict, Any
from sklearn.mixture import GaussianMixture
from langchain.chains.llm import LLMChain
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.schema import AIMessage
from langchain.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings
from langchain.docstore.document import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.document_loaders import PyPDFLoader
import matplotlib.pyplot as plt
import logging
from dotenv import load_dotenv
from src.tools.summarize import TextSummarizer

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..'))) # Add the parent directory to the path sicnce we work with notebooks
from src.utils.helper_functions import *
#from evaluation.evalute_rag import *

# Load environment variables from a .env file
load_dotenv()

# Set the OpenAI API key environment variable
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
class RaptorRetriever:
    llm:BaseChatModel
    embeddingsModel:Embeddings
    text_summarizer:TextSummarizer
    max_levels:int = 3
    vectorStore:VectorStore
    def __init__(self, llm: BaseChatModel, embeddingsModel: Embeddings, vectorStore:VectorStore, text_summarizer: TextSummarizer, max_level=3):
        self.llm = llm
        self.embeddingsModel = embeddingsModel
        self.vectorStore=vectorStore
        self.text_summarizer = text_summarizer
        self.max_levels = max_level
        self.raptor_tree = None
        self.vectorstore = None
        self.retriever:ContextualCompressionRetriever = None

    def extract_text(self,item):
        """Extract text content from either a string or an AIMessage object."""
        if isinstance(item, AIMessage):
            return item.content
        return item

    def embed_texts(self,texts: List[str]) -> List[List[float]]:
        """Embed texts using OpenAIEmbeddings."""
        logging.info(f"Embedding {len(texts)} texts")
        return self.embeddingsModel.embed_documents([self.extract_text(text) for text in texts])

    def perform_clustering(self, embeddings: np.ndarray,n_clusters: int = 10) -> np.ndarray:
        """Perform clustering on embeddings using Gaussian Mixture Model."""
        logging.info(f"Performing clustering with {n_clusters} clusters")
        gm = GaussianMixture(n_components=n_clusters, random_state=42)
        return gm.fit_predict(embeddings)

    def visualize_clusters(self,embeddings: np.ndarray, labels: np.ndarray, level: int):
        """Visualize clusters using PCA."""
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis')
        plt.colorbar(scatter)
        plt.title(f'Cluster Visualization - Level {level}')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.show()

    def build_raptor_tree(self,texts: List[str]) -> Dict[int, pd.DataFrame]:
        """Build the RAPTOR tree structure with level metadata and parent-child relationships."""
        results = {}
        current_texts = [self.extract_text(text) for text in texts]
        current_metadata = [{"level": 0, "origin": "original", "parent_id": None} for _ in texts]
        
        for level in range(1, self.max_levels + 1):
            logging.info(f"Processing level {level}")
            
            embeddings = self.embed_texts(current_texts)
            n_clusters = min(10, len(current_texts) // 2)
            cluster_labels = self.perform_clustering(np.array(embeddings), n_clusters)
            
            df = pd.DataFrame({
                'text': current_texts,
                'embedding': embeddings,
                'cluster': cluster_labels,
                'metadata': current_metadata
            })
            
            results[level-1] = df
            
            summaries = []
            new_metadata = []
            for cluster in df['cluster'].unique():
                cluster_docs = df[df['cluster'] == cluster]
                cluster_texts = cluster_docs['text'].tolist()
                cluster_metadata = cluster_docs['metadata'].tolist()
                summary = self.text_summarizer.summarize_texts(cluster_texts)
                summaries.append(summary)
                new_metadata.append({
                    "level": level,
                    "origin": f"summary_of_cluster_{cluster}_level_{level-1}",
                    "child_ids": [meta.get('id') for meta in cluster_metadata],
                    "id": f"summary_{level}_{cluster}"
                })
            
            current_texts = summaries
            current_metadata = new_metadata
            
            if len(current_texts) <= 1:
                results[level] = pd.DataFrame({
                    'text': current_texts,
                    'embedding': self.embed_texts(current_texts),
                    'cluster': [0],
                    'metadata': current_metadata
                })
                logging.info(f"Stopping at level {level} as we have only one summary")
                break
        
        return results

    def build_vectorstore(self,tree_results: Dict[int, pd.DataFrame]) -> VectorStore:
        """Build a vectorstore from all texts in the RAPTOR tree."""
        all_texts = []
        all_embeddings = []
        all_metadatas = []
        
        for level, df in tree_results.items():
            all_texts.extend([str(text) for text in df['text'].tolist()])
            all_embeddings.extend([embedding.tolist() if isinstance(embedding, np.ndarray) else embedding for embedding in df['embedding'].tolist()])
            all_metadatas.extend(df['metadata'].tolist())
        
        logging.info(f"Building vectorstore with {len(all_texts)} texts")
        
        # Create Document objects manually to ensure correct types
        documents = [Document(page_content=str(text), metadata=metadata) 
                    for text, metadata in zip(all_texts, all_metadatas)]
        
        return self.vectorStore.from_documents(documents, embeddingsModel)

    def build(self, texts: List[str]) -> None:
        """Build the RAPTOR tree and vectorstore."""
        self.raptor_tree = self.build_raptor_tree(texts)
        self.vectorstore = self.build_vectorstore(self.raptor_tree)
        self.create_retriever()
        
    def tree_traversal_retrieval(self,query: str,k: int = 3) -> List[Document]:
        """Perform tree traversal retrieval."""
        query_embedding = self.embeddingsModel.embed_query(query)
        
        def retrieve_level(level: int, parent_ids: List[str] = None) -> List[Document]:
            if parent_ids:
                docs = vectorstore.similarity_search_by_vector_with_relevance_scores(
                    query_embedding,
                    k=k,
                    filter=lambda meta: meta['level'] == level and meta['id'] in parent_ids
                )
            else:
                docs = vectorstore.similarity_search_by_vector_with_relevance_scores(
                    query_embedding,
                    k=k,
                    filter=lambda meta: meta['level'] == level
                )
            
            if not docs or level == 0:
                return docs
            
            child_ids = [doc.metadata.get('child_ids', []) for doc, _ in docs]
            child_ids = [item for sublist in child_ids for item in sublist]  # Flatten the list
            
            child_docs = retrieve_level(level - 1, child_ids)
            return docs + child_docs
        
        max_level = max(doc.metadata['level'] for doc in vectorstore.docstore.values())
        return retrieve_level(max_level)

    def create_retriever(self) -> None:
        """Create a retriever with contextual compression."""
        logging.info("Creating contextual compression retriever")
        base_retriever = self.vectorstore.as_retriever()
        
        prompt = ChatPromptTemplate.from_template(
            "Given the following context and question, extract only the relevant information for answering the question:\n\n"
            "Context: {context}\n"
            "Question: {question}\n\n"
            "Relevant Information:"
        )
        
        extractor = LLMChainExtractor.from_llm(self.llm, prompt=prompt)
        
        self.retriever = ContextualCompressionRetriever(
            base_compressor=extractor,
            base_retriever=base_retriever
        )

    def hierarchical_retrieval(self,query: str) -> List[Document]:
        """Perform hierarchical retrieval starting from the highest level, handling potential None values."""
        all_retrieved_docs = []
        
        for level in range(self.max_levels, -1, -1):
            # Retrieve documents from the current level
            level_docs = self.retriever.get_relevant_documents(
                query,
                filter=lambda meta: meta['level'] == level
            )
            all_retrieved_docs.extend(level_docs)
            
            # If we've found documents, retrieve their children from the next level down
            if level_docs and level > 0:
                child_ids = [doc.metadata.get('child_ids', []) for doc in level_docs]
                child_ids = [item for sublist in child_ids for item in sublist if item is not None]  # Flatten and filter None
                
                if child_ids:  # Only modify query if there are valid child IDs
                    child_query = f" AND id:({' OR '.join(str(id) for id in child_ids)})"
                    query += child_query
        
        return all_retrieved_docs

    def query(self,query: str) -> Dict[str, Any]:
        """Process a query using the RAPTOR system with hierarchical retrieval."""
        logging.info(f"Processing query: {query}")
        
        relevant_docs = self.hierarchical_retrieval(query)
        
        doc_details = []
        for i, doc in enumerate(relevant_docs, 1):
            doc_details.append({
                "index": i,
                "content": doc.page_content,
                "metadata": doc.metadata,
                "level": doc.metadata.get('level', 'Unknown'),
                "similarity_score": doc.metadata.get('score', 'N/A')
            })
        
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        prompt = ChatPromptTemplate.from_template(
            "Given the following context, please answer the question:\n\n"
            "Context: {context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )
        chain = LLMChain(llm=self.llm, prompt=prompt)
        answer = chain.run(context=context, question=query)
        
        logging.info("Query processing completed")
        
        result = {
            "query": query,
            "retrieved_documents": doc_details,
            "num_docs_retrieved": len(relevant_docs),
            "context_used": context,
            "answer": answer,
            "model_used": llm.model_name,
        }
        
        return result


def print_query_details(result: Dict[str, Any]):
    """Print detailed information about the query process, including tree level metadata."""
    print(f"Query: {result['query']}")
    print(f"\nNumber of documents retrieved: {result['num_docs_retrieved']}")
    print(f"\nRetrieved Documents:")
    for doc in result['retrieved_documents']:
        print(f"  Document {doc['index']}:")
        print(f"    Content: {doc['content'][:100]}...")  # Show first 100 characters
        print(f"    Similarity Score: {doc['similarity_score']}")
        print(f"    Tree Level: {doc['metadata'].get('level', 'Unknown')}")
        print(f"    Origin: {doc['metadata'].get('origin', 'Unknown')}")
        if 'child_docs' in doc['metadata']:
            print(f"    Number of Child Documents: {len(doc['metadata']['child_docs'])}")
        print()
    
    print(f"\nContext used for answer generation:")
    print(result['context_used'])
    
    print(f"\nGenerated Answer:")
    print(result['answer'])
    
    print(f"\nModel Used: {result['model_used']}")

if __name__ == "__main__":
    # Example usage
    path = r'C:\Users\mmerl\projects\iahackaton\cleverbot\data\Understanding_Climate_Change.pdf'
    loader = PyPDFLoader(path)
    documents = loader.load()
    embeddingsModel = OpenAIEmbeddings()
    llm = ChatOpenAI(model_name="gpt-4o-mini")
    vectorstore = FAISS
    text_summarizer = TextSummarizer()
    max_level = 3  # Adjust based on your tree depth
    texts = [doc.page_content for doc in documents]
    raptorRetriever = RaptorRetriever(llm=llm, embeddingsModel=embeddingsModel, vectorStore=vectorstore,text_summarizer=text_summarizer, max_level=max_level)
    # Build the RAPTOR tree
    raptorRetriever.build(texts)
  
    
    query = "What is the greenhouse effect?"
    result = raptorRetriever.query(query)
    print_query_details(result)