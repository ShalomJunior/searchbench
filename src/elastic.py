from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import time

class ElasticBM25Engine:
    """
    Production-grade BM25 retrieval using Elasticsearch.
    Replaces the pure Python BM25Engine to solve latency bottlenecks.
    """
    
    def __init__(self, index_name: str = "searchbench_corpus", host: str = "http://localhost:9200"):
        self.index_name = index_name
        self.es = Elasticsearch(host)
        
        # Ensure connection is established
        try:
            info = self.es.info()
            print(f"Connected to Elasticsearch {info['version']['number']}")
        except Exception as e:
            raise ConnectionError(f"Could not connect to Elasticsearch at {host}. Error: {e}")
            
    def fit(self, corpus: dict[str, str]) -> None:
        """
        Indexes the entire corpus into Elasticsearch.
        """
        # Delete index if it exists to start fresh
        if self.es.indices.exists(index=self.index_name):
            self.es.indices.delete(index=self.index_name)
            
        # Create index with standard English analyzer for BM25
        self.es.indices.create(
            index=self.index_name,
            body={
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "analysis": {
                        "analyzer": {
                            "default": {
                                "type": "english"
                            }
                        }
                    }
                },
                "mappings": {
                    "properties": {
                        "text": {"type": "text"}
                    }
                }
            }
        )
        
        print(f"Indexing {len(corpus)} documents into Elasticsearch...")
        
        # Prepare bulk indexing actions
        def generate_actions():
            for doc_id, text in corpus.items():
                yield {
                    "_index": self.index_name,
                    "_id": str(doc_id),
                    "_source": {"text": text}
                }
                
        # Execute bulk indexing
        success, _ = bulk(self.es, generate_actions(), chunk_size=5000)
        print(f"Successfully indexed {success} documents.")
        
        # Force refresh to make documents immediately searchable
        self.es.indices.refresh(index=self.index_name)
        
    def search(self, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        """
        Retrieves top_k documents using Elasticsearch's native BM25.
        Returns a list of (doc_id, score) tuples.
        """
        response = self.es.search(
            index=self.index_name,
            body={
                "query": {
                    "match": {
                        "text": query
                    }
                },
                "size": top_k,
                "_source": False # We only need IDs and scores, not the full text
            }
        )
        
        results = []
        for hit in response["hits"]["hits"]:
            results.append((hit["_id"], hit["_score"]))
            
        return results
