import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path("/Users/prince/Downloads/Onboard")))

from rag.ingestor import get_vectorstore

def main():
    try:
        vs = get_vectorstore()
        raw = vs.get(include=["metadatas"])
        metadatas = raw.get("metadatas", [])
        
        if not metadatas:
            print("No documents found in the vector store database.")
            return

        unique_docs = {}
        for meta in metadatas:
            if not meta:
                continue
            file_name = meta.get("file_name", "unknown")
            source_type = meta.get("source_type", "unknown")
            
            if file_name not in unique_docs:
                unique_docs[file_name] = {
                    "source_type": source_type,
                    "chunks_count": 0
                }
            unique_docs[file_name]["chunks_count"] += 1

        print(f"Total unique documents ingested: {len(unique_docs)}")
        print("-" * 60)
        for name, info in unique_docs.items():
            print(f"📄 Name/URL: {name}")
            print(f"   Type: {info['source_type']}")
            print(f"   Chunks: {info['chunks_count']}")
            print("-" * 60)
            
    except Exception as e:
        print(f"Error querying vector store: {e}")

if __name__ == "__main__":
    main()
