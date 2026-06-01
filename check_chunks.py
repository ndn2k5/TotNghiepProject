import json
import random
chunks = [json.loads(l) for l in open('data/raw_chunks.jsonl', encoding='utf-8')]
print(f'Total chunks: {len(chunks)}')
sample = random.sample(chunks, 5)
for i, c in enumerate(sample):
    print(f"\n--- Chunk {i+1} ({c['source']}/{c['filename']}) ---")
    print(c['text'][:300])