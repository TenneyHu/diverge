import sys
import json
import time
from ddgs import DDGS
from ddgs.exceptions import TimeoutException

def main():
    if len(sys.argv) < 3:
        print("[]")
        return

    query = sys.argv[1]
    max_results = int(sys.argv[2])

    results = []

    for _ in range(3):  
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            break  
        except Exception as e:
            time.sleep(1.0)  
            continue

    print(json.dumps(results, ensure_ascii=False))

if __name__ == "__main__":
    main()