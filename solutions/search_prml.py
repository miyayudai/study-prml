import urllib.request
import json
import urllib.parse
import sys

query = urllib.parse.quote('repo:GoldenCheese/PRML-Solution-Manual "2.57"')
url = f"https://api.github.com/search/code?q={query}"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
        print(json.dumps(data, indent=2))
except Exception as e:
    print("Error:", e)
