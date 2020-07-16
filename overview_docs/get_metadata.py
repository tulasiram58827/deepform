import requests, json, base64
import pandas as pd

#set up api headers
apiToken = '<api token>' + ':x-auth-token' #add API token
encoded_u = base64.b64encode(apiToken.encode()).decode()
headers = {'Content-Type': 'application/json',
            "Authorization" : "Basic %s" % encoded_u}

documentSetId = <document set id> #22186 is the sample document set
limit = 1000 #set limit
docs_url = 'https://www.overviewdocs.com/api/v1/document-sets/{documentSetId}/documents?fields=fields&limit={limit}'.format(
            documentSetId = documentSetId, limit = limit)

#get all documents
r_docs = requests.get(docs_url, headers=headers)
num_docs = r_docs.json()['pagination']['total']

docs = {'doc_id': [], 'doc_title': []}

items = r_docs.json()['items']
for item in items:
    docs['doc_id'].append(item['id'])
    docs['doc_title'].append(item['title'])

#save to csv
docs_df = pd.DataFrame(docs)
docs_df.to_csv('overview_ids.csv', index=False)
