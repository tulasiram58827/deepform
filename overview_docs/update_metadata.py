import requests, json, base64

apiToken = '<api token>' + ':x-auth-token' #add API token
encoded_u = base64.b64encode(apiToken.encode()).decode()
headers = {'Content-Type': 'application/json',
            "Authorization" : "Basic %s" % encoded_u}

documentSetId = <document set id> #22186 is the sample document set
doc_id = <document id>

url = 'https://www.overviewdocs.com/api/v1/document-sets/{documentSetId}/documents/{doc_id}'.format(
    documentSetId = documentSetId, doc_id = doc_id)

values = """
  {
    "metadata": {
      "model_gross_amount": "<value for gross amount>"
    }
  }
"""

r_update = requests.patch(meta_url, headers=headers, data=values)
