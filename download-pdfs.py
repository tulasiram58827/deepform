import aiohttp
import asyncio
import pandas as pd
import os
# import logging
#
# log = logging.getLogger()

DEFAULT_REMOTE_STORE = 'https://documentcloud.org/documents'
DEFAULT_LOCAL_DIR = "data/pdfs"

DEFAULT_CONFIG = {
    "remote_store": DEFAULT_REMOTE_STORE,
    "local_dir": DEFAULT_LOCAL_DIR,
    "connection_limit": 5
}


def filename(document_id, config=DEFAULT_CONFIG):
    return f'{config["local_dir"]}/{document_id}.pdf'


def document_url(document_id, config=DEFAULT_CONFIG):
    return f'{config["remote_store"]}/{document_id}.pdf'


async def fetch(client, document_id, config=DEFAULT_CONFIG):
    localname = filename(document_id, config)
    if not os.path.isfile(localname):
        url = document_url(document_id, config)
        print(f'downloading {url}')
        async with await client.get(url) as response:
            if response:
                with open(localname, 'wb') as file:
                    file.write(await response.read())
    else:
        print(f'skipping {document_id}')


async def fetch_one(document_id, config=DEFAULT_CONFIG):
    with aiohttp.ClientSession() as client:
        await fetch(client, document_id, config)


async def fetch_many(document_ids, config=DEFAULT_CONFIG):
    conn = aiohttp.TCPConnector(limit=config["connection_limit"])
    async with aiohttp.ClientSession(connector=conn) as client:
        tasks = [asyncio.ensure_future(fetch(client, document_id, config))
                 for document_id in document_ids]
        await asyncio.gather(*tasks)


async def main():
    documents = pd.read_csv('source/ftf-all-filings.tsv', sep='\t')
    document_ids = [document_id for document_id in documents['dc_slug']]
    await fetch_many(document_ids[0:10])


if __name__ == "__main__":
    asyncio.run(main())
