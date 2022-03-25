import io
from functools import partial

import pycurl
from tqdm import tqdm


def request_url_content(url: str, stream: io.IOBase, verbose: bool = True):

    def _curl_callback(pbar: tqdm, download_total: int, downloaded: int, *_):
        pbar.total = download_total
        pbar.n = downloaded
        pbar.display()

    curl = pycurl.Curl()
    curl.setopt(pycurl.URL, url)
    curl.setopt(pycurl.WRITEDATA, stream)
    if verbose:
        print(f'Downloading content of: {url}')
        pbar = tqdm(unit='byte', unit_scale=True, unit_divisor=1024, miniters=1)
        curl.setopt(pycurl.NOPROGRESS, False)
        curl.setopt(pycurl.XFERINFOFUNCTION, partial(_curl_callback, pbar))

    curl.perform()

    curl.close()
    if verbose:
        pbar.close()
