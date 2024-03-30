from typing import Iterator, Optional
import os
import hashlib
import urllib
from tqdm import tqdm
import tarfile

# Adapted for deep-learning-framework from torchvision project.
# https://github.com/pytorch/vision
# Copyright (c) Soumith Chintala 2016
# Licensed under BSD 3-Clause License
# you may not use this file except in compliance with the License.

USER_AGENT = "deep-learning"

def check_sha1(filename, sha1_hash):
    """Check whether the sha1 hash of the file content matches the expected hash.
    Parameters
    ----------
    filename : str
        Path to the file.
    sha1_hash : str
        Expected sha1 hash in hexadecimal digits.
    Returns
    -------
    bool
        Whether the file content matches the expected hash.
    """
    sha1 = hashlib.sha1()
    with open(filename, 'rb') as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)

    sha1_file = sha1.hexdigest()
    l = min(len(sha1_file), len(sha1_hash))
    return sha1.hexdigest()[0:l] == sha1_hash[0:l]

def _urlretrieve(url: str, filename: str, chunk_size: int = 1024 * 32) -> None:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT})) as response:
        _save_response_content(iter(lambda: response.read(chunk_size), b""), filename, length=response.length)

def _save_response_content(
    content: Iterator[bytes],
    destination: str,
    length: Optional[int] = None,
) -> None:
    with open(destination, "wb") as fh, tqdm(total=length) as pbar:
        for chunk in content:
            # filter out keep-alive new chunks
            if not chunk:
                continue

            fh.write(chunk)
            pbar.update(len(chunk))

def download_and_extract_archive(
    url: str,
    download_root: str,
    extract_folder_name: Optional[str] = None,
    extract_root: Optional[str] = None,
    filename: Optional[str] = None,
    md5: Optional[str] = None,
    remove_finished: bool = False,
    ) -> None:

    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if not filename:
        filename = os.path.basename(url)

    download_url(url, download_root, filename, md5)

    dataset_path = os.path.join(download_root, filename)

    if md5 and not check_sha1(dataset_path, md5):
        raise UserWarning('File {} is downloaded but the content hash does not match. ' \
                            'The repo may be outdated or download may be incomplete. ' \
                            'If the "repo_url" is overridden, consider switching to ' \
                            'the default repo.'.format(dataset_path))

    print(f"Extracting {dataset_path} to {extract_root}")

    if extract_folder_name and os.path.isdir(extract_folder_name):
        print("Already uncompressed dataset: " + dataset_path)
        return
    
    with tarfile.open(dataset_path) as tar:
            tar.extractall(extract_root)

def download_url(
    url: str, root: str, filename: Optional[str] = None, md5: Optional[str] = None, max_redirect_hops: int = 3
) -> None:
    """Download a file from a url and place it in root.

    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
        md5 (str, optional): MD5 checksum of the download. If None, do not check
        max_redirect_hops (int, optional): Maximum number of redirect hops allowed
    """
    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    os.makedirs(root, exist_ok=True)

    if os.path.isfile(fpath):
        print("Using downloaded file: " + fpath)
        return
    
    # expand redirect chain if needed
    url = _get_redirect_url(url, max_hops=max_redirect_hops)

    # download the file
    try:
        print("Downloading " + url + " to " + fpath)
        _urlretrieve(url, fpath)
    except (urllib.error.URLError, OSError) as e:  # type: ignore[attr-defined]
        if url[:5] == "https":
            url = url.replace("https:", "http:")
            print("Failed download. Trying https -> http instead. Downloading " + url + " to " + fpath)
            _urlretrieve(url, fpath)
        else:
            raise e
    
def _get_redirect_url(url: str, max_hops: int = 3) -> str:
    initial_url = url
    headers = {"Method": "HEAD", "User-Agent": USER_AGENT}

    for _ in range(max_hops + 1):
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as response:
            if response.url == url or response.url is None:
                return url

            url = response.url
    else:
        raise RecursionError(
            f"Request to {initial_url} exceeded {max_hops} redirects. The last redirect points to {url}."
        )
    