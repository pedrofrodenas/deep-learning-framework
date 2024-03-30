import argparse
import os
import tarfile

from download import download

_DOWNLOAD_URLS = [
    ('http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
        '34ed68851bce2a36e2a223fa52c661d592c66b3c'),
    ('http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar',
        '41a8d6e12baa5ab18ee7f8f8029b9e11805b4ef1'),
    ('http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
        '4e443f8a2eca6b1dac8a6c57641b67dd40621a49')]

def get_args():
    """Define the task arguments with the default values.
    """
    args_parser = argparse.ArgumentParser()
    
    args_parser.add_argument(
        '--dst-folder',
        help='path to place downloaded VOC dataset',
        type=str,
        required=True)
    return args_parser.parse_args()

def main():

    args = get_args()
    dst_folder = args.dst_folder

    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    for url, checksum in _DOWNLOAD_URLS:
        filename = download(url, path=dst_folder, overwrite=False, sha1_hash=checksum)
        with tarfile.open(filename) as tar:

            tar.extractall(dst_folder)



if __name__ == '__main__':
    main()