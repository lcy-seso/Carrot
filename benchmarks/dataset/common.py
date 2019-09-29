import six as _six

import os
import sys
import errno

import requests
import hashlib
import shutil

__all__ = [
    'DATA_HOME',
    'md5',
    'download',
]

DATA_HOME = os.path.expanduser('~/.cache/benchmark/dataset')


def must_mkdirs(path):
    try:
        os.makedirs(DATA_HOME)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass


must_mkdirs(DATA_HOME)


def md5file(fname):
    """ Returns the md5 checksum of the file. """
    hash_md5 = hashlib.md5()
    f = open(fname, 'rb')
    for chunk in iter(lambda: f.read(4096), b""):
        hash_md5.update(chunk)
    f.close()
    return hash_md5.hexdigest()


def download(url, module_name, md5sum, save_name=None):
    """ Download data from the Internet using the given URL.

    Args:
        url: String, URL of the data.
        module_name: String, Name of the directory that is to save the
            downloaded data.
        md5sum: String, MD5 checksum of the data.
        save_name: String, You can specify the file name of the downloaded data
            using `save_name`.

    Returns:
        Path of the data.

    Raises:
        RuntimeError: if it fails to download the data.
    """

    dirname = os.path.join(DATA_HOME, module_name)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    filename = os.path.join(
        dirname,
        url.split('/')[-1] if save_name is None else save_name)

    if os.path.exists(filename) and md5file(filename) == md5sum:
        return filename

    retry = 0
    retry_limit = 3
    while not (os.path.exists(filename) and md5file(filename) == md5sum):
        if os.path.exists(filename):
            sys.stderr.write("calculated md5: %s\nexpected md5 %s" %
                             (md5file(filename), md5sum))
        if retry < retry_limit:
            retry += 1
        else:
            raise RuntimeError(
                "Cannot download {0} within retry limit {1}".format(
                    url, retry_limit))
        sys.stderr.write("Cache file %s is not found, begin downloading %s" %
                         (filename, url))
        r = requests.get(url, stream=True)
        total_length = r.headers.get('content-length')

        if total_length is None:
            with open(filename, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        else:
            with open(filename, 'wb') as f:
                dl = 0
                total_length = int(total_length)
                for data in r.iter_content(chunk_size=4096):
                    if _six.PY2:
                        data = six.b(data)
                    dl += len(data)
                    f.write(data)
                    done = int(50 * dl / total_length)
                    sys.stderr.write(
                        "\r[%s%s]" % ('=' * done, ' ' * (50 - done)))
                    sys.stdout.flush()
    sys.stderr.write("\n")
    sys.stdout.flush()
    return filename
