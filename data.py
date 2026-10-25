"""Download the authors' cue-conflict stimuli and released decisions.

Everything comes from one pinned commit of github.com/rgeirhos/texture-vs-shape,
so a rerun next year gets byte-identical images. The stimuli are licensed
CC BY 4.0 (the repository's DATASET_LICENSE, which is extracted alongside).
"""

import hashlib
import re
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

COMMIT = "ccd0c01d42c0a8a834e90ec141e7f24fa915dc23"
TARBALL_URL = f"https://codeload.github.com/rgeirhos/texture-vs-shape/tar.gz/{COMMIT}"

# SHA-256 over every stimulus (relative path and bytes, in sorted order).
# Computed from the first download (whose files also match a git clone of the
# same commit); a mismatch means the images are not the ones the results in
# this repository were produced from.
STIMULI_SHA256 = "1a28def11ac5b4bc3264beb6cf8fbb22042b088d443be95cba28e5925725ab2a"

DATA_DIR = Path(__file__).resolve().parent / "data"
SOURCE_DIR = DATA_DIR / "texture-vs-shape"
STIMULI_DIR = SOURCE_DIR / "stimuli" / "style-transfer-preprocessed-512"

# Paths inside the repository that the reproduction uses. Nothing else in the
# 163 MB tarball is written to disk.
WANTED = (
    "DATASET_LICENSE",
    "stimuli/style-transfer-preprocessed-512/",
    "raw-data/style-transfer-512-nomask-experiment/",
    "raw-data/style-transfer-512-nomask-experiment-trained-CNNs/"
    "style-transfer-512-nomask-experiment_resnet50_session-1.csv",
    "code/",  # only read by the test that compares decision rules
)

# "cat1-airplane1.png": shape cat (exemplar 1), texture airplane (exemplar 1).
FILENAME = re.compile(r"^([a-z]+)\d+-([a-z]+)\d+\.png$")


def download():
    """Fetch and extract the wanted files unless present, then verify them."""
    if not STIMULI_DIR.is_dir():
        _fetch()
    digest = stimuli_digest()
    if digest != STIMULI_SHA256:
        raise RuntimeError(f"Stimuli checksum {digest} does not match {STIMULI_SHA256}; "
                           f"delete {DATA_DIR} and rerun.")


def _fetch():
    DATA_DIR.mkdir(exist_ok=True)
    prefix = f"texture-vs-shape-{COMMIT}/"
    with tempfile.TemporaryDirectory(dir=DATA_DIR) as tmp:
        archive = Path(tmp) / "repo.tar.gz"
        print(f"Downloading {TARBALL_URL}")
        with urllib.request.urlopen(TARBALL_URL) as response, open(archive, "wb") as f:
            shutil.copyfileobj(response, f)
        staging = Path(tmp) / "extract"
        with tarfile.open(archive) as tar:
            members = [m for m in tar.getmembers()
                       if m.name.startswith(prefix)
                       and m.name[len(prefix):].startswith(WANTED)]
            # filter="data" refuses absolute paths, links out of the target
            # directory and device files: the archive is outside input.
            tar.extractall(staging, members=members, filter="data")
        # Move into place only after a complete extraction, so an interrupted
        # run leaves no half-filled folder that the is_dir() check would accept.
        shutil.move(staging / prefix.rstrip("/"), SOURCE_DIR)


def stimuli_digest():
    h = hashlib.sha256()
    for path in sorted(STIMULI_DIR.rglob("*.png")):
        h.update(path.relative_to(STIMULI_DIR).as_posix().encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def list_stimuli():
    """Return [(path, shape, texture)] for all 1,280 images, sorted by path."""
    stimuli = []
    for path in sorted(STIMULI_DIR.rglob("*.png")):
        shape, texture = FILENAME.match(path.name).groups()
        # The folder is the shape category; the filename must agree with it.
        assert shape == path.parent.name, path
        stimuli.append((path, shape, texture))
    return stimuli
