# Shared fixtures live here. Prefer fixtures over setup/teardown; keep tests
# deterministic (see the shared Python conventions).

import os
import os.path
from urllib.request import urlretrieve

import pytest
import ray

RAY_GITHUB_REPO = "https://raw.githubusercontent.com/ray-project/ray"
GITHUB_CONFIG_PATH = "python/ray/autoscaler/_private/fake_multi_node/example.yaml"
LOCAL_CONFIG_PATH = "autoscaler/_private/fake_multi_node/example.yaml"


@pytest.fixture(scope="session", autouse=True)
def install_example_yaml() -> None:
  """Install the example config needed by AutoscalingCluster.

  Pinned to the installed Ray version so the YAML schema can't drift away
  from what the local Ray expects.
  """
  url = f"{RAY_GITHUB_REPO}/refs/tags/ray-{ray.__version__}/{GITHUB_CONFIG_PATH}"
  file_path = os.path.join(os.path.dirname(ray.__file__), LOCAL_CONFIG_PATH)
  if not os.path.exists(file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    urlretrieve(url, file_path)
