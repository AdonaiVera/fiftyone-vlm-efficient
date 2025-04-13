import os

os.environ['FIFTYONE_ALLOW_LEGACY_ORCHESTRATORS'] = 'true'

import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types    

from .caption_operator import CaptionWithFlorence2
from .sieve_operator import SieveDatasetPruning
from .grounding_operator import CaptionToPhraseGroundingWithFlorence2

def register(plugin):
    """Register operators with the plugin."""
    # Register individual task operators
    plugin.register(CaptionWithFlorence2)
    plugin.register(SieveDatasetPruning)