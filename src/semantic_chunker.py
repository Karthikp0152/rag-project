import re
import numpy as np
from src.embeddings import get_model
#Here we are looking to chunk based on sentence similarity rather than the traditional chunking techniques
def split_into_sentences