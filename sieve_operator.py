import os
os.environ['FIFTYONE_ALLOW_LEGACY_ORCHESTRATORS'] = 'true'

import fiftyone.operators as foo
from fiftyone.operators import types
from sentence_transformers import SentenceTransformer
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch
import numpy as np
import clip
import logging
from tqdm import tqdm
import gc
from typing import List, Dict, Any, Optional, Union, Tuple

from .utils import (
    _execution_mode,
    _sieve_captioning_model_inputs,
    _sieve_alt_text_field_inputs,
    _sieve_encoder_model_inputs,
    _sieve_medium_phrases_inputs,
    _sieve_fusion_inputs,
    _sieve_k_fraction_inputs,
    _sieve_num_captions_inputs
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='sieve_operator.log'
)
logger = logging.getLogger("sieve_operator")



# Constants from SIEVE
REMOVE_PHRASES = ['an image of', "a photo of", "stock photo", "photo stock", "a photo", "an image", "image", "photo"]

# Batch processing constants
DEFAULT_BATCH_SIZE = 8
MAX_MEMORY_THRESHOLD = 0.9  

def get_device():
    """Get the appropriate device for model inference."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def clean_memory():
    """Clean up GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def batch_generator(dataset, batch_size: int):
    """Generate batches of samples from the dataset."""
    samples = list(dataset.iter_samples())  
    for i in range(0, len(samples), batch_size):
        yield samples[i:i + batch_size]

class SieveDatasetPruning(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="sieve_dataset_pruning",
            label="SIEVE Dataset Pruning",
            description="Prune dataset based on semantic similarity between image captions and alt text, with optional fusion with CLIP",
            icon="/assets/sieve-icon.svg", 
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()

        # Add all SIEVE-specific inputs
        _sieve_captioning_model_inputs(ctx, inputs)
        _sieve_alt_text_field_inputs(ctx, inputs)
        _sieve_encoder_model_inputs(ctx, inputs)
        _sieve_medium_phrases_inputs(ctx, inputs)
        _sieve_fusion_inputs(ctx, inputs)
        _sieve_k_fraction_inputs(ctx, inputs)
        _sieve_num_captions_inputs(ctx, inputs)

        # Add output fields
        inputs.str(
            "output_field",
            default="sieve_score",
            label="SIEVE Score Field",
            description="Field to store the similarity score between alt text and generated captions"
        )

        inputs.str(
            "captions_field",
            default="generated_captions",
            label="Generated Captions Field",
            description="Field to store the list of generated captions"
        )

        inputs.str(
            "selection_field",
            default="is_selected",
            label="Selection Status Field",
            description="Field to store whether the sample passed the pruning threshold"
        )

        # Add execution mode input
        _execution_mode(ctx, inputs)

        # Add batch size input
        inputs.int(
            "batch_size",
            default=8,
            label="Batch Size",
            description="Number of samples to process in each batch",
            min=1,
            max=32
        )

        return types.Property(inputs)

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)

    def execute(self, ctx):
        logger.info("Starting SIEVE dataset pruning operation")
        
        # Get parameters
        captioning_model = ctx.params.get("captioning_model", "Salesforce/blip-image-captioning-base")
        alt_text_field = ctx.params.get("alt_text_field", None)
        sentence_encoder = ctx.params.get("sentence_encoder", "sentence-transformers/all-MiniLM-L6-v2")
        medium_phrases = ctx.params.get("medium_phrases", ["an image of", "photo of", "picture of", "there is"])
        use_fusion = ctx.params.get("use_fusion", False)
        fusion_weight = ctx.params.get("fusion_weight", 0.5)
        k_fraction = ctx.params.get("k_fraction", 0.2)
        num_captions = ctx.params.get("num_captions", 4)
        output_field = ctx.params.get("output_field", "sieve_score")
        batch_size = ctx.params.get("batch_size", DEFAULT_BATCH_SIZE)
        
        logger.info(f"Parameters: captioning_model={captioning_model}, alt_text_field={alt_text_field}, "
                   f"sentence_encoder={sentence_encoder}, use_fusion={use_fusion}, "
                   f"fusion_weight={fusion_weight}, k_fraction={k_fraction}, num_captions={num_captions}")

        device = get_device()
        
        try:
            # Initialize models
            logger.info("Initializing models...")
            
            # Initialize captioning model
            if "blip" in captioning_model.lower():
                logger.info(f"Loading BLIP model: {captioning_model}")
                captioning_processor = BlipProcessor.from_pretrained(captioning_model)
                captioning_model = BlipForConditionalGeneration.from_pretrained(captioning_model)
            else:
                logger.info(f"Loading GIT model: {captioning_model}")
                captioning_processor = AutoProcessor.from_pretrained(captioning_model)
                captioning_model = AutoModelForCausalLM.from_pretrained(captioning_model)
            
            captioning_model = captioning_model.to(device)
            captioning_model.eval()
            
            # Initialize sentence encoder
            logger.info(f"Loading sentence encoder model: {sentence_encoder}")
            sentence_encoder_model = SentenceTransformer(sentence_encoder)
            sentence_encoder_model = sentence_encoder_model.to(device)
            
            # Initialize CLIP if needed
            clip_model = None
            clip_preprocess = None
            if use_fusion:
                logger.info("Loading CLIP model for fusion")
                clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
                clip_model.eval()
            
            # Process dataset in batches
            dataset = ctx.dataset
            total_samples = len(dataset)
            logger.info(f"Processing {total_samples} samples in batches of {batch_size}")
            
            with torch.no_grad():  # Disable gradient computation
                for batch in tqdm(batch_generator(dataset, batch_size), total=(total_samples + batch_size - 1) // batch_size):
                    batch_images = []
                    batch_alt_texts = []
                    batch_filepaths = []
                    
                    # Prepare batch data
                    for sample in batch:
                        try:
                            image = Image.open(sample.filepath)
                            batch_images.append(image)
                            batch_alt_texts.append(sample[alt_text_field] if alt_text_field else "")
                            batch_filepaths.append(sample.filepath)
                        except (FileNotFoundError, OSError) as e:
                            logger.warning(f"Could not open image {sample.filepath}: {str(e)}. Skipping...")
                            continue
                    
                    if not batch_images:
                        continue
                    
                    # Generate captions for batch
                    batch_captions = []
                    for image in batch_images:
                        try:
                            if "blip" in captioning_model.__class__.__name__.lower():
                                inputs = captioning_processor(image, return_tensors="pt").to(device)
                                captions = []
                                
                                # Generate multiple diverse captions using the paper's strategy
                                outputs = captioning_model.generate(
                                    **inputs,
                                    max_length=50,
                                    num_return_sequences=num_captions,  
                                    num_beams=5,
                                    do_sample=True,  # Enable sampling for diversity
                                    top_p=0.9,  # Nucleus sampling
                                    temperature=0.8,  # Temperature for diversity
                                    repetition_penalty=1.2,  # Prevent repetitive captions
                                    length_penalty=1.0  # Balanced length
                                )
                                captions = captioning_processor.batch_decode(outputs, skip_special_tokens=True)
                                
                                # Remove duplicates while maintaining order
                                seen = set()
                                captions = [x for x in captions if not (x in seen or seen.add(x))]
                                
                                # If we lost some captions due to deduplication, generate more
                                while len(captions) < num_captions:
                                    extra_output = captioning_model.generate(
                                        **inputs,
                                        max_length=50,
                                        num_return_sequences=1,
                                        do_sample=True,
                                        top_p=0.9,
                                        temperature=1.0  # Even higher temperature for more variety
                                    )
                                    extra_caption = captioning_processor.decode(extra_output[0], skip_special_tokens=True)
                                    if extra_caption not in seen:
                                        captions.append(extra_caption)
                                        seen.add(extra_caption)
                            else:
                                inputs = captioning_processor(images=image, return_tensors="pt").to(device)
                                generated_ids = captioning_model.generate(
                                    **inputs,
                                    max_length=30,
                                    num_return_sequences=num_captions * 2,
                                    num_beams=5,
                                    do_sample=False, 
                                    diversity_penalty=0.0,  
                                    num_beam_groups=1 
                                )
                                all_captions = captioning_processor.batch_decode(generated_ids, skip_special_tokens=True)
                                
                                # Remove duplicates while maintaining order
                                seen = set()
                                captions = []
                                for cap in all_captions:
                                    if len(captions) >= num_captions:
                                        break
                                    if cap not in seen:
                                        captions.append(cap)
                                        seen.add(cap)
                                
                                # If we still need more captions
                                while len(captions) < num_captions:
                                    extra_ids = captioning_model.generate(
                                        **inputs,
                                        max_length=30,
                                        num_return_sequences=1,
                                        do_sample=True,
                                        top_p=0.9,
                                        temperature=1.0
                                    )
                                    extra_caption = captioning_processor.batch_decode(extra_ids, skip_special_tokens=True)[0]
                                    if extra_caption not in seen:
                                        captions.append(extra_caption)
                                        seen.add(extra_caption)
                            
                            # Remove medium phrases
                            for phrase in medium_phrases:
                                captions = [caption.replace(phrase, "").strip() for caption in captions]
                            
                            batch_captions.append(captions)
                        except Exception as e:
                            logger.error(f"Error generating captions: {str(e)}")
                            batch_captions.append([""] * num_captions)
                    
                    # Compute SIEVE scores
                    for i, (captions, alt_text, filepath) in enumerate(zip(batch_captions, batch_alt_texts, batch_filepaths)):
                        try:
                            # Compute semantic similarity
                            caption_embeddings = sentence_encoder_model.encode(captions)
                            alt_text_embedding = sentence_encoder_model.encode([alt_text])[0]
                            
                            similarities = np.dot(caption_embeddings, alt_text_embedding) / (
                                np.linalg.norm(caption_embeddings, axis=1) * np.linalg.norm(alt_text_embedding)
                            )
                            
                            sieve_score = np.max(similarities)
                            
                            # Compute CLIP score if fusion is enabled
                            if use_fusion and clip_model is not None:
                                try:
                                    image = batch_images[i]
                                    image_input = clip_preprocess(image).unsqueeze(0).to(device)
                                    text_input = clip.tokenize([alt_text]).to(device)
                                    
                                    image_features = clip_model.encode_image(image_input)
                                    text_features = clip_model.encode_text(text_input)
                                    
                                    clip_score = torch.cosine_similarity(image_features, text_features).item()
                                    
                                    # Fuse scores
                                    sieve_score = (1 - fusion_weight) * sieve_score + fusion_weight * clip_score
                                except Exception as e:
                                    logger.warning(f"Error computing CLIP score: {str(e)}. Using only SIEVE score.")
                            
                            # Update sample with all information
                            sample = dataset[filepath]
                            captions_field = ctx.params.get("captions_field", "generated_captions")
                            selection_field = ctx.params.get("selection_field", "is_selected")
                            
                            # Store generated captions
                            sample[captions_field] = captions
                            
                            # Store SIEVE score
                            sample[output_field] = float(sieve_score)
                            
                            # Calculate threshold dynamically based on k_fraction
                            if not hasattr(self, '_score_threshold'):
                                all_scores = [s[output_field] for s in dataset if output_field in s and s[output_field] is not None]
                                if all_scores:
                                    all_scores.sort(reverse=True)
                                    threshold_idx = int(len(all_scores) * k_fraction)
                                    self._score_threshold = all_scores[threshold_idx] if threshold_idx < len(all_scores) else 0.0
                                else:
                                    self._score_threshold = 0.0
                            
                            # Mark if sample passes pruning threshold
                            sample[selection_field] = float(sieve_score) >= self._score_threshold if sieve_score is not None else False
                            sample.save()
                            
                        except Exception as e:
                            logger.error(f"Error processing sample {filepath}: {str(e)}")
                            continue
                    
                    # Clean up memory
                    clean_memory()
            
            logger.info("SIEVE dataset pruning completed successfully")
            return {"success": True}
            
        except Exception as e:
            logger.error(f"Error during SIEVE processing: {str(e)}")
            raise
        
        finally:
            # Clean up
            clean_memory()
            ctx.ops.reload_dataset()

def register(plugin):
    plugin.register(SieveDatasetPruning)
